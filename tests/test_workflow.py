"""Offline workflow integration. Deterministic fake images are not research results."""
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image
from transformers import CLIPTokenizer

import adaptive_p2p as ap
import evaluate_adaptive_p2p as ev


@pytest.mark.parametrize("use_real", [False, True])
def test_full_workflow_resume_and_repair(tmp_path, monkeypatch, use_real):
    tokenizer = CLIPTokenizer.from_pretrained(ap.ROOT / "tests" / "fixtures" / "tokenizer", local_files_only=True)
    class FakeEditor:
        revision = "a" * 40
        pipe = type("Pipe", (), {"tokenizer": tokenizer})()
        def initial_is_cached(self, case, config):
            return True
    editor = FakeEditor()
    calls = []
    def fake_edit(case, method, config):
        calls.append((case.seed, method))
        rid = ap.digest([asdict(case), method, asdict(config)])
        folder = Path(config.output_dir) / "runs" / rid
        folder.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (512, 512), (100, 120, 140)).save(folder / "source.png")
        Image.new("RGB", (512, 512), (110 + ap.METHODS.index(method), 120, 140)).save(folder / "edited.png")
        Image.new("RGB", (512, 512), (100, 120, 140)).save(folder / "source_branch.png")
        ap.save_json(folder / "weights.json", [{"u": 0., "alpha_edited": 1., "attention_js": .1},
                                              {"u": 1., "alpha_edited": 0., "attention_js": .2}])
        return {"run_id": rid, "method": method, "case": asdict(case), "config": asdict(config),
                "source_path": str(folder / "source.png"), "source_branch_path": str(folder / "source_branch.png"),
                "edited_path": str(folder / "edited.png"), "weights_path": str(folder / "weights.json"),
                "reconstruction_path": None, "edit_seconds": .01, "inversion_seconds": 0., "peak_memory_mib": 0.,
                "checksums": {name: ap.file_hash(folder / name) for name in ("source_branch.png", "edited.png", "weights.json")}}
    class FakeMetrics:
        def __init__(self, family, device):
            self.family = family
        def calculate(self, source, edited, prompt, mask):
            if self.family == "clip":
                return {"clip_whole": float(np.asarray(edited)[0, 0, 0]),
                        "clip_edit": float(np.asarray(edited)[0, 0, 0]) if mask is not None else None}
            if self.family == "structure":
                return {"structure_distance": .01}
            if self.family == "lpips":
                return {"lpips_bg": .1 if mask is not None else None}
            return {"psnr_bg": 20 if mask is not None else None,
                    "ssim_bg": .8 if mask is not None else None, "mse_bg": .01 if mask is not None else None}
    monkeypatch.setattr(ev, "get_editor", lambda config: editor)
    monkeypatch.setattr(ev, "invert_case", lambda case, config: {})
    monkeypatch.setattr(ev, "edit_case", fake_edit)
    monkeypatch.setattr(ev, "release_editor", lambda: None)
    monkeypatch.setattr(ev, "OfficialMetricAdapter", FakeMetrics)
    config = ap.EditConfig(output_dir=str(tmp_path), device="cpu", dtype="float32")
    data_root = None
    if use_real:
        data_root = tmp_path / "pie"
        images = data_root / "annotation_images"
        images.mkdir(parents=True)
        mapping = {}
        for cat in ev.CATEGORIES:
            for i in range(4):
                name = f"{cat}_{i}.png"
                Image.new("RGB", (512, 512), (int(cat) * 20, i * 30, 128)).save(images / name)
                mapping[name] = dict(editing_type_id=cat, image_path=name, original_prompt="a red car",
                                     editing_prompt="a blue car", mask=[1024, 1000])
        ap.save_json(data_root / "mapping_file.json", mapping)
    ev.run_experiment(config, data_root)
    expected = 68 if use_real else 12
    assert len(calls) == expected
    assert (tmp_path / "summary.csv").exists()
    assert len(list((tmp_path / "qualitative").glob("*/comparison.png"))) == (16 if use_real else 4)
    assert len(ev.read_json(tmp_path / "test_selection.json")["sample_ids"]) == (12 if use_real else 0)
    assert ev.read_json(tmp_path / "selected_tau.json")["tau"] == (.8 if use_real else .6)
    score = tmp_path / "blind_review" / "scores.csv"
    score.write_text(score.read_text() + "# human input\n")
    ev.run_experiment(config, data_root)
    assert len(calls) == expected
    assert "# human input" in score.read_text()
    record = next(r for r in ev.read_json(tmp_path / "registry.json")["records"] if r["split"] == "generated")
    Path(record["result"]["edited_path"]).write_bytes(b"corrupt")
    ev.run_experiment(config, data_root)
    assert len(calls) == expected + 1
    assert len(ev.read_json(tmp_path / "registry.json")["records"]) == expected


def test_initial_cache_key_excludes_edit_settings_but_checks_image(tmp_path):
    editor = ap.Editor.__new__(ap.Editor)
    editor.revision = "b" * 40
    image = tmp_path / "source.png"
    Image.new("RGB", (512, 512), "red").save(image)
    case = ap.prepare_case("a red car", "a blue car", image)
    from dataclasses import replace
    config = ap.EditConfig(output_dir=str(tmp_path), device="cpu", dtype="float32")
    identity = editor.base_identity(case, config)
    assert identity == editor.base_identity(case, replace(config, tau=.8))
    cache = tmp_path / "cache" / ap.digest(identity)
    cache.mkdir(parents=True)
    torch.save(torch.zeros(1, 4, 64, 64), cache / "initial.pt")
    Image.open(image).save(cache / "source.png")
    Image.open(image).save(cache / "reconstruction.png")
    record = {"checksums": {name: ap.file_hash(cache / name) for name in ("initial.pt", "source.png", "reconstruction.png")}}
    ap.save_json(cache / "initial.json", record)
    assert editor.initial_is_cached(case, config)
    (cache / "initial.pt").write_bytes(b"corrupt")
    assert not editor.initial_is_cached(case, config)
    assert identity != editor.base_identity(case, replace(config, steps=50))
