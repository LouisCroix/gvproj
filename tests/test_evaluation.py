import ast
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image

import adaptive_p2p as ap
import evaluate_adaptive_p2p as ev


def test_mask_decoder_matches_official():
    tree = ast.parse((ap.ROOT / "vendor" / "pnp_inversion" / "evaluate.py").read_text())
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "mask_decode")
    namespace = {"np": np}
    exec(compile(ast.Module([node], type_ignores=[]), "official_mask", "exec"), namespace)
    encoded = [25, 9, 70, 14]
    np.testing.assert_array_equal(ev.decode_mask(encoded, (16, 16)), namespace["mask_decode"](encoded, [16, 16]))
    with pytest.raises(ValueError):
        ev.decode_mask([0, 1, 3])


def test_pixel_metrics_match_official_and_reset():
    adapter = ev.OfficialMetricAdapter("pixels", "cpu")
    source = Image.fromarray(np.full((32, 32, 3), 100, dtype="uint8"))
    target = Image.fromarray(np.full((32, 32, 3), 130, dtype="uint8"))
    mask = ev.decode_mask([200, 100], (32, 32))
    values = adapter.calculate(source, target, "unused", mask)
    background = np.repeat((1 - mask)[..., None], 3, axis=2)
    for name in ("psnr", "mse", "ssim"):
        metric = getattr(adapter.calculator, name + "_metric_calculator")
        metric.reset()
        expected = getattr(adapter.calculator, "calculate_" + name)(source, target, background, background)
        assert values[name + "_bg"] == pytest.approx(expected)
    adapter.calculate(source, source, "unused", mask)
    repeated = adapter.calculate(source, target, "unused", mask)
    assert values == repeated
    assert adapter.calculate(source, target, "unused", np.ones((32, 32)))["mse_bg"] is None
    assert adapter.calculate(source, target, "unused", None)["mse_bg"] is None


def test_upstream_dino_input_range_is_preserved():
    official = ev.load_official()
    calculator = official.MetricsCalculator.__new__(official.MetricsCalculator)
    calculator.device = "cpu"
    class Spy:
        def calculate_global_ssim_loss(self, source, target):
            # Official routine passes img_gt first; the distance is symmetric.
            assert source.max().item() == 128
            assert target.max().item() == 255
            return torch.tensor(.25)
    calculator.structure_distance_metric_calculator = Spy()
    a = Image.fromarray(np.full((16, 16, 3), 255, dtype="uint8"))
    b = Image.fromarray(np.full((16, 16, 3), 128, dtype="uint8"))
    assert float(calculator.calculate_structure_distance(a, b)) == .25


def test_tau_selection_and_incomplete_pairs():
    rows = []
    for tau, clip, lpips in ((.4, 30, .2), (.6, 25, .15), (.8, 20, .1)):
        for i in range(4):
            rows.append(dict(sample_id=str(i), split="dev", method="p2p", tau=tau, clip_edit=clip, lpips_bg=lpips))
    tau, _ = ev.select_tau(pd.DataFrame(rows))
    assert tau == .8  # equal rank sums -> larger tau
    with pytest.raises(ValueError):
        ev.select_tau(pd.DataFrame(rows[:-1]))
    test = pd.DataFrame([dict(sample_id=str(i), category=str(i % 2), split="test", method=m,
                             clip_edit=i + (1 if m == "soft_adaptive" else 0))
                         for i in range(4) for m in ap.METHODS] +
                        [dict(sample_id="incomplete", category="1", split="test", method="p2p", clip_edit=100)])
    stats = ev.paired_bootstrap(test, "clip_edit", "p2p", iterations=100)
    assert stats["n"] == 4 and stats["delta"] == stats["ci_low"] == stats["ci_high"] == 1


def test_manifest_deterministic_filters_and_freezes(tmp_path):
    from transformers import CLIPTokenizer
    tokenizer = CLIPTokenizer.from_pretrained(ap.ROOT / "tests" / "fixtures" / "tokenizer", local_files_only=True)
    data = tmp_path / "data"
    images = data / "annotation_images"
    images.mkdir(parents=True)
    mapping = {}
    for cat in ev.CATEGORIES:
        for i in range(5):
            name = f"{cat}_{i}.png"
            Image.new("RGB", (512, 512), (20 * i, int(cat) * 20, 80)).save(images / name)
            mapping[f"{cat}_{i}"] = dict(editing_type_id=cat, image_path=name,
                original_prompt="a red car", editing_prompt="a [blue] car", mask=[1024, 1000])
    mapping["invalid"] = dict(editing_type_id="1", image_path="1_0.png", original_prompt="a car",
                               editing_prompt="a blue car", mask=[1024, 1000])
    ap.save_json(data / "mapping_file.json", mapping)
    a = ev.build_manifest(tmp_path / "a", tokenizer, data)
    b = ev.build_manifest(tmp_path / "b", tokenizer, data)
    assert a == b
    assert len([s for s in a["samples"] if s["split"] == "dev"]) == 4
    assert len([s for s in a["samples"] if s["split"] == "test"]) == 12
    assert a["exclusions"][0]["id"] == "invalid"
    selected = next(s for s in a["samples"] if s["kind"] == "real")
    Image.new("RGB", (512, 512), "white").save(selected["case"]["source_image"])
    with pytest.raises(ValueError, match="changed"):
        ev.verify_manifest(a)


def test_budget_resume_charges_interrupted_operation(tmp_path):
    ap.save_json(tmp_path / "budget.json", {"operations": [{"name": "edit", "seconds": 12}],
                                           "in_progress": {"name": "edit", "estimate": 35}})
    budget = ev.Budget(tmp_path, limit=100, reserve=20)
    assert budget.spent == 47
    assert not budget.can_edit(30)
    assert budget.can_edit(10)
