"""Small, paired P2P experiment; pinned PnP Inversion evaluation protocol.

No benchmark masks enter the editor. DINO preprocessing intentionally matches
the upstream evaluator (including its 0..255 float input, without /255).
"""
from __future__ import annotations

import argparse
import contextlib
import gc
import importlib.util
import json
import math
import random
import time
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageDraw

from adaptive_p2p import (ROOT, METHODS, EditConfig, digest, edit_case, environment_snapshot,
                          file_hash, get_editor, invert_case, prepare_case, release_editor,
                          save_json, validate_alignment)

PNP_COMMIT = "07f97f448150e2ca220bebd54c8f687c5c50c67a"
DINO_COMMIT = "7c446df5b9f45747937fb0d72314eb9f7b66930a"
CATEGORIES = {"1": "object", "5": "pose", "6": "color", "7": "material"}
METRICS = ("structure_distance", "psnr_bg", "lpips_bg", "mse_bg", "ssim_bg", "clip_whole", "clip_edit")
HIGHER = {"psnr_bg", "ssim_bg", "clip_whole", "clip_edit"}
SCALES = {"structure_distance": 1000, "lpips_bg": 1000, "mse_bg": 10000, "ssim_bg": 100}


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def finite_json(value):
    if isinstance(value, dict):
        return {str(k): finite_json(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [finite_json(v) for v in value]
    if isinstance(value, (float, np.floating)):
        return float(value) if math.isfinite(value) else None
    if isinstance(value, np.integer):
        return int(value)
    return value


def decode_mask(encoded, shape=(512, 512), boundary=True):
    """Upstream RLE is zero-based C-order start/length, not COCO RLE."""
    if len(encoded) % 2:
        raise ValueError("RLE must consist of start/length pairs.")
    flat = np.zeros(shape[0] * shape[1], dtype=np.float32)
    for start, length in zip(encoded[::2], encoded[1::2]):
        if int(start) != start or int(length) != length or start < 0 or length < 0 or start >= flat.size:
            raise ValueError("Invalid RLE start/length.")
        flat[int(start):min(int(start + length), flat.size)] = 1
    mask = flat.reshape(shape)
    if boundary:
        mask[0, :] = mask[-1, :] = 1
        mask[:, 0] = mask[:, -1] = 1
    return mask


def generated_samples():
    cases = [
        ("6", "a photo of a red car parked on a street", "a photo of a blue car parked on a street", 42),
        ("7", "a photo of a wooden bowl on a table", "a photo of a glass bowl on a table", 43),
        ("1", "a photo of a bicycle beside a brick wall", "a photo of a motorcycle beside a brick wall", 44),
        ("5", "a photo of a sitting dog in a grassy park", "a photo of a standing dog in a grassy park", 45),
    ]
    return [{"id": f"generated_{cat}", "category": cat, "kind": "generated", "split": "generated",
             "order": i, "case": asdict(prepare_case(src, tgt, seed=seed)), "mask": None,
             "image_sha256": None} for i, (cat, src, tgt, seed) in enumerate(cases)]


def build_manifest(output_dir, tokenizer, data_root=None):
    """Freeze eligibility and selection before looking at generated results."""
    folder = Path(output_dir).resolve()
    destination = folder / "manifest.json"
    if destination.exists():
        old = read_json(destination)
        requested = str(Path(data_root).resolve()) if data_root else None
        if old["data_root"] != requested:
            raise ValueError("Manifest is frozen. Use a new output directory when adding/changing PIE-Bench.")
        verify_manifest(old)
        return old
    samples, exclusions = generated_samples(), []
    for item in samples:
        validate_alignment(prepare_case(**item["case"]), tokenizer)
    if data_root:
        data_root = Path(data_root).resolve()
        annotations = read_json(data_root / "mapping_file.json")
        image_root = (data_root / "annotation_images").resolve()
        candidates = {cat: [] for cat in CATEGORIES}
        for sample_id, ann in sorted(annotations.items()):
            cat = str(ann["editing_type_id"])
            if cat not in CATEGORIES:
                continue
            try:
                path = (image_root / ann["image_path"]).resolve()
                if not path.is_relative_to(image_root):
                    raise ValueError("Image path escapes annotation_images.")
                case = prepare_case(ann["original_prompt"].replace("[", "").replace("]", ""),
                                    ann["editing_prompt"].replace("[", "").replace("]", ""), path)
                changed, _, _ = validate_alignment(case, tokenizer)
                if not changed:
                    raise ValueError("No textual edit.")
                with Image.open(path) as image:
                    image.load()
                    if image.size != (512, 512):
                        raise ValueError("Image must be 512x512 to match the official mask.")
                raw_mask = decode_mask(ann["mask"], boundary=False)
                if raw_mask.sum() == 0 or (1 - raw_mask).sum() == 0:
                    raise ValueError("Empty editing or non-editing region.")
                mask = decode_mask(ann["mask"])
                if (1 - mask).sum() == 0:
                    raise ValueError("No background remains after official boundary handling.")
                candidates[cat].append({"id": str(sample_id), "category": cat, "kind": "real",
                                        "case": asdict(case), "mask": ann["mask"], "image_sha256": file_hash(path)})
            except (ValueError, OSError, IndexError, KeyError) as exc:
                exclusions.append({"id": str(sample_id), "reason": str(exc)})
        rng = random.Random(2026)
        for cat, eligible in candidates.items():
            rng.shuffle(eligible)
            if len(eligible) < 4:
                raise ValueError(f"Category {cat}: requires >=4 eligible replacements; found {len(eligible)}.")
            for order, item in enumerate(eligible[:4]):
                item.update(split="dev" if order == 0 else "test", order=order - 1)
                samples.append(item)
    manifest = {"schema_version": 1, "seed": 2026,
                "data_root": str(data_root) if data_root else None,
                "samples": samples, "exclusions": exclusions,
                "test_order": "round-robin: order 0/1/2, category 1/5/6/7",
                "protocol": "replacement-only; no LocalBlend; masks evaluation-only"}
    save_json(destination, manifest)
    return manifest


def verify_manifest(manifest):
    for sample in manifest["samples"]:
        if sample["kind"] == "real" and file_hash(sample["case"]["source_image"]) != sample["image_sha256"]:
            raise ValueError(f"Source image changed after selection: {sample['id']}")


def load_official():
    path = ROOT / "vendor" / "pnp_inversion" / "matrics_calculator.py"
    provenance = read_json(ROOT / "vendor" / "reference_manifest.json")
    expected = next(x["sha256"] for x in provenance["files"] if x["path"].endswith("matrics_calculator.py"))
    if provenance["pnp_commit"] != PNP_COMMIT or file_hash(path) != expected:
        raise RuntimeError("Official metric source does not match the pinned checksum.")
    spec = importlib.util.spec_from_file_location("pinned_pnp_metrics", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    class PinnedExtractor(module.VitExtractor):
        def __init__(self, model_name, device):
            self.model = torch.hub.load(f"facebookresearch/dino:{DINO_COMMIT}", model_name,
                                        trust_repo=True).to(device).eval()
            self.model.requires_grad_(False)
            self.model_name, self.device = model_name, device
            self.hook_handlers, self.layers_dict, self.outputs_dict = [], {}, {}
            self._init_hooks_data()
    module.VitExtractor = PinnedExtractor
    return module


class OfficialMetricAdapter:
    """Call unmodified upstream methods; load only one metric model at a time."""
    def __init__(self, family, device="cuda"):
        from torchmetrics.multimodal import CLIPScore
        from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
        from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
        from torchmetrics.regression import MeanSquaredError
        self.official = load_official()
        self.calculator = self.official.MetricsCalculator.__new__(self.official.MetricsCalculator)
        self.calculator.device = device
        self.family = family
        self.models = []
        if family == "clip":
            pairs = [("clip_metric_calculator", CLIPScore(model_name_or_path="openai/clip-vit-large-patch14"))]
        elif family == "lpips":
            pairs = [("lpips_metric_calculator", LearnedPerceptualImagePatchSimilarity(net_type="squeeze"))]
        elif family == "pixels":
            pairs = [("psnr_metric_calculator", PeakSignalNoiseRatio(data_range=1.0)),
                     ("mse_metric_calculator", MeanSquaredError()),
                     ("ssim_metric_calculator", StructuralSimilarityIndexMeasure(data_range=1.0))]
        elif family == "structure":
            pairs = [("structure_distance_metric_calculator", self.official.LossG({
                "dino_model_name": "dino_vitb8", "dino_global_patch_size": 224,
                "lambda_global_cls": 10., "lambda_global_ssim": 1., "lambda_global_identity": 1.,
                "entire_A_every": 75, "lambda_entire_cls": 10, "lambda_entire_ssim": 1.}, device))]
        else:
            raise ValueError(family)
        for name, model in pairs:
            setattr(self.calculator, name, model.to(device).eval().requires_grad_(False))
            self.models.append(model)

    @torch.inference_mode()
    def calculate(self, source, edited, prompt, mask=None):
        c = self.calculator
        for model in self.models:
            if hasattr(model, "reset"):
                model.reset()
        rgbmask = np.repeat(mask[..., None], 3, axis=2) if mask is not None else None
        background = 1 - rgbmask if rgbmask is not None else None
        values = {}
        if self.family == "clip":
            values["clip_whole"] = c.calculate_clip_similarity(edited, prompt)
            c.clip_metric_calculator.reset()
            values["clip_edit"] = c.calculate_clip_similarity(edited, prompt, rgbmask) if rgbmask is not None and rgbmask.sum() else None
        elif self.family == "structure":
            values["structure_distance"] = float(c.calculate_structure_distance(source, edited))
        elif self.family == "lpips":
            values["lpips_bg"] = c.calculate_lpips(source, edited, background, background) if background is not None and background.sum() else None
        else:
            for name in ("psnr", "mse", "ssim"):
                values[name + "_bg"] = getattr(c, "calculate_" + name)(source, edited, background, background) if background is not None and background.sum() else None
        return finite_json(values)


def metric_protocol():
    return {"pnp_commit": PNP_COMMIT, "dino_commit": DINO_COMMIT,
            "reference_manifest": file_hash(ROOT / "vendor" / "reference_manifest.json"),
            "adapter_sha256": file_hash(__file__), "environment": environment_snapshot(),
            "structure_input": "upstream uint8->float32 0..255; Resize(224,max_size=480); ImageNet normalization",
            "mask": "zero-based RLE; boundaries=1; excluded pixels zeroed; no crop"}


@torch.inference_mode()
def verify_pretrained_metrics(device="cpu"):
    """Numeric adapter/upstream parity on one deterministic pair, including real weights.

    This checks evaluation implementation, not editing quality. Run before the
    T4 experiment on CPU after downloading weights, or separately on request.
    """
    grid = np.indices((128, 128)).sum(0).astype(np.uint8)
    source = Image.fromarray(np.stack([grid, np.flipud(grid), grid // 2], axis=-1))
    pixels = np.array(source).copy()
    pixels[32:96, 32:96, 0] = 220
    pixels[..., 1] = np.minimum(pixels[..., 1].astype(np.int16) + 3, 255).astype(np.uint8)
    edited = Image.fromarray(pixels)
    mask = np.zeros((128, 128), dtype=np.float32)
    mask[24:104, 24:104] = 1
    rgbmask = np.repeat(mask[..., None], 3, axis=2)
    prompt = "a colorful geometric pattern"
    report = {"device": device, "pnp_commit": PNP_COMMIT, "checks": {}}
    for family in ("pixels", "clip", "lpips", "structure"):
        print("Validating pretrained metric family:", family, flush=True)
        adapter = OfficialMetricAdapter(family, device)
        actual = adapter.calculate(source, edited, prompt, mask)
        c = adapter.calculator
        expected = {}
        if family == "clip":
            c.clip_metric_calculator.reset()
            expected["clip_whole"] = c.calculate_clip_similarity(edited, prompt)
            c.clip_metric_calculator.reset()
            expected["clip_edit"] = c.calculate_clip_similarity(edited, prompt, rgbmask)
        elif family == "structure":
            expected["structure_distance"] = float(c.calculate_structure_distance(source, edited))
        else:
            for name in (("lpips",) if family == "lpips" else ("psnr", "mse", "ssim")):
                getattr(c, name + "_metric_calculator").reset()
                expected[name + "_bg"] = getattr(c, "calculate_" + name)(source, edited, 1-rgbmask, 1-rgbmask)
        for metric, value in expected.items():
            if actual[metric] is None or not np.isclose(actual[metric], value, rtol=1e-5, atol=1e-6):
                raise AssertionError(f"Official metric parity failed: {metric}: {actual[metric]} vs {value}")
            report["checks"][metric] = {"adapter": actual[metric], "official": float(value), "passed": True}
        del c, adapter
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return report


def evaluate_results(manifest, output_dir, families=("pixels", "clip", "lpips", "structure"), device="cuda", splits=None, budget=None):
    """Resume per family/run; writes raw.csv, summary, CIs and coverage."""
    manifest = read_json(manifest) if isinstance(manifest, (str, Path)) else manifest
    output = Path(output_dir).resolve()
    registry_path = output / "registry.json"
    if not registry_path.exists():
        raise FileNotFoundError("No experiment registry. Run generation/editing first.")
    registry = read_json(registry_path)
    samples = {x["id"]: x for x in manifest["samples"]}
    records = [x for x in registry["records"] if splits is None or x["split"] in splits]
    release_editor()
    protocol = metric_protocol()
    cache_root = output / "metric_cache" / digest(protocol)
    cache_root.mkdir(parents=True, exist_ok=True)
    save_json(output / "metric_protocol.json", protocol)
    rows = {}
    for record in records:
        rid = record["result"]["run_id"]
        path = cache_root / f"{rid}.json"
        rows[rid] = read_json(path) if path.exists() else {"run_id": rid, "families": []}
    for family in families:
        pending = [r for r in records if family not in rows[r["result"]["run_id"]]["families"]]
        if not pending:
            continue
        if budget and budget.remaining < 5:
            break
        with budget.measure("metric_load_" + family) if budget else contextlib.nullcontext():
            adapter = OfficialMetricAdapter(family, device)
        for record in pending:
            if budget and budget.remaining < 2:
                break
            result = record["result"]
            sample = samples[record["sample_id"]]
            with budget.measure("metric_" + family) if budget else contextlib.nullcontext():
                with Image.open(result["source_path"]) as src, Image.open(result["edited_path"]) as dst:
                    mask = decode_mask(sample["mask"]) if sample["mask"] is not None else None
                    values = adapter.calculate(src.convert("RGB"), dst.convert("RGB"), sample["case"]["target_prompt"], mask)
            row = rows[result["run_id"]]
            row.update(values)
            row["families"].append(family)
            save_json(cache_root / f"{result['run_id']}.json", row)
        del adapter
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    # Include earlier evaluated splits when producing the final table.
    all_rows = []
    for record in registry["records"]:
        result = record["result"]
        path = cache_root / f"{result['run_id']}.json"
        if not path.exists():
            continue
        metrics = read_json(path)
        sample = samples[record["sample_id"]]
        row = {"sample_id": sample["id"], "category": sample["category"], "kind": sample["kind"],
               "split": sample["split"], "method": result["method"], "tau": result["config"]["tau"],
               "run_id": result["run_id"], "edit_seconds": result["edit_seconds"],
               "inversion_seconds": result["inversion_seconds"], "peak_memory_mib": result["peak_memory_mib"]}
        row.update({name: metrics.get(name) for name in METRICS})
        all_rows.append(row)
    frame = pd.DataFrame(all_rows, columns=["sample_id", "category", "kind", "split", "method", "tau", "run_id",
                                           "edit_seconds", "inversion_seconds", "peak_memory_mib", *METRICS])
    frame.to_csv(output / "raw_metrics.csv", index=False)
    summarize(frame, manifest, output)
    return frame


def select_tau(frame):
    dev = frame[(frame.split == "dev") & (frame.method == "p2p")].copy()
    taus = (0.4, 0.6, 0.8)
    for tau in taus:
        rows = dev[np.isclose(dev.tau, tau)]
        if len(rows) != 4 or rows[["clip_edit", "lpips_bg"]].isna().any().any():
            raise ValueError("Tau selection requires all four development cases at all three tau values.")
    means = dev.groupby("tau")[["clip_edit", "lpips_bg"]].mean()
    means["rank_sum"] = means.clip_edit.rank(ascending=False) + means.lpips_bg.rank(ascending=True)
    means = means.reset_index().sort_values(["rank_sum", "tau"], ascending=[True, False])
    return float(means.iloc[0].tau), means.to_dict("records")


def complete_pairs(frame, split, metric=None):
    data = frame[frame.split == split]
    if metric:
        data = data.dropna(subset=[metric])
    if data.empty:
        return data
    ids = data.groupby("sample_id").method.agg(lambda v: set(v) == set(METHODS))
    return data[data.sample_id.isin(ids[ids].index)]


def paired_bootstrap(frame, metric, comparator, seed=2026, iterations=2000):
    data = complete_pairs(frame, "test", metric)
    if not len(data):
        return {"n": 0, "delta": None, "ci_low": None, "ci_high": None}
    pivot = data.pivot(index=["category", "sample_id"], columns="method", values=metric)
    difference = pivot.soft_adaptive - pivot[comparator]
    rng = np.random.default_rng(seed)
    groups = [x.to_numpy() for _, x in difference.groupby(level=0)]
    means = np.empty(iterations)
    for i in range(iterations):
        means[i] = np.concatenate([rng.choice(group, len(group), replace=True) for group in groups]).mean()
    return {"n": len(difference), "delta": float(difference.mean()),
            "ci_low": float(np.quantile(means, .025)), "ci_high": float(np.quantile(means, .975)),
            "direction": "positive better" if metric in HIGHER else "negative better",
            "caution": "Exploratory; tiny strata can produce degenerate intervals. No population-level claim."}


def summarize(frame, manifest, output_dir):
    output = Path(output_dir)
    summaries, coverage = [], []
    for split in ("generated", "test"):
        paired = complete_pairs(frame, split)
        selected = [s for s in manifest["samples"] if s["split"] == split]
        coverage.append({"split": split, "planned_samples": len(selected),
                         "samples_with_all_methods": paired.sample_id.nunique(),
                         "paired_valid_n_per_metric": {metric: int(complete_pairs(frame, split, metric).sample_id.nunique())
                                                       for metric in METRICS},
                         "missing_ids": sorted(set(s["id"] for s in selected) - set(paired.sample_id))})
        for metric in (*METRICS, "edit_seconds", "inversion_seconds", "peak_memory_mib"):
            valid = complete_pairs(frame, split, metric)
            for category, group in [("all", valid), *list(valid.groupby("category"))]:
                for method, rows in group.groupby("method"):
                    summaries.append({"split": split, "category": category, "method": method,
                                      "metric": metric, "mean": rows[metric].mean(), "n": len(rows),
                                      "display_multiplier": SCALES.get(metric, 1),
                                      "display_mean": rows[metric].mean() * SCALES.get(metric, 1)})
    pd.DataFrame(summaries, columns=["split", "category", "method", "metric", "mean", "n",
                                     "display_multiplier", "display_mean"]).to_csv(output / "summary.csv", index=False)
    cis = [{"metric": metric, "comparator": comparator, **paired_bootstrap(frame, metric, comparator)}
           for metric in METRICS for comparator in ("p2p", "soft_fixed")]
    pd.DataFrame(cis).to_csv(output / "paired_bootstrap.csv", index=False)
    save_json(output / "coverage.json", coverage)
    checks = []
    joint = complete_pairs(frame.dropna(subset=["clip_edit", "lpips_bg"]), "test")
    for comparator in ("p2p", "soft_fixed"):
        if not len(joint):
            checks.append({"comparator": comparator, "status": "insufficient_data"})
            continue
        means = joint.groupby("method")[["clip_edit", "lpips_bg"]].mean()
        improvement = means.loc["soft_adaptive", "clip_edit"] - means.loc[comparator, "clip_edit"]
        bg = means.loc[comparator, "lpips_bg"]
        adaptive_bg = means.loc["soft_adaptive", "lpips_bg"]
        checks.append({"comparator": comparator, "n": joint.sample_id.nunique(),
                       "clip_edit_delta": improvement,
                       "lpips_bg_relative_change": (adaptive_bg - bg) / bg if bg > 0 else None,
                       "exploratory_support": bool(improvement > 0 and adaptive_bg <= bg * 1.05),
                       "interpretation": "Requires visual review; 5% is a project tolerance, not a paper criterion."})
    save_json(output / "exploratory_criteria.json", finite_json(checks))


def export_qualitative(manifest, output_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    output = Path(output_dir)
    registry = read_json(output / "registry.json")["records"]
    blind_rows, keys = [], []
    rng = random.Random(2026)
    for sample in manifest["samples"]:
        if sample["split"] == "dev":
            continue
        runs = {r["result"]["method"]: r["result"] for r in registry if r["sample_id"] == sample["id"]}
        if set(runs) != set(METHODS):
            continue
        folder = output / "qualitative" / sample["id"]
        folder.mkdir(parents=True, exist_ok=True)
        ref = runs["p2p"]
        tiles = [("Source", ref["source_path"])]
        if ref["reconstruction_path"]:
            tiles.append(("DDIM reconstruction", ref["reconstruction_path"]))
        tiles += [(method, runs[method]["edited_path"]) for method in METHODS]
        if sample["mask"] is not None:
            mask = Image.fromarray((decode_mask(sample["mask"]) * 255).astype("uint8")).convert("RGB")
            mask.save(folder / "evaluation_mask.png")
            tiles.append(("Evaluation mask", str(folder / "evaluation_mask.png")))
        grid = Image.new("RGB", (256 * len(tiles), 320), "white")
        draw = ImageDraw.Draw(grid)
        for i, (label, path) in enumerate(tiles):
            with Image.open(path) as image:
                grid.paste(image.convert("RGB").resize((256, 256)), (i * 256, 30))
            draw.text((i * 256 + 5, 5), label, fill="black")
        grid.save(folder / "comparison.png")
        save_json(folder / "prompts.json", sample["case"])
        fig, axes = plt.subplots(1, 2, figsize=(10, 3))
        for method, result in runs.items():
            logs = read_json(result["weights_path"])
            axes[0].plot([x["u"] for x in logs], [x["alpha_edited"] for x in logs], label=method)
            axes[1].plot([x["u"] for x in logs], [x["attention_js"] for x in logs], label=method)
        axes[0].set(title="Edited-token source weight", xlabel="Denoising progress", ylim=(-.05, 1.05))
        axes[1].set(title="Pre-injection attention JS", xlabel="Denoising progress")
        for ax in axes:
            ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(folder / "weights.png", dpi=130)
        plt.close(fig)
        shuffled = list(METHODS)
        rng.shuffle(shuffled)
        blind_dir = output / "blind_review" / sample["id"]
        blind_dir.mkdir(parents=True, exist_ok=True)
        with Image.open(ref["source_path"]) as image:
            image.save(blind_dir / "source.png")
        for label, method in zip("ABC", shuffled):
            with Image.open(runs[method]["edited_path"]) as image:
                image.save(blind_dir / f"{label}.png")
            blind_rows.append({"sample_id": sample["id"], "label": label,
                               "source_prompt": sample["case"]["source_prompt"], "target_prompt": sample["case"]["target_prompt"],
                               "edit_success_0_1_2": "", "preservation_0_1_2": "", "artifact_yes_no": "", "notes": ""})
            keys.append({"sample_id": sample["id"], "label": label, "method": method})
    # Never overwrite a user's completed score sheet when resuming.
    score_path = output / "blind_review" / "scores.csv"
    score_path.parent.mkdir(parents=True, exist_ok=True)
    if not score_path.exists():
        pd.DataFrame(blind_rows).to_csv(score_path, index=False)
    pd.DataFrame(keys).to_csv(output / "blind_key_do_not_open_before_scoring.csv", index=False)


class Budget:
    """Persisted measured operation time; notebook idle time/downloads excluded."""
    def __init__(self, output_dir, limit=3600, reserve=600):
        self.path = Path(output_dir) / "budget.json"
        self.limit, self.reserve = float(limit), float(reserve)
        self.data = read_json(self.path) if self.path.exists() else {"operations": [], "in_progress": None}
        # An interrupted operation is charged its reserved estimate, not lost.
        if self.data.get("in_progress"):
            pending = self.data.pop("in_progress")
            self.data["operations"].append({"name": pending["name"], "seconds": pending["estimate"], "interrupted": True})
            self.data["in_progress"] = None
            save_json(self.path, self.data)

    @property
    def spent(self):
        return sum(x["seconds"] for x in self.data["operations"])

    @property
    def remaining(self):
        return max(0., self.limit - self.spent)

    def estimate(self, name, default):
        values = [x["seconds"] for x in self.data["operations"] if x["name"] == name and not x.get("cached")]
        return float(np.median(values[-8:])) if values else default

    def can_edit(self, estimated_seconds):
        return self.remaining >= self.reserve + 1.2 * estimated_seconds

    @contextlib.contextmanager
    def measure(self, name, estimate=60):
        self.data["in_progress"] = {"name": name, "estimate": estimate}
        save_json(self.path, self.data)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        start = time.perf_counter()
        try:
            yield
        finally:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            self.data["operations"].append({"name": name, "seconds": time.perf_counter() - start})
            self.data["in_progress"] = None
            save_json(self.path, self.data)


def run_cases(samples, methods, config, budget, registry):
    """Checkpoint every method. Never select/drop samples based on output quality."""
    output = Path(config.output_dir)
    for sample in samples:
        case = prepare_case(**sample["case"])
        missing = [method for method in methods if not any(
            r["sample_id"] == sample["id"] and r["result"]["method"] == method
            and r["result"]["config"] == asdict(config) and valid_result(r["result"])
            for r in registry["records"])]
        if not missing:
            continue
        estimate = budget.estimate("initial", 45) + len(missing) * budget.estimate("edit", 35)
        if not budget.can_edit(estimate):
            return False
        with budget.measure("editor_load", estimate=15):
            editor = get_editor(config)
            cached = editor.initial_is_cached(case, config)
            del editor
        with budget.measure("initial_cached" if cached else "initial", estimate=45):
            invert_case(case, config)
        for method in missing:
            with budget.measure("edit", estimate=budget.estimate("edit", 35)):
                result = edit_case(case, method, config)
            registry["records"] = [r for r in registry["records"] if not (
                r["sample_id"] == sample["id"] and r["result"]["method"] == method
                and r["result"]["config"] == asdict(config))]
            registry["records"].append({"sample_id": sample["id"], "split": sample["split"], "result": result})
            save_json(output / "registry.json", registry)
        # A source-branch regression is an error, not an editing result.
        paired = [r["result"] for r in registry["records"] if r["sample_id"] == sample["id"]
                  and r["result"]["config"] == asdict(config)]
        if len(paired) > 1:
            src = np.asarray(Image.open(paired[0]["source_branch_path"])).astype(int)
            for result in paired[1:]:
                other = np.asarray(Image.open(result["source_branch_path"])).astype(int)
                if np.abs(src - other).max() > 1:
                    raise AssertionError("Source branches differ across methods; results cannot be compared fairly.")
    return True


def valid_result(result):
    folder = Path(result["edited_path"]).parent
    return all((folder / name).is_file() and file_hash(folder / name) == value
               for name, value in result["checksums"].items())


def run_experiment(config, data_root=None, limit_seconds=3600):
    """End-to-end notebook entrypoint, resumable with a fixed run directory."""
    output = Path(config.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    # Weight downloads must be completed by prefetch_assets() before timing.
    budget = Budget(output, limit_seconds)
    with budget.measure("editor_load", estimate=15):
        editor = get_editor(config)
    config = replace(config, output_dir=str(output), model_revision=editor.revision)
    manifest = build_manifest(output, editor.pipe.tokenizer, data_root)
    del editor  # release_editor() must actually free SD before loading metrics.
    run_spec = {"config": asdict(config), "manifest_sha256": file_hash(output / "manifest.json"),
                "algorithm_sha256": file_hash(ROOT / "adaptive_p2p.py"),
                "runner_sha256": file_hash(__file__), "environment": environment_snapshot()}
    if (output / "run_spec.json").exists() and read_json(output / "run_spec.json") != run_spec:
        raise ValueError("Run settings/code/environment changed. Use a new output directory.")
    save_json(output / "run_spec.json", run_spec)
    save_json(output / "environment.json", environment_snapshot())
    registry = read_json(output / "registry.json") if (output / "registry.json").exists() else {"records": []}
    save_json(output / "registry.json", registry)
    generated = [s for s in manifest["samples"] if s["split"] == "generated"]
    dev = [s for s in manifest["samples"] if s["split"] == "dev"]
    selected = output / "selected_tau.json"
    if not selected.exists():
        if dev:
            for tau in (.4, .6, .8):
                if not run_cases(dev, ("p2p",), replace(config, tau=tau), budget, registry):
                    save_json(output / "status.json", {"status": "budget_exhausted_during_development", "spent": budget.spent})
                    return
            release_editor()
            frame = evaluate_results(manifest, output, families=("clip", "lpips"), device=config.device, splits=("dev",), budget=budget)
            try:
                tau, scores = select_tau(frame)
            except (ValueError, AttributeError):
                save_json(output / "status.json", {"status": "development_metrics_incomplete", "spent": budget.spent})
                return
            save_json(selected, {"tau": tau, "selection": scores, "source": "development_only"})
        else:
            save_json(selected, {"tau": config.tau, "source": "preregistered_generated_only_default; no PIE data"})
    config = replace(config, tau=read_json(selected)["tau"])
    if dev:
        run_cases(dev, ("soft_fixed", "soft_adaptive"), config, budget, registry)
    run_cases(generated, METHODS, config, budget, registry)
    tests = sorted([s for s in manifest["samples"] if s["split"] == "test"], key=lambda s: (s["order"], s["category"]))
    selection_path = output / "test_selection.json"
    if not selection_path.exists():
        cost = 1.2 * (budget.estimate("initial", 45) + 3 * budget.estimate("edit", 35))
        available = max(0, budget.remaining - budget.reserve)
        count = next((n for n in (12, 8, 4) if n <= len(tests) and n * cost <= available), 0)
        save_json(selection_path, {"sample_ids": [s["id"] for s in tests[:count]], "estimated_seconds_per_case": cost,
                                  "remaining_at_selection": budget.remaining, "reason": "runtime only; before inspecting test outputs"})
    ids = set(read_json(selection_path)["sample_ids"])
    run_cases([s for s in tests if s["id"] in ids], METHODS, config, budget, registry)
    release_editor()
    evaluate_results(manifest, output, device=config.device, splits=("test", "generated"), budget=budget)
    # Development ablations are diagnostics and use only spare time after primary results.
    if dev and budget.remaining > 60:
        evaluate_results(manifest, output, families=("clip", "lpips"), device=config.device, splits=("dev",), budget=budget)
    export_qualitative(manifest, output)
    save_json(output / "status.json", {"status": "finished_available_budget", "spent_seconds": budget.spent,
                                      "selected_test_samples": len(ids), "limit_seconds": limit_seconds,
                                      "note": "Inspect coverage.json for missing methods/metrics; no claim of improvement."})
    return output


def prefetch_assets(config):
    """Run before starting the GPU budget: download SD/tokenizer and metric weights."""
    from huggingface_hub import HfApi, snapshot_download
    revision = HfApi().model_info(config.model_id, revision=config.model_revision).sha
    available = set(HfApi().list_repo_files(config.model_id, revision=revision))
    patterns = ["model_index.json", "scheduler/*", "tokenizer/*", "feature_extractor/*"]
    for component, stem in (("text_encoder", "model"), ("unet", "diffusion_pytorch_model"), ("vae", "diffusion_pytorch_model")):
        patterns.append(f"{component}/config.json")
        safe = f"{component}/{stem}.safetensors"
        binary = f"{component}/{'pytorch_model' if component == 'text_encoder' else stem}.bin"
        patterns.append(safe if safe in available else binary)
    snapshot_download(config.model_id, revision=revision, allow_patterns=patterns)
    for family in ("clip", "lpips", "structure"):
        adapter = OfficialMetricAdapter(family, "cpu")
        del adapter
        gc.collect()
    return replace(config, model_revision=revision)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prefetch", "run", "evaluate", "figures"))
    parser.add_argument("--output-dir", default="adaptive_results")
    parser.add_argument("--data-root")
    parser.add_argument("--budget-seconds", type=int, default=3600)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    cfg = EditConfig(output_dir=args.output_dir, device=args.device, dtype="float32" if args.device == "cpu" else "float16")
    if args.command == "prefetch":
        print(prefetch_assets(cfg))
    elif args.command == "run":
        run_experiment(cfg, args.data_root, args.budget_seconds)
    elif args.command == "evaluate":
        evaluate_results(Path(args.output_dir) / "manifest.json", args.output_dir, device=args.device)
    else:
        export_qualitative(read_json(Path(args.output_dir) / "manifest.json"), args.output_dir)
