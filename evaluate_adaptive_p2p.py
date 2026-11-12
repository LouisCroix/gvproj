"""Configurable, paired PIE-Bench experiment; pinned PnP evaluation protocol.

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
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageDraw
from tqdm.auto import tqdm

from adaptive_p2p import (ROOT, METHODS, EditConfig, digest, edit_case, environment_snapshot,
                          file_hash, get_editor, invert_case, prepare_case, release_editor,
                          save_json, validate_alignment)

PNP_COMMIT = "07f97f448150e2ca220bebd54c8f687c5c50c67a"
DINO_COMMIT = "7c446df5b9f45747937fb0d72314eb9f7b66930a"
CATEGORIES = {"1": "object", "5": "pose", "6": "color", "7": "material"}
ALL_CATEGORIES = ("0", "1", "2", "3", "4", "5", "6", "7", "8", "9")
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


@dataclass(frozen=True)
class DatasetConfig:
    categories: tuple[str, ...] | str = ("1", "5", "6", "7")
    sampling: str = "per_category"
    dev_per_category: int | dict[str, int] = 1
    test_per_category: int | dict[str, int] | None = 3
    split_seed: int = 2026

    def __post_init__(self):
        categories = ALL_CATEGORIES if self.categories == "all" else self.categories
        if isinstance(categories, str):
            raise ValueError("categories must be a sequence of category IDs or 'all'.")
        categories = tuple(sorted(set(map(str, categories)), key=int))
        if not categories or not set(categories) <= set(ALL_CATEGORIES):
            raise ValueError("PIE category IDs must be in 0..9.")
        object.__setattr__(self, "categories", categories)
        if self.sampling not in ("all", "per_category"):
            raise ValueError("sampling must be 'all' or 'per_category'.")
        if self.sampling == "all" and self.test_per_category is not None:
            raise ValueError("Use test_per_category=None with sampling='all'.")
        for name in ("dev_per_category", "test_per_category"):
            value = getattr(self, name)
            if value is None and name == "test_per_category" and self.sampling == "all":
                continue
            if isinstance(value, dict):
                value = {str(k): v for k, v in value.items()}
                if set(value) != set(categories):
                    raise ValueError(f"{name} must specify every selected category exactly once.")
                object.__setattr__(self, name, value)
                values = value.values()
            else:
                values = [value]
            minimum = 0 if name == "dev_per_category" else 1
            if any(type(v) is not int or v < minimum for v in values):
                raise ValueError(f"{name} must contain integers >= {minimum}.")
        if type(self.split_seed) is not int:
            raise ValueError("split_seed must be an integer.")

    def quota(self, name, category):
        value = getattr(self, name)
        return value[category] if isinstance(value, dict) else value


def prepare_dataset(data_root, output_dir, dataset_config=None, tokenizer=None, model_config=None):
    """CPU preflight and frozen selection; loads only the SD tokenizer if needed."""
    cfg = dataset_config or DatasetConfig()
    if not data_root:
        raise ValueError("PIE-Bench is required: set data_root to its extracted directory.")
    data_root = Path(data_root).resolve()
    annotation_path = data_root / "mapping_file.json"
    if not annotation_path.is_file() or not (data_root / "annotation_images").is_dir():
        raise FileNotFoundError(f"PIE-Bench needs mapping_file.json and annotation_images/: {data_root}")
    folder = Path(output_dir).resolve()
    destination = folder / "manifest.json"
    settings = json.loads(json.dumps(asdict(cfg)))
    model = model_config or EditConfig()
    if model.model_revision == "main":
        # The repository records the immutable SD revision used by the tokenizer fixtures.
        revision = read_json(ROOT / "vendor" / "reference_manifest.json")["sd14_revision"]
    else:
        revision = model.model_revision
    tokenizer_spec = {"model_id": model.model_id, "revision": revision}
    if destination.exists():
        old = read_json(destination)
        if (old.get("schema_version") != 2 or old["data_root"] != str(data_root)
                or old.get("dataset_config") != settings or old.get("tokenizer") != tokenizer_spec):
            raise ValueError("Manifest/settings are frozen. Use a new output directory for the new protocol/configuration.")
        verify_manifest(old)
        return old
    if tokenizer is None:
        from transformers import CLIPTokenizer
        print("Loading tokenizer only for CPU data preflight.", flush=True)
        tokenizer = CLIPTokenizer.from_pretrained(model.model_id, subfolder="tokenizer", revision=revision)
    annotations = read_json(annotation_path)
    image_root = (data_root / "annotation_images").resolve()
    candidates = {cat: [] for cat in cfg.categories}
    counts = {cat: {"category": cat, "raw": 0, "eligible": 0, "dev": 0, "test": 0,
                    "status": "unselected"} for cat in cfg.categories}
    samples, exclusions = [], []
    for sample_id, ann in tqdm(sorted(annotations.items()), desc="PIE eligibility", unit="record"):
        cat = str(ann.get("editing_type_id"))
        if cat not in candidates:
            continue
        counts[cat]["raw"] += 1
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
            if (1 - decode_mask(ann["mask"])).sum() == 0:
                raise ValueError("No background remains after official boundary handling.")
            candidates[cat].append({"id": str(sample_id), "category": cat, "kind": "real",
                                   "case": asdict(case), "mask": ann["mask"], "image_sha256": file_hash(path)})
        except (ValueError, OSError, IndexError, KeyError, TypeError) as exc:
            exclusions.append({"id": str(sample_id), "category": cat, "reason": str(exc)})
    errors = []
    for cat, eligible in candidates.items():
        random.Random(f"pie-v2:{cfg.split_seed}:{cat}").shuffle(eligible)
        counts[cat]["eligible"] = len(eligible)
        if not eligible:
            counts[cat]["status"] = "skipped_no_eligible_samples"
            continue
        ndev = cfg.quota("dev_per_category", cat)
        ntest = len(eligible) - ndev if cfg.sampling == "all" else cfg.quota("test_per_category", cat)
        if ntest < 1 or len(eligible) < ndev + ntest:
            counts[cat]["status"] = "insufficient_quota"
            errors.append(f"Category {cat}: eligible={len(eligible)}, requested dev={ndev}, test={ntest} (at least 1 test required).")
            continue
        counts[cat].update(dev=ndev, test=ntest, status="selected")
        for i, item in enumerate(eligible[:ndev + ntest]):
            item.update(split="dev" if i < ndev else "test", order=i if i < ndev else i - ndev)
            samples.append(item)
    if samples and not any(s["split"] == "dev" for s in samples):
        errors.append("At least one development sample is required overall to select tau.")
    report = {"dataset_config": settings, "categories": list(counts.values()), "exclusions": exclusions, "errors": errors}
    save_json(folder / "dataset_preflight.json", report)
    print(pd.DataFrame(counts.values()).to_string(index=False), flush=True)
    if errors or not samples:
        raise ValueError("Dataset preflight failed. " + " ".join(errors or ["No eligible samples in selected categories."]))
    manifest = {"schema_version": 2, "dataset_config": settings, "tokenizer": tokenizer_spec,
                "data_root": str(data_root), "annotation_sha256": file_hash(annotation_path),
                "samples": samples, "exclusions": exclusions,
                "category_counts": list(counts.values()), "test_order": "round-robin by order then category",
                "protocol": "replacement-only; no LocalBlend; masks evaluation-only"}
    save_json(destination, manifest)
    return manifest


def build_manifest(output_dir, tokenizer, data_root=None, dataset_config=None):
    """Compatibility wrapper; new callers may use prepare_dataset directly."""
    return prepare_dataset(data_root, output_dir, dataset_config, tokenizer)


def verify_manifest(manifest):
    annotation = Path(manifest["data_root"]) / "mapping_file.json"
    if file_hash(annotation) != manifest["annotation_sha256"]:
        raise ValueError("Annotations changed after selection. Use a new output directory.")
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


def evaluate_results(manifest, output_dir, families=("pixels", "clip", "lpips", "structure"), device="cuda", splits=None, budget=None, summarize_output=True):
    """Resume per family/run; writes raw.csv, summary, CIs and coverage."""
    manifest = read_json(manifest) if isinstance(manifest, (str, Path)) else manifest
    output = Path(output_dir).resolve()
    registry_path = output / "registry.json"
    if not registry_path.exists():
        raise FileNotFoundError("No experiment registry. Run generation/editing first.")
    registry = read_json(registry_path)
    samples = {x["id"]: x for x in manifest["samples"]}
    records = [x for x in registry["records"] if (splits is None or x["split"] in splits) and valid_result(x["result"])]
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
        cached_count = len(records) - len(pending)
        if not pending:
            continue
        if budget and budget.remaining < 5:
            break
        print(f"Loading {family} metrics: pending={len(pending)}, cached={cached_count}", flush=True)
        with budget.measure("metric_load_" + family) if budget else contextlib.nullcontext():
            adapter = OfficialMetricAdapter(family, device)
        for record in tqdm(pending, desc=f"Metrics {family} (cached {cached_count})", initial=cached_count,
                           total=len(records), unit="result"):
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
        if not valid_result(result):
            continue
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
    if summarize_output:
        summarize(frame, manifest, output)
    return frame


def select_tau(frame, expected_ids=None):
    dev = frame[(frame.split == "dev") & (frame.method == "p2p")].copy()
    taus = (0.4, 0.6, 0.8)
    expected = set(map(str, expected_ids)) if expected_ids is not None else set(dev.sample_id.astype(str))
    for tau in taus:
        rows = dev[np.isclose(dev.tau, tau)]
        if (not expected or len(rows) != len(expected) or set(rows.sample_id.astype(str)) != expected
                or not np.isfinite(rows[["clip_edit", "lpips_bg"]].to_numpy(dtype=float)).all()):
            raise ValueError("Tau selection requires the same complete development IDs at all three tau values.")
    means = dev.groupby("tau")[["clip_edit", "lpips_bg"]].mean()
    means["rank_sum"] = means.clip_edit.rank(ascending=False) + means.lpips_bg.rank(ascending=True)
    means = means.reset_index().sort_values(["rank_sum", "tau"], ascending=[True, False])
    return float(means.iloc[0].tau), means.to_dict("records")


def complete_pairs(frame, split, metric=None):
    data = frame[frame.split == split]
    if metric:
        data = data[np.isfinite(pd.to_numeric(data[metric], errors="coerce"))]
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
    registry_path = output / "registry.json"
    registry = read_json(registry_path)["records"] if registry_path.exists() else []
    for split in ("test",):
        paired = complete_pairs(frame, split)
        selected = [s for s in manifest["samples"] if s["split"] == split]
        edited_ids = {s["id"] for s in selected if set(METHODS) <= {
            r["result"]["method"] for r in registry if r["sample_id"] == s["id"] and valid_result(r["result"])}}
        finite = np.isfinite(frame[list(METRICS)].to_numpy(dtype=float)).all(axis=1)
        metric_ids = set(complete_pairs(frame[finite], split).sample_id)
        planned = set(s["id"] for s in selected)
        coverage.append({"split": split, "planned_samples": len(selected),
                         "editing_complete_samples": len(edited_ids), "metrics_complete_samples": len(metric_ids),
                         "samples_with_all_methods": paired.sample_id.nunique(),
                         "paired_valid_n_per_metric": {metric: int(complete_pairs(frame, split, metric).sample_id.nunique())
                                                       for metric in METRICS},
                         "missing_ids": sorted(planned - metric_ids),
                         "pending_edit_ids": sorted(planned - edited_ids),
                         "pending_metric_ids": sorted(edited_ids - metric_ids),
                         "intermediate": metric_ids != planned})
        for metric in (*METRICS, "edit_seconds", "inversion_seconds", "peak_memory_mib"):
            valid = complete_pairs(frame, split, metric)
            for category, group in [("all", valid), *list(valid.groupby("category"))]:
                for method, rows in group.groupby("method"):
                    summaries.append({"split": split, "category": category, "method": method,
                                      "metric": metric, "mean": rows[metric].mean(), "n": len(rows),
                                      "display_multiplier": SCALES.get(metric, 1),
                                      "display_mean": rows[metric].mean() * SCALES.get(metric, 1)})
            for method, rows in valid.groupby("method"):
                mean = rows.groupby("category")[metric].mean().mean()
                summaries.append({"split": split, "category": "all_category_equal", "method": method,
                                  "metric": metric, "mean": mean, "n": len(rows),
                                  "display_multiplier": SCALES.get(metric, 1),
                                  "display_mean": mean * SCALES.get(metric, 1)})
    pd.DataFrame(summaries, columns=["split", "category", "method", "metric", "mean", "n",
                                     "display_multiplier", "display_mean"]).to_csv(output / "summary.csv", index=False)
    comparisons = [(metric, comparator) for metric in METRICS for comparator in ("p2p", "soft_fixed")]
    cis = [{"metric": metric, "comparator": comparator, **paired_bootstrap(frame, metric, comparator)}
           for metric, comparator in tqdm(comparisons, desc="Paired bootstrap", unit="comparison")]
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
    for sample in tqdm(manifest["samples"], desc="Export comparisons", unit="sample"):
        if sample["split"] == "dev":
            continue
        runs = {r["result"]["method"]: r["result"] for r in registry
                if r["sample_id"] == sample["id"] and valid_result(r["result"])}
        if set(runs) != set(METHODS):
            continue
        folder = output / "qualitative" / sample["id"]
        folder.mkdir(parents=True, exist_ok=True)
        shuffled = list(METHODS)
        random.Random(f"blind-v2:2026:{sample['id']}").shuffle(shuffled)
        for label, method in zip("ABC", shuffled):
            blind_rows.append({"sample_id": sample["id"], "label": label,
                               "source_prompt": sample["case"]["source_prompt"], "target_prompt": sample["case"]["target_prompt"],
                               "edit_success_0_1_2": "", "preservation_0_1_2": "", "artifact_yes_no": "", "notes": ""})
            keys.append({"sample_id": sample["id"], "label": label, "method": method})
        stamp = {method: result["run_id"] for method, result in runs.items()}
        stamp_path = folder / "export.json"
        blind_dir = output / "blind_review" / sample["id"]
        if (stamp_path.exists() and read_json(stamp_path) == stamp
                and all((folder / name).exists() for name in ("comparison.png", "weights.png", "prompts.json", "evaluation_mask.png"))
                and all((blind_dir / name).exists() for name in ("source.png", "A.png", "B.png", "C.png"))):
            continue
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
        blind_dir.mkdir(parents=True, exist_ok=True)
        with Image.open(ref["source_path"]) as image:
            image.save(blind_dir / "source.png")
        for label, method in zip("ABC", shuffled):
            with Image.open(runs[method]["edited_path"]) as image:
                image.save(blind_dir / f"{label}.png")
        save_json(stamp_path, stamp)
    # Never overwrite a user's completed score sheet when resuming.
    score_path = output / "blind_review" / "scores.csv"
    score_path.parent.mkdir(parents=True, exist_ok=True)
    if blind_rows:
        new = pd.DataFrame(blind_rows)
        if score_path.exists():
            old = pd.read_csv(score_path, dtype=str, keep_default_na=False)
            existing = set(zip(old.sample_id, old.label))
            new = pd.concat([old, new[[ (str(r.sample_id), r.label) not in existing for r in new.itertuples() ]]], ignore_index=True)
        new.to_csv(score_path, index=False)
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
        self.session_start_spent = self.spent
        self.session_started = time.perf_counter()

    @property
    def spent(self):
        return sum(x["seconds"] for x in self.data["operations"])

    @property
    def remaining(self):
        return max(0., self.limit - self.session_spent)

    @property
    def session_spent(self):
        return max(self.spent - self.session_start_spent, time.perf_counter() - self.session_started)

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
    def is_done(sample, method):
        return any(r["sample_id"] == sample["id"] and r["result"]["method"] == method
                   and r["result"]["config"] == asdict(config) and valid_result(r["result"])
                   for r in registry["records"])
    completed = sum(is_done(s, m) for s in samples for m in methods)
    progress = tqdm(total=len(samples) * len(methods), initial=completed, desc=f"Editing tau={config.tau}", unit="result")
    announced = False
    try:
        for sample in samples:
            case = prepare_case(**sample["case"])
            missing = [method for method in methods if not is_done(sample, method)]
            if not missing:
                continue
            estimate = budget.estimate("initial", 45) + len(missing) * budget.estimate("edit", 35)
            if not budget.can_edit(estimate):
                return False
            progress.set_postfix(id=sample["id"], stage=sample["split"], remaining_s=round(budget.remaining))
            if not announced:
                print("Loading/reusing diffusion model.", flush=True)
                announced = True
            with budget.measure("editor_load", estimate=15):
                editor = get_editor(config)
                cached = editor.initial_is_cached(case, config)
                del editor
            with budget.measure("initial_cached" if cached else "initial", estimate=45):
                invert_case(case, config)
            for method in missing:
                if not budget.can_edit(budget.estimate("edit", 35)):
                    return False
                progress.set_postfix(id=sample["id"], method=method, remaining_s=round(budget.remaining))
                with budget.measure("edit", estimate=budget.estimate("edit", 35)):
                    result = edit_case(case, method, config)
                registry["records"] = [r for r in registry["records"] if not (
                    r["sample_id"] == sample["id"] and r["result"]["method"] == method
                    and r["result"]["config"] == asdict(config))]
                registry["records"].append({"sample_id": sample["id"], "split": sample["split"], "result": result})
                save_json(output / "registry.json", registry)
                progress.update(1)
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
    finally:
        progress.close()


def valid_result(result):
    folder = Path(result["edited_path"]).parent
    return all((folder / name).is_file() and file_hash(folder / name) == value
               for name, value in result["checksums"].items())


def run_experiment(config, data_root=None, limit_seconds=3600, dataset_config=None):
    """Frozen PIE selection, development tuning, and resumable 12-case blocks."""
    output = Path(config.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    manifest = prepare_dataset(data_root, output, dataset_config, model_config=config)
    config = replace(config, output_dir=str(output), model_revision=manifest["tokenizer"]["revision"])
    run_spec = {"config": asdict(config), "manifest_sha256": file_hash(output / "manifest.json"),
                "algorithm_sha256": file_hash(ROOT / "adaptive_p2p.py"),
                "runner_sha256": file_hash(__file__), "environment": environment_snapshot()}
    if (output / "run_spec.json").exists() and read_json(output / "run_spec.json") != run_spec:
        raise ValueError("Run settings/code/environment changed. Use a new output directory.")
    save_json(output / "run_spec.json", run_spec)
    save_json(output / "environment.json", environment_snapshot())
    # Prefetch downloads outside this call. Each call receives a fresh time allowance.
    budget = Budget(output, limit_seconds)
    registry = read_json(output / "registry.json") if (output / "registry.json").exists() else {"records": []}
    save_json(output / "registry.json", registry)
    ordered = sorted(manifest["samples"], key=lambda s: (s["order"], int(s["category"])))
    dev = [s for s in ordered if s["split"] == "dev"]
    tests = [s for s in ordered if s["split"] == "test"]
    save_json(output / "test_selection.json", {"sample_ids": [s["id"] for s in tests],
              "reason": "Frozen complete selection; budget pauses execution, never reduces this list."})
    selected = output / "selected_tau.json"

    def metrics(split):
        return evaluate_results(manifest, output, device=config.device, splits=(split,), budget=budget,
                                families=("clip", "lpips") if split == "dev" else ("pixels", "clip", "lpips", "structure"),
                                summarize_output=False)

    def blocks(samples, methods, settings):
        for start in range(0, len(samples), 12):
            # Reserve measured metric throughput for the next block, with a 10-minute minimum.
            block = samples[start:start + 12]
            if all(any(r["sample_id"] == s["id"] and r["result"]["method"] == m
                       and r["result"]["config"] == asdict(settings) and valid_result(r["result"])
                       for r in registry["records"]) for s in block for m in methods):
                continue
            print(f"{block[0]['split']} block {start // 12 + 1}/{math.ceil(len(samples) / 12)}: "
                  f"{len(block)} samples, tau={settings.tau}", flush=True)
            metric_cost = sum(budget.estimate("metric_" + f, 1.) for f in
                              (("clip", "lpips") if block[0]["split"] == "dev" else ("pixels", "clip", "lpips", "structure")))
            budget.reserve = max(600., len(block) * len(methods) * metric_cost * 1.2)
            done = run_cases(block, methods, settings, budget, registry)
            frame = metrics(block[0]["split"])
            if not done or budget.remaining <= budget.reserve:
                return False, frame
        return True, None

    def finish(reason):
        release_editor()
        # Reads existing caches even if time is exhausted; no new GPU work then.
        frame = evaluate_results(manifest, output, families=(), device=config.device, budget=budget)
        with budget.measure("export", estimate=30):
            export_qualitative(manifest, output)
        coverage = read_json(output / "coverage.json")[0]
        dev_complete = False
        if selected.exists():
            chosen = float(read_json(selected)["tau"])
            rows = frame[(frame.split == "dev") & np.isclose(frame.tau, chosen)]
            dev_complete = complete_pairs(rows.dropna(subset=["clip_edit", "lpips_bg"]), "dev").sample_id.nunique() == len(dev)
        complete = coverage["metrics_complete_samples"] == len(tests) and dev_complete
        status = {"status": "complete" if complete else "paused_budget", "reason": reason,
                  "spent_seconds": budget.spent, "session_seconds": budget.session_spent,
                  "limit_seconds": limit_seconds, "planned_test_samples": len(tests),
                  "note": "Development excluded from primary results; see coverage.json for pending IDs."}
        save_json(output / "status.json", status)
        return output

    try:
        # Resume metric work before loading diffusion again.
        metrics("test")
        metrics("dev")
        for tau in (.4, .6, .8):
            done, _ = blocks(dev, ("p2p",), replace(config, tau=tau))
            if not done:
                return finish("Development generation/metrics await the next session.")
        if not selected.exists():
            frame = metrics("dev")
            try:
                tau, scores = select_tau(frame, [s["id"] for s in dev])
            except ValueError:
                return finish("Development metrics incomplete.")
            save_json(selected, {"tau": tau, "selection": scores, "source": "development_only"})
        config = replace(config, tau=read_json(selected)["tau"])
        done, _ = blocks(dev, ("soft_fixed", "soft_adaptive"), config)
        if not done:
            return finish("Development diagnostics incomplete.")
        blocks(tests, METHODS, config)
        return finish("All currently available work saved.")
    except BaseException as exc:
        release_editor()
        save_json(output / "status.json", {"status": "paused_budget" if isinstance(exc, KeyboardInterrupt) else "failed",
                  "error": f"{type(exc).__name__}: {exc}", "spent_seconds": budget.spent,
                  "session_seconds": budget.session_spent})
        raise


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
    parser.add_argument("command", choices=("prepare", "prefetch", "run", "evaluate", "figures", "verify-metrics"))
    parser.add_argument("--output-dir", default="adaptive_results")
    parser.add_argument("--data-root")
    parser.add_argument("--budget-seconds", type=int, default=3600)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dataset-config", help="JSON object file containing DatasetConfig fields")
    args = parser.parse_args()
    cfg = EditConfig(output_dir=args.output_dir, device=args.device, dtype="float32" if args.device == "cpu" else "float16")
    dataset = DatasetConfig(**read_json(args.dataset_config)) if args.dataset_config else DatasetConfig()
    if args.command == "prepare":
        prepare_dataset(args.data_root, args.output_dir, dataset, model_config=cfg)
    elif args.command == "verify-metrics":
        save_json(Path(args.output_dir) / "pretrained_metric_parity.json", verify_pretrained_metrics(args.device))
    elif args.command == "prefetch":
        print(prefetch_assets(cfg))
    elif args.command == "run":
        run_experiment(cfg, args.data_root, args.budget_seconds, dataset)
    elif args.command == "evaluate":
        evaluate_results(Path(args.output_dir) / "manifest.json", args.output_dir, device=args.device)
    else:
        export_qualitative(read_json(Path(args.output_dir) / "manifest.json"), args.output_dir)
