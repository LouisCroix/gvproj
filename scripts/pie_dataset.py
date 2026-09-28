"""Acquire official PIE-Bench archives and audit exact duplicates; no GPU required."""
import hashlib
import json
import shutil
import tempfile
import urllib.parse
import urllib.request
import zipfile
from collections import defaultdict
from pathlib import Path, PureWindowsPath

from PIL import Image

OFFICIAL_ENTRY = "https://forms.gle/hVMkTABb4uvZVjme9"


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def pixel_hash(path):
    with Image.open(path) as image:
        rgb = image.convert("RGB")
        return hashlib.sha256(str(rgb.size).encode() + rgb.tobytes()).hexdigest()


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key/sample ID: {key}")
        result[key] = value
    return result


def audit_dataset(root, report_path=None):
    root = Path(root).resolve()
    annotations = json.loads((root / "mapping_file.json").read_text(encoding="utf-8"),
                             object_pairs_hook=unique_object)
    if not isinstance(annotations, dict) or not annotations:
        raise ValueError("mapping_file.json must contain a nonempty object.")
    image_root = (root / "annotation_images").resolve()
    images, tasks, rows, errors = defaultdict(list), defaultdict(list), [], []
    for sample_id, ann in sorted(annotations.items()):
        try:
            relative = ann["image_path"]
            path = (image_root / relative).resolve()
            if PureWindowsPath(relative).drive or "\\" in relative or not path.is_relative_to(image_root):
                raise ValueError("Image path escapes annotation_images")
            with Image.open(path) as image:
                image.load()
                size = list(image.size)
            digest = pixel_hash(path)
            prompts = tuple(" ".join(ann[k].replace("[", "").replace("]", "").split())
                            for k in ("original_prompt", "editing_prompt"))
            if not isinstance(ann["mask"], list) or len(ann["mask"]) % 2:
                raise ValueError("Invalid RLE mask shape")
            images[digest].append(sample_id)
            tasks[(digest, *prompts, json.dumps(ann["mask"]))].append(sample_id)
            rows.append({"id": sample_id, "category": str(ann["editing_type_id"]),
                         "path": relative, "size": size, "sha256": sha256(path), "pixel_sha256": digest})
        except (OSError, ValueError, TypeError, KeyError) as exc:
            errors.append({"id": sample_id, "reason": str(exc)})
    report = {"data_root": str(root), "mapping_sha256": sha256(root / "mapping_file.json"),
              "sample_count": len(annotations), "valid_files": len(rows), "errors": errors,
              "duplicate_images": [v for v in images.values() if len(v) > 1],
              "duplicate_tasks": [v for v in tasks.values() if len(v) > 1], "files": rows,
              "note": "Exact decoded RGB duplicates only; not perceptual similarity. Eligibility/masks/token alignment checked during selection."}
    if report_path:
        path = Path(report_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def prepare_pie_dataset(destination, archive_path=None, download_url=None, expected_sha256=None):
    """Reuse existing data; download only when necessary. Never replace an existing dataset."""
    destination = Path(destination).resolve()
    if (destination / "mapping_file.json").is_file():
        report = audit_dataset(destination, destination / "dataset_audit.json")
        if report["errors"]:
            raise ValueError("Dataset contains missing/corrupt records; see dataset_audit.json")
        return destination
    if destination.exists() and any(destination.iterdir()):
        raise ValueError("Destination is nonempty but incomplete; choose a new destination.")
    archive = Path(archive_path).expanduser().resolve() if archive_path else destination.with_suffix(".zip")
    if not archive.is_file():
        if not download_url:
            raise FileNotFoundError(f"Provide an existing ZIP or authorized download URL from {OFFICIAL_ENTRY}")
        parsed = urllib.parse.urlparse(download_url)
        if parsed.scheme != "https":
            raise ValueError("Download URL must use HTTPS.")
        archive.parent.mkdir(parents=True, exist_ok=True)
        partial = archive.with_name(archive.name + ".partial")
        if parsed.hostname == "drive.google.com":
            import gdown
            gdown.download(url=download_url, output=str(partial), quiet=False, fuzzy=True, resume=True)
        else:
            with urllib.request.urlopen(download_url, timeout=60) as response, partial.open("wb") as out:
                shutil.copyfileobj(response, out)
        if not zipfile.is_zipfile(partial):
            raise ValueError("Download is not a ZIP (possibly login/quota HTML). Use official manual download; no dataset was replaced.")
        partial.replace(archive)
    checksum = sha256(archive)
    if expected_sha256 and checksum.lower() != expected_sha256.lower():
        raise ValueError("Archive SHA256 mismatch")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="pie-extract-", dir=destination.parent) as temp:
        staging = Path(temp).resolve()
        with zipfile.ZipFile(archive) as bundle:
            names = set()
            for member in bundle.infolist():
                name = member.filename
                target = (staging / name).resolve()
                if (PureWindowsPath(name).drive or "\\" in name or not target.is_relative_to(staging)
                        or (member.external_attr >> 16) & 0o170000 == 0o120000):
                    raise ValueError(f"Unsafe archive path: {name}")
                if name in names:
                    raise ValueError(f"Duplicate archive member: {name}")
                names.add(name)
            bundle.extractall(staging)
        roots = [p.parent for p in staging.rglob("mapping_file.json")
                 if (p.parent / "annotation_images").is_dir()]
        if len(roots) != 1:
            raise ValueError("ZIP must contain exactly one PIE mapping_file.json + annotation_images directory.")
        report = audit_dataset(roots[0])
        if report["errors"]:
            raise ValueError(f"Archive contains invalid records: {report['errors'][:5]}")
        destination.mkdir(exist_ok=True)
        shutil.copytree(roots[0], destination, dirs_exist_ok=True)
    (destination / "download_provenance.json").write_text(json.dumps({
        "official_entry": OFFICIAL_ENTRY, "archive": str(archive), "archive_sha256": checksum,
        "checksum_verified_against_supplied_value": bool(expected_sha256),
        "source_host": urllib.parse.urlparse(download_url).hostname if download_url else None,
    }, indent=2), encoding="utf-8")
    audit_dataset(destination, destination / "dataset_audit.json")
    return destination


def audit_split_duplicates(manifest):
    groups = defaultdict(list)
    for sample in manifest["samples"]:
        if sample["kind"] == "real":
            groups[pixel_hash(sample["case"]["source_image"])].append(
                {"id": sample["id"], "split": sample["split"]})
    duplicates = [v for v in groups.values() if len(v) > 1]
    return {"duplicate_selected_images": duplicates,
            "dev_test_overlap": [v for v in duplicates if {x["split"] for x in v} >= {"dev", "test"}]}
