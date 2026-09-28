import json
import io
import zipfile

import pytest
from PIL import Image

from scripts.pie_dataset import prepare_pie_dataset, audit_dataset, audit_split_duplicates


def fixture_data(root):
    images = root / "annotation_images"
    images.mkdir(parents=True)
    Image.new("RGB", (512, 512), "red").save(images / "a.png")
    Image.new("RGB", (512, 512), "red").save(images / "b.png")
    ann = {name: {"image_path": name + ".png", "editing_type_id": "6",
                  "original_prompt": "red car", "editing_prompt": "blue car", "mask": [1, 3]}
           for name in ("a", "b")}
    (root / "mapping_file.json").write_text(json.dumps(ann))


def test_extract_reuse_duplicates_and_leakage(tmp_path):
    source = tmp_path / "source"
    fixture_data(source)
    archive = tmp_path / "data.zip"
    with zipfile.ZipFile(archive, "w") as z:
        for p in source.rglob("*"):
            if p.is_file(): z.write(p, "data/" + p.relative_to(source).as_posix())
    dest = tmp_path / "dataset"
    assert prepare_pie_dataset(dest, archive) == dest
    assert prepare_pie_dataset(dest, tmp_path / "absent.zip") == dest
    report = audit_dataset(dest)
    assert report["duplicate_images"] == [["a", "b"]]
    assert report["duplicate_tasks"] == [["a", "b"]]
    manifest = {"samples": [{"id": name, "kind": "real", "split": split,
                              "case": {"source_image": str(dest / "annotation_images" / (name + ".png"))}}
                             for name, split in (("a", "dev"), ("b", "test"))]}
    assert len(audit_split_duplicates(manifest)["dev_test_overlap"]) == 1
    (dest / "annotation_images" / "a.png").write_bytes(b"bad image")
    with pytest.raises(ValueError, match="missing/corrupt"):
        prepare_pie_dataset(dest)


def test_unsafe_zip_and_checksum(tmp_path):
    archive = tmp_path / "bad.zip"
    with zipfile.ZipFile(archive, "w") as z: z.writestr("../escaped", "bad")
    with pytest.raises(ValueError, match="SHA256"):
        prepare_pie_dataset(tmp_path / "out", archive, expected_sha256="invalid")
    with pytest.raises(ValueError, match="Unsafe"):
        prepare_pie_dataset(tmp_path / "out", archive)
    assert not (tmp_path / "escaped").exists()


def test_duplicate_ids_and_missing_archive(tmp_path):
    root = tmp_path / "dataset"
    fixture_data(root)
    (root / "mapping_file.json").write_text('{"a": {}, "a": {}}')
    with pytest.raises(ValueError, match="Duplicate JSON"):
        audit_dataset(root)
    with pytest.raises(FileNotFoundError, match="authorized"):
        prepare_pie_dataset(tmp_path / "absent")


def test_https_download_and_invalid_response(tmp_path, monkeypatch):
    source = tmp_path / "source"
    fixture_data(source)
    payload = io.BytesIO()
    with zipfile.ZipFile(payload, "w") as z:
        for p in source.rglob("*"):
            if p.is_file(): z.write(p, p.relative_to(source).as_posix())
    calls = []
    def fetch(url, timeout):
        calls.append(url)
        return io.BytesIO(payload.getvalue())
    monkeypatch.setattr("urllib.request.urlopen", fetch)
    dest = tmp_path / "downloaded"
    prepare_pie_dataset(dest, download_url="https://example.test/official.zip")
    prepare_pie_dataset(dest, download_url="https://example.test/official.zip")
    assert len(calls) == 1
    provenance = json.loads((dest / "download_provenance.json").read_text())
    assert not provenance["checksum_verified_against_supplied_value"]
    monkeypatch.setattr("urllib.request.urlopen", lambda *a, **k: io.BytesIO(b"<html>login</html>"))
    with pytest.raises(ValueError, match="not a ZIP"):
        prepare_pie_dataset(tmp_path / "failed", download_url="https://example.test/login")
    assert not (tmp_path / "failed.zip").exists()
