import os
import sys
import zipfile
from pathlib import Path

import pytest
from scripts import runtime_paths as rp


def make_project(path):
    for name in ("adaptive_p2p.py", "evaluate_adaptive_p2p.py", "requirements-adaptive.txt",
                 "prompt-to-prompt/seq_aligner.py", "vendor/reference_manifest.json"):
        file = path / name
        file.parent.mkdir(parents=True, exist_ok=True)
        file.write_text("test fixture", encoding="utf-8")


@pytest.fixture(autouse=True)
def isolate(monkeypatch):
    monkeypatch.chdir(Path.cwd())
    monkeypatch.setattr(sys, "path", list(sys.path))
    for key in ("HF_HOME", "TORCH_HOME", "TOKENIZERS_PARALLELISM"):
        monkeypatch.delenv(key, raising=False)
    for key in ("adaptive_p2p", "evaluate_adaptive_p2p"):
        monkeypatch.delitem(sys.modules, key, raising=False)


def test_remote_bundle_bootstrap_without_browser(tmp_path):
    content = tmp_path / "content"
    content.mkdir()
    fixture = tmp_path / "fixture"
    make_project(fixture)
    bundle = content / "adaptive_p2p_colab_bundle.zip"
    with zipfile.ZipFile(bundle, "w") as archive:
        for path in fixture.rglob("*"):
            if path.is_file():
                archive.write(path, path.relative_to(fixture))
    project, results = rp.setup_runtime(remote_root=content)
    assert project == content / "adaptive_p2p"
    assert results == content / "p2p_results"
    assert Path.cwd() == project
    # Re-running an unchanged bundle does not clobber remote edits or require widgets.
    (project / "adaptive_p2p.py").write_text("remote edit")
    rp.setup_runtime(remote_root=content)
    assert (project / "adaptive_p2p.py").read_text() == "remote edit"


def test_uploaded_source_override_and_mounted_drive(tmp_path):
    remote = tmp_path / "content"
    project = remote / "uploaded project"
    make_project(project)
    (remote / "drive" / "MyDrive").mkdir(parents=True)
    found, results = rp.setup_runtime(project_dir=project, remote_root=remote)
    assert found == project
    assert results == remote / "drive" / "MyDrive" / "p2p_research"


def test_local_project_discovery(tmp_path, monkeypatch):
    make_project(tmp_path)
    nested = tmp_path / "nested"
    nested.mkdir()
    monkeypatch.chdir(nested)
    project, results = rp.setup_runtime()
    assert project == tmp_path and results == tmp_path / "results"


def test_missing_upload_and_unmounted_drive(tmp_path):
    with pytest.raises(FileNotFoundError, match="Upload to Colab"):
        rp.setup_runtime(remote_root=tmp_path)
    make_project(tmp_path / "adaptive_p2p")
    with pytest.raises(FileNotFoundError, match="Mount Google Drive"):
        rp.setup_runtime(remote_root=tmp_path, results_root=tmp_path / "drive" / "MyDrive" / "runs")


def test_windows_path_rejected_on_linux(monkeypatch):
    monkeypatch.setattr(rp.os, "name", "posix")
    with pytest.raises(ValueError, match="Windows path"):
        rp.runtime_path(r"C:\Users\user\PIE-Bench")


def test_archive_validation_precedes_extraction(tmp_path):
    bundle = tmp_path / "bad.zip"
    with zipfile.ZipFile(bundle, "w") as archive:
        archive.writestr("adaptive_p2p.py", "safe")
        archive.writestr("../outside.py", "unsafe")
    destination = tmp_path / "unpacked"
    with pytest.raises(ValueError, match="Unsafe"):
        rp.extract_runtime_bundle(bundle, destination)
    assert not destination.exists()


def test_new_bundle_requires_restart_if_modules_loaded(tmp_path, monkeypatch):
    bundle = tmp_path / "new.zip"
    with zipfile.ZipFile(bundle, "w") as archive:
        archive.writestr("adaptive_p2p.py", "new")
    monkeypatch.setitem(sys.modules, "adaptive_p2p", object())
    with pytest.raises(RuntimeError, match="Restart Kernel"):
        rp.extract_runtime_bundle(bundle, tmp_path / "project")
