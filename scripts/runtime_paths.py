"""Standard-library bootstrap, embedded in the notebook before project imports.

Works with VS Code Colab extension, browser Colab and a local gvproj kernel.
No frontend-specific upload, download or authentication widgets are invoked.
"""
import hashlib
import os
import sys
import zipfile
from pathlib import Path, PureWindowsPath


def runtime_path(value, *, base=None):
    value = str(value)
    if os.name != "nt" and (PureWindowsPath(value).drive or "\\" in value):
        raise ValueError("Remote Linux cannot read a Windows path. Upload the file and use its /content/... path: " + value)
    path = Path(value).expanduser()
    return ((Path(base) / path) if base is not None and not path.is_absolute() else path).resolve()


def extract_runtime_bundle(bundle, destination):
    bundle, destination = Path(bundle), Path(destination).resolve()
    checksum = hashlib.sha256(bundle.read_bytes()).hexdigest()
    marker = destination / ".uploaded_bundle_sha256"
    if marker.exists() and marker.read_text().strip() == checksum:
        return destination
    if any(name in sys.modules for name in ("adaptive_p2p", "evaluate_adaptive_p2p")):
        raise RuntimeError("New project bundle detected. Restart Kernel before extracting updated code.")
    with zipfile.ZipFile(bundle) as archive:
        # Check the complete archive before extracting anything.
        for member in archive.infolist():
            name = member.filename
            path = destination / name
            if PureWindowsPath(name).drive or "\\" in name or not path.resolve().is_relative_to(destination):
                raise ValueError("Unsafe archive member: " + name)
            if (member.external_attr >> 16) & 0o170000 == 0o120000:
                raise ValueError("Archive symlinks are not supported: " + name)
        if "adaptive_p2p.py" not in archive.namelist():
            raise ValueError("Use the generated flat adaptive_p2p_colab_bundle.zip.")
        destination.mkdir(parents=True, exist_ok=True)
        archive.extractall(destination)
    marker.write_text(checksum, encoding="utf-8")
    return destination


def setup_runtime(project_dir=None, bundle_path=None, remote_root=None, results_root=None):
    cwd = Path.cwd()
    remote = runtime_path(remote_root) if remote_root else None
    if project_dir:
        project = runtime_path(project_dir, base=remote or cwd)
    elif remote:
        project = remote / "adaptive_p2p"
    else:
        candidates = [p for p in (cwd, *cwd.parents) if (p / "adaptive_p2p.py").is_file()]
        if not candidates:
            raise FileNotFoundError("Set PROJECT_DIR_OVERRIDE to the project directory visible to this kernel.")
        project = candidates[0]
    bundle = runtime_path(bundle_path, base=remote or cwd) if bundle_path else (
        remote / "adaptive_p2p_colab_bundle.zip" if remote else None)
    # Explicit project override uses uploaded source files; explicit bundle overrides it.
    if bundle and bundle.is_file() and (not project_dir or bundle_path):
        extract_runtime_bundle(bundle, project)
    required = ("adaptive_p2p.py", "evaluate_adaptive_p2p.py", "requirements-adaptive.txt",
                "prompt-to-prompt/seq_aligner.py", "vendor/reference_manifest.json")
    missing = [name for name in required if not (project / name).is_file()]
    if missing:
        visible = sorted(p.name for p in remote.iterdir()) if remote and remote.is_dir() else []
        raise FileNotFoundError(
            f"Project not available to kernel at {project}. Missing: {missing}. "
            f"Upload adaptive_p2p_colab_bundle.zip using VS Code 'Upload to Colab', then set BUNDLE_PATH_OVERRIDE "
            f"to the uploaded remote path if needed. Remote files: {visible}")
    drive = remote / "drive" / "MyDrive" if remote else None
    if results_root:
        results = runtime_path(results_root, base=remote or project)
    elif drive is not None and drive.is_dir():
        results = drive / "p2p_research"
    else:
        results = (remote / "p2p_results") if remote else (project / "results")
    if remote and "drive" in results.parts and drive is not None and not drive.is_dir():
        raise FileNotFoundError("Drive is not mounted. Use 'Colab: Mount Google Drive to Server...' first.")
    results.mkdir(parents=True, exist_ok=True)
    os.chdir(project)
    if str(project) not in sys.path:
        sys.path.insert(0, str(project))
    os.environ.setdefault("HF_HOME", str(remote / "p2p_model_cache" if remote else project / ".model_cache"))
    os.environ.setdefault("TORCH_HOME", str(Path(os.environ["HF_HOME"]) / "torch"))
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    print("Kernel Python:", sys.executable)
    print("Project:", project)
    print("Results root:", results)
    if remote and (drive is None or not results.is_relative_to(drive)):
        print("Remote temporary storage: download the checkpoint ZIP before deleting/replacing the Colab server.")
    return project, results
