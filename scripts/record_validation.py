"""Record local tests, notebook syntax, dependency health, and environment."""
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import nbformat
from adaptive_p2p import environment_snapshot, file_hash, save_json

folder = ROOT / "validation"
folder.mkdir(exist_ok=True)
test = subprocess.run([sys.executable, "-m", "pytest", "tests", "-q", "--disable-warnings",
                       "--junitxml=" + str(folder / "pytest.xml")], cwd=ROOT, capture_output=True, text=True)
(folder / "pytest.txt").write_text(test.stdout + test.stderr, encoding="utf-8")
print(test.stdout, flush=True)
check = subprocess.run([sys.executable, "-m", "pip", "check"], capture_output=True, text=True)
(folder / "pip-check.txt").write_text(check.stdout + check.stderr, encoding="utf-8")
(folder / "gvproj-pip-freeze.txt").write_text(subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True), encoding="utf-8")
notebook = nbformat.read(ROOT / "adaptive_p2p_colab.ipynb", as_version=4)
nbformat.validate(notebook)
for i, cell in enumerate(notebook.cells):
    if cell.cell_type == "code":
        compile(cell.source, f"notebook_cell_{i}", "exec")
result = {"environment": environment_snapshot(), "pytest_passed": test.returncode == 0,
          "pip_check_passed": check.returncode == 0, "notebook_schema_and_syntax_passed": True,
          "full_sd14_inference_run": False, "colab_t4_run": False, "pie_bench_available": False,
          "pretrained_metric_parity": (folder / "pretrained_metric_parity.json").exists(),
          "source_sha256": {name: file_hash(ROOT / name) for name in (
              "adaptive_p2p.py", "evaluate_adaptive_p2p.py", "adaptive_p2p_colab.ipynb", "requirements-adaptive.txt",
              "scripts/runtime_paths.py")}}
save_json(folder / "local_validation.json", result)
if test.returncode or check.returncode:
    raise SystemExit(1)
print("Validation artifacts saved in validation/.")
