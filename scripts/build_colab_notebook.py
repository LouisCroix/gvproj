"""Validate the canonical notebook and build the compact Colab upload bundle.

The notebook is edited directly; this tool never overwrites user settings.
"""
import json
import zipfile
from pathlib import Path

import nbformat

ROOT = Path(__file__).resolve().parents[1]


def build_bundle():
    notebook = nbformat.read(ROOT / "adaptive_p2p_colab.ipynb", as_version=4)
    nbformat.validate(notebook)
    for i, cell in enumerate(notebook.cells):
        if cell.cell_type == "code":
            compile(cell.source, f"notebook_cell_{i}", "exec")
            cell.outputs = []
            cell.execution_count = None
    with zipfile.ZipFile(ROOT / "adaptive_p2p_colab_bundle.zip", "w", zipfile.ZIP_DEFLATED) as archive:
        files = ["adaptive_p2p.py", "evaluate_adaptive_p2p.py", "requirements-adaptive.txt",
                 "ADAPTIVE_P2P_README.md", "scripts/runtime_paths.py",
                 "prompt-to-prompt/seq_aligner.py", "prompt-to-prompt/ptp_utils.py", "prompt-to-prompt/LICENSE"]
        files += [p.relative_to(ROOT).as_posix() for directory in ("tests", "vendor")
                  for p in (ROOT / directory).rglob("*")
                  if p.is_file() and "__pycache__" not in p.parts and p.suffix != ".pyc"]
        for name in files:
            archive.write(ROOT / name, name)
        archive.writestr("adaptive_p2p_colab.ipynb", nbformat.writes(notebook))
        reference = json.loads((ROOT / "prompt-to-prompt/prompt-to-prompt_stable.ipynb").read_text(encoding="utf-8"))
        for cell in reference["cells"]:
            if cell["cell_type"] == "code":
                cell["outputs"], cell["execution_count"] = [], None
        archive.writestr("prompt-to-prompt/prompt-to-prompt_stable.ipynb", json.dumps(reference))
    print("Notebook validated; compact Colab bundle built (historical outputs excluded).")


if __name__ == "__main__":
    build_bundle()
