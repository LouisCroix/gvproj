"""Fetch immutable official evaluator code and small tokenizer fixtures (no model weights)."""
import hashlib
import json
from pathlib import Path
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
PNP_COMMIT = "07f97f448150e2ca220bebd54c8f687c5c50c67a"
DINO_COMMIT = "7c446df5b9f45747937fb0d72314eb9f7b66930a"


def fetch(url, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    data = urlopen(Request(url, headers={"User-Agent": "adaptive-p2p-research"}), timeout=60).read()
    path.write_bytes(data)
    return {"url": url, "path": path.relative_to(ROOT).as_posix(), "sha256": hashlib.sha256(data).hexdigest()}


if __name__ == "__main__":
    records = []
    for filename in ("matrics_calculator.py", "evaluate.py"):
        records.append(fetch(f"https://raw.githubusercontent.com/cure-lab/PnPInversion/{PNP_COMMIT}/evaluation/{filename}",
                             ROOT / "vendor" / "pnp_inversion" / filename))
    # Resolve the model revision before downloading tokenizer files; all fixtures share it.
    model = json.load(urlopen("https://huggingface.co/api/models/CompVis/stable-diffusion-v1-4"))
    revision = model["sha"]
    for filename in ("vocab.json", "merges.txt", "tokenizer_config.json", "special_tokens_map.json"):
        records.append(fetch(f"https://huggingface.co/CompVis/stable-diffusion-v1-4/resolve/{revision}/tokenizer/{filename}",
                             ROOT / "tests" / "fixtures" / "tokenizer" / filename))
    (ROOT / "vendor" / "reference_manifest.json").write_text(json.dumps({
        "pnp_commit": PNP_COMMIT, "dino_commit": DINO_COMMIT,
        "sd14_revision": revision, "files": records}, indent=2), encoding="utf-8")
    print(json.dumps({"downloaded_files": len(records), "sd14_revision": revision}))
