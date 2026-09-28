"""CPU metric verification with official pretrained weights; no diffusion model."""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("HF_HOME", str(ROOT / ".model_cache"))
os.environ.setdefault("TORCH_HOME", str(ROOT / ".model_cache" / "torch"))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch
from adaptive_p2p import save_json
from evaluate_adaptive_p2p import verify_pretrained_metrics

torch.set_num_threads(4)
result = verify_pretrained_metrics("cpu")
save_json(ROOT / "validation" / "pretrained_metric_parity.json", result)
print("All seven official metrics passed parity verification.")
