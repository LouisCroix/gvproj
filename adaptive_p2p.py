"""Training-free P2P comparison for SD 1.4 / diffusers 0.35.1.

Attention replacement and token alignment follow Google Prompt-to-Prompt
(Apache-2.0); the unmodified local seq_aligner.py is loaded explicitly.
Only the target conditional branch is edited. LocalBlend is deliberately off.
"""
from __future__ import annotations

import gc
import hashlib
import importlib.util
import importlib.metadata
import json
import math
import platform
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

ROOT = Path(__file__).resolve().parent
METHODS = ("p2p", "soft_fixed", "soft_adaptive")
ALGORITHM_VERSION = "adaptive-p2p-v1"


def digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     allow_nan=False).encode()).hexdigest()


def file_hash(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    for attempt in range(6):
        try:
            temp.replace(path)
            break
        except PermissionError:
            # Windows readers/indexers can briefly hold the destination open.
            if attempt == 5:
                raise
            time.sleep(.05 * 2 ** attempt)


def environment_snapshot():
    names = ("torch", "torchvision", "diffusers", "transformers", "accelerate",
             "huggingface-hub", "torchmetrics", "numpy", "Pillow", "pandas")
    versions = {}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return {"python": platform.python_version(), "platform": platform.platform(),
            "packages": versions, "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}


def load_aligner():
    path = ROOT / "prompt-to-prompt" / "seq_aligner.py"
    if not path.is_file():
        raise FileNotFoundError(f"Upload the original prompt-to-prompt/seq_aligner.py too: {path}")
    spec = importlib.util.spec_from_file_location("original_p2p_seq_aligner", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@dataclass(frozen=True)
class EditCase:
    source_prompt: str
    target_prompt: str
    source_image: str | None = None
    seed: int = 42


def prepare_case(source_prompt, target_prompt, source_image=None, seed=42):
    source = " ".join(source_prompt.split())
    target = " ".join(target_prompt.split())
    if not source or not target:
        raise ValueError("Prompts must be nonempty.")
    if len(source.split(" ")) != len(target.split(" ")):
        raise ValueError("Replacement only: source/target must have equal whitespace word counts.")
    if source_image is not None:
        source_image = str(Path(source_image).resolve())
        if not Path(source_image).is_file():
            raise FileNotFoundError(source_image)
    return EditCase(source, target, source_image, int(seed))


@dataclass(frozen=True)
class EditConfig:
    output_dir: str = "adaptive_results"
    model_id: str = "CompVis/stable-diffusion-v1-4"
    model_revision: str = "main"
    steps: int = 30
    size: int = 512
    guidance_scale: float = 7.5
    inverse_guidance_scale: float = 1.0
    tau: float = 0.6
    self_replace_steps: float = 0.4
    fixed_strength: float = 0.75
    ema_decay: float = 0.8
    min_strength: float = 0.2
    device: str = "cuda"
    dtype: str = "float16"

    def __post_init__(self):
        if self.steps < 2 or self.size != 512:
            raise ValueError("This experiment requires >=2 DDIM steps and 512x512 images.")
        for name in ("tau", "self_replace_steps", "fixed_strength", "ema_decay", "min_strength"):
            if not 0 <= getattr(self, name) <= 1:
                raise ValueError(f"{name} must be in [0,1].")
        if self.inverse_guidance_scale != 1.0:
            raise ValueError("The preregistered shared inversion uses CFG=1.")
        if self.device == "cpu" and self.dtype != "float32":
            raise ValueError("Use dtype='float32' for CPU smoke tests.")


def validate_alignment(case, tokenizer):
    limit = tokenizer.model_max_length
    for prompt in (case.source_prompt, case.target_prompt):
        if len(tokenizer.encode(prompt, truncation=False)) > limit:
            raise ValueError(f"Prompt exceeds {limit} tokens; silent truncation is disabled.")
    aligner = load_aligner()
    changed = [i for i, (a, b) in enumerate(zip(case.source_prompt.split(), case.target_prompt.split())) if a != b]
    src_ids, tgt_ids = [], []
    for i in changed:
        src_ids.extend(aligner.get_word_inds(case.source_prompt, i, tokenizer).tolist())
        tgt_ids.extend(aligner.get_word_inds(case.target_prompt, i, tokenizer).tolist())
    if changed and (not src_ids or not tgt_ids):
        raise ValueError("Changed words could not be aligned to tokenizer positions.")
    return changed, src_ids, tgt_ids


def alignment(case, tokenizer):
    changed, src_ids, tgt_ids = validate_alignment(case, tokenizer)
    mapper = load_aligner().get_replacement_mapper_(case.source_prompt, case.target_prompt,
                                                   tokenizer, tokenizer.model_max_length)
    if mapper.shape != (tokenizer.model_max_length, tokenizer.model_max_length):
        raise ValueError("Unexpected mapper shape.")
    if not torch.isfinite(mapper).all():
        raise ValueError("Nonfinite token mapper.")
    return mapper, changed, src_ids, tgt_ids


def cosine_window(step, steps, tau):
    u = step / (steps - 1)
    return 0.5 * (1 + math.cos(math.pi * u / tau)) if tau > 0 and u < tau else 0.0


def normalized_js(source, target):
    """Spatial distributions -> JS/log(2), calculated in float32."""
    p, q = source.float().flatten(), target.float().flatten()
    eps = torch.finfo(torch.float32).eps
    p, q = p.clamp_min(eps), q.clamp_min(eps)
    p, q = p / p.sum(), q / q.sum()
    m = 0.5 * (p + q)
    return (0.5 * ((p * (p / m).log()).sum() + (q * (q / m).log()).sum()) / math.log(2)).clamp(0, 1)


class AttentionController:
    """One scalar per step; end_step() is called by the sampling loop, never by layers."""
    def __init__(self, mapper, target_indices, config, method="p2p", semantic_distance=0.0, no_edit=False):
        if method not in METHODS:
            raise ValueError(f"Unknown method: {method}")
        self.mapper = mapper
        self.indices = list(target_indices)
        self.config, self.method = config, method
        self.semantic = float(semantic_distance)
        self.no_edit = no_edit
        self.ema = 0.0
        self.logs = []
        self.step = 0
        self._stats = []

    def begin_step(self, step):
        self.step, self._stats = int(step), []
        self.previous_ema = self.ema
        if self.method == "p2p":
            # Original ptp_utils alpha has steps+1 entries and integer truncation.
            self.weight = float(step < int(self.config.tau * (self.config.steps + 1)))
        else:
            strength = self.config.fixed_strength if self.method == "soft_fixed" else max(
                self.config.min_strength, min(1.0, 1 - 0.5 * self.semantic - 0.5 * self.previous_ema))
            self.weight = cosine_window(step, self.config.steps, self.config.tau) * strength

    def apply(self, probabilities, is_cross):
        """[4, heads, positions, keys], ordered uncond(src,tgt), cond(src,tgt)."""
        if self.no_edit:
            return probabilities
        if probabilities.shape[0] != 4:
            raise ValueError("Expected a single source/target pair with batched classifier-free guidance.")
        source, target = probabilities[2], probabilities[3]
        if is_cross:
            mapped = torch.einsum("hqw,wn->hqn", source, self.mapper.to(source))
            if self.indices and source.shape[1] <= 32 ** 2:
                a = mapped[..., self.indices].float().sum(-1).mean(0)
                b = target[..., self.indices].float().sum(-1).mean(0)
                self._stats.append(normalized_js(a, b).detach())
            weights = target.new_ones(target.shape[-1])
            weights[self.indices] = self.weight
            out = probabilities.clone()
            out[3] = mapped * weights + target * (1 - weights)
            return out
        if source.shape[1] <= 16 ** 2 and self.step < int(self.config.steps * self.config.self_replace_steps):
            out = probabilities.clone()
            out[3] = source
            return out
        return probabilities

    def end_step(self):
        distance = torch.stack(self._stats).mean().item() if self._stats else 0.0
        self.ema = self.config.ema_decay * self.previous_ema + (1 - self.config.ema_decay) * distance
        self.logs.append({"step": self.step, "u": self.step / (self.config.steps - 1),
                          "semantic_distance": self.semantic, "attention_js": distance,
                          "ema_used": self.previous_ema, "ema_next": self.ema,
                          "alpha_edited": self.weight if not self.no_edit else 0.0,
                          "statistic_layers": len(self._stats)})
        self._stats.clear()


class ControlledAttentionProcessor:
    """Diffusers processor: explicit probabilities only where intervention is needed."""
    def __init__(self, runtime):
        from diffusers.models.attention_processor import AttnProcessor2_0
        self.runtime = runtime
        self.fallback = AttnProcessor2_0()

    def __call__(self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None, temb=None, *args, **kwargs):
        controller = self.runtime.controller
        cross = encoder_hidden_states is not None
        length = hidden_states.shape[1] if hidden_states.ndim == 3 else hidden_states.shape[-2] * hidden_states.shape[-1]
        controlled_self = (not cross and length <= 256 and controller is not None
                           and controller.step < int(controller.config.steps * controller.config.self_replace_steps))
        if controller is None or controller.no_edit or (not cross and not controlled_self):
            return self.fallback(attn, hidden_states, encoder_hidden_states, attention_mask, temb)
        residual = hidden_states
        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)
        ndim = hidden_states.ndim
        if ndim == 4:
            batch, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch, channel, height * width).transpose(1, 2)
        batch = hidden_states.shape[0]
        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)
        query = attn.to_q(hidden_states)
        context = hidden_states if encoder_hidden_states is None else encoder_hidden_states
        if cross and attn.norm_cross:
            context = attn.norm_encoder_hidden_states(context)
        key, value = attn.to_k(context), attn.to_v(context)
        query, key, value = [attn.head_to_batch_dim(x) for x in (query, key, value)]
        mask = attn.prepare_attention_mask(attention_mask, context.shape[1], batch)
        probs = attn.get_attention_scores(query, key, mask)
        probs = controller.apply(probs.reshape(batch, attn.heads, *probs.shape[1:]), cross).reshape_as(probs)
        hidden_states = torch.bmm(probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)
        hidden_states = attn.to_out[1](attn.to_out[0](hidden_states))
        if ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch, channel, height, width)
        if attn.residual_connection:
            hidden_states = hidden_states + residual
        return hidden_states / attn.rescale_output_factor


def sync(device):
    if str(device).startswith("cuda"):
        torch.cuda.synchronize()


class Editor:
    def __init__(self, config):
        from diffusers import StableDiffusionPipeline
        from huggingface_hub import HfApi
        if config.device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable. Select Colab T4; local CPU is for tests only.")
        revision = config.model_revision
        if len(revision) != 40:
            revision = HfApi().model_info(config.model_id, revision=revision).sha
        self.revision = revision
        self.dtype = getattr(torch, config.dtype)
        self.device = torch.device(config.device)
        self.pipe = StableDiffusionPipeline.from_pretrained(
            config.model_id, revision=revision, torch_dtype=self.dtype,
            safety_checker=None, requires_safety_checker=False).to(self.device)
        for component in (self.pipe.unet, self.pipe.vae, self.pipe.text_encoder):
            component.requires_grad_(False).eval()
        # VAE float32 avoids fp16 decode/encode instabilities; its memory cost is small.
        self.pipe.vae.to(dtype=torch.float32)
        self.pipe.enable_vae_slicing()
        self.controller = None
        self.pipe.unet.set_attn_processor({name: ControlledAttentionProcessor(self)
                                          for name in self.pipe.unet.attn_processors})
        self._semantics = {}

    def scheduler(self, config, inverse=False):
        from diffusers import DDIMScheduler, DDIMInverseScheduler
        cls = DDIMInverseScheduler if inverse else DDIMScheduler
        sched = cls(num_train_timesteps=1000, beta_start=0.00085, beta_end=0.012,
                    beta_schedule="scaled_linear", clip_sample=False,
                    set_alpha_to_one=False, steps_offset=1, timestep_spacing="leading")
        sched.set_timesteps(config.steps, device=self.device)
        return sched

    @torch.inference_mode()
    def embeddings(self, prompts):
        tokens = self.pipe.tokenizer(prompts, padding="max_length",
                                     max_length=self.pipe.tokenizer.model_max_length,
                                     truncation=False, return_tensors="pt")
        return self.pipe.text_encoder(tokens.input_ids.to(self.device))[0]

    @torch.inference_mode()
    def semantic_distance(self, case, changed):
        key = (case.source_prompt, case.target_prompt)
        if key not in self._semantics:
            if not changed:
                self._semantics[key] = 0.0
            else:
                phrases = [" ".join(prompt.split()[i] for i in changed)
                           for prompt in key]
                tokens = self.pipe.tokenizer(phrases, padding=True, truncation=False, return_tensors="pt")
                ids = tokens.input_ids.to(self.device)
                states = self.pipe.text_encoder(ids)[0].float()
                keep = tokens.attention_mask.to(self.device).bool()
                for special in self.pipe.tokenizer.all_special_ids:
                    keep &= ids != special
                if not keep.any(1).all():
                    raise ValueError("Edited phrase contains no ordinary text tokens.")
                pooled = (states * keep[..., None]).sum(1) / keep.sum(1, keepdim=True)
                self._semantics[key] = (1 - F.cosine_similarity(pooled[:1], pooled[1:])).clamp(0, 1).item()
        return self._semantics[key]

    @torch.inference_mode()
    def decode(self, latents):
        decoded = self.pipe.vae.decode(latents.float() / self.pipe.vae.config.scaling_factor).sample
        if not torch.isfinite(decoded).all():
            raise FloatingPointError("VAE produced nonfinite pixels.")
        pixels = ((decoded / 2 + 0.5).clamp(0, 1) * 255).round().byte().cpu().permute(0, 2, 3, 1).numpy()
        return [Image.fromarray(x) for x in pixels]

    @torch.inference_mode()
    def sample(self, initial, prompts, config, controller=None):
        scheduler = self.scheduler(config)
        n = len(prompts)
        latents = initial.to(self.device, self.dtype).expand(n, -1, -1, -1).clone()
        context = torch.cat([self.embeddings([""] * n), self.embeddings(prompts)])
        self.controller = controller
        try:
            for k, t in enumerate(scheduler.timesteps):
                if controller is not None:
                    controller.begin_step(k)
                noise = self.pipe.unet(scheduler.scale_model_input(torch.cat([latents] * 2), t),
                                       t, encoder_hidden_states=context).sample
                uncond, cond = noise.chunk(2)
                noise = uncond + config.guidance_scale * (cond - uncond)
                latents = scheduler.step(noise, t, latents, eta=0).prev_sample
                if controller is not None:
                    controller.end_step()
            if not torch.isfinite(latents).all():
                raise FloatingPointError("Nonfinite sampled latent.")
            return self.decode(latents)
        finally:
            self.controller = None

    def base_identity(self, case, config):
        settings = asdict(config)
        for key in ("output_dir", "tau", "self_replace_steps", "fixed_strength", "ema_decay", "min_strength"):
            settings.pop(key)
        settings["model_revision"] = self.revision
        return {"case": asdict(case), "image_sha256": file_hash(case.source_image) if case.source_image else None,
                "settings": settings, "version": ALGORITHM_VERSION,
                "implementation_sha256": file_hash(__file__),
                "aligner_sha256": file_hash(ROOT / "prompt-to-prompt" / "seq_aligner.py"),
                "environment": environment_snapshot()}

    def initial_is_cached(self, case, config):
        cache = Path(config.output_dir).resolve() / "cache" / digest(self.base_identity(case, config))
        metadata = cache / "initial.json"
        if not metadata.exists():
            return False
        record = json.loads(metadata.read_text(encoding="utf-8"))
        return all((cache / name).is_file() and file_hash(cache / name) == value
                   for name, value in record["checksums"].items())

    @torch.inference_mode()
    def prepare_initial(self, case, config):
        alignment(case, self.pipe.tokenizer)
        identity = self.base_identity(case, config)
        cache = Path(config.output_dir).resolve() / "cache" / digest(identity)
        metadata = cache / "initial.json"
        if metadata.exists():
            record = json.loads(metadata.read_text(encoding="utf-8"))
            if all((cache / name).is_file() and file_hash(cache / name) == value
                   for name, value in record["checksums"].items()):
                return torch.load(cache / "initial.pt", map_location="cpu", weights_only=True), record
        cache.mkdir(parents=True, exist_ok=True)
        sync(self.device)
        start = time.perf_counter()
        inverse_seconds = 0.0
        if case.source_image:
            with Image.open(case.source_image) as img:
                image = img.convert("RGB")
                if image.size != (512, 512):
                    raise ValueError("Real benchmark images must be 512x512; no silent mask-changing crop/resize.")
            image.save(cache / "source.png")
            pixels = torch.from_numpy(np.array(image).copy()).permute(2, 0, 1)[None].to(self.device, torch.float32)
            pixels = pixels / 127.5 - 1
            latent = self.pipe.vae.encode(pixels).latent_dist.mean * self.pipe.vae.config.scaling_factor
            latent = latent.to(self.dtype)
            inverse = self.scheduler(config, inverse=True)
            self.controller = None
            context = self.embeddings([case.source_prompt])
            for t in inverse.timesteps:
                noise = self.pipe.unet(inverse.scale_model_input(latent, t), t, encoder_hidden_states=context).sample
                latent = inverse.step(noise, t, latent).prev_sample
            sync(self.device)
            inverse_seconds = time.perf_counter() - start
        else:
            generator = torch.Generator(device="cpu").manual_seed(case.seed)
            latent = torch.randn((1, 4, 64, 64), generator=generator, dtype=torch.float32).to(self.device, self.dtype)
        if not torch.isfinite(latent).all():
            raise FloatingPointError("Nonfinite initial latent.")
        torch.save(latent.cpu(), cache / "initial.pt")
        reference = self.sample(latent, [case.source_prompt], config)[0]
        name = "reconstruction.png" if case.source_image else "source.png"
        reference.save(cache / name)
        sync(self.device)
        files = ["initial.pt", "source.png"] + (["reconstruction.png"] if case.source_image else [])
        record = {"identity": identity, "cache_dir": str(cache),
                  "source_path": str(cache / "source.png"),
                  "reconstruction_path": str(cache / "reconstruction.png") if case.source_image else None,
                  "inversion_seconds": inverse_seconds, "initial_total_seconds": time.perf_counter() - start,
                  "checksums": {name: file_hash(cache / name) for name in files}}
        save_json(metadata, record)
        return latent.cpu(), record

    @torch.inference_mode()
    def edit(self, case, method, config):
        mapper, changed, _, tgt_ids = alignment(case, self.pipe.tokenizer)
        initial, cached = self.prepare_initial(case, config)
        key = digest({"initial": cached["identity"], "method": method, "config": asdict(config)})
        folder = Path(config.output_dir).resolve() / "runs" / key
        metadata = folder / "result.json"
        if metadata.exists():
            record = json.loads(metadata.read_text(encoding="utf-8"))
            if all((folder / name).is_file() and file_hash(folder / name) == value
                   for name, value in record["checksums"].items()):
                return record
        folder.mkdir(parents=True, exist_ok=True)
        sync(self.device)
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        semantic = self.semantic_distance(case, changed)
        controller = AttentionController(mapper, tgt_ids, config, method, semantic, no_edit=not changed)
        images = self.sample(initial, [case.source_prompt, case.target_prompt], config, controller)
        sync(self.device)
        elapsed = time.perf_counter() - start
        images[0].save(folder / "source_branch.png")
        images[1].save(folder / "edited.png")
        save_json(folder / "weights.json", controller.logs)
        record = {"run_id": key, "method": method, "case": asdict(case), "config": asdict(config),
                  "model_revision": self.revision, "source_path": cached["source_path"],
                  "reconstruction_path": cached["reconstruction_path"],
                  "source_branch_path": str(folder / "source_branch.png"),
                  "edited_path": str(folder / "edited.png"), "weights_path": str(folder / "weights.json"),
                  "edit_seconds": elapsed, "inversion_seconds": cached["inversion_seconds"],
                  "initial_total_seconds": cached["initial_total_seconds"],
                  "peak_memory_mib": torch.cuda.max_memory_allocated() / 2**20 if self.device.type == "cuda" else 0,
                  "environment": environment_snapshot(),
                  "checksums": {name: file_hash(folder / name) for name in ("source_branch.png", "edited.png", "weights.json")}}
        save_json(metadata, record)
        return record


_EDITOR = None
_EDITOR_KEY = None


def get_editor(config):
    global _EDITOR, _EDITOR_KEY
    key = (config.model_id, config.model_revision, config.device, config.dtype)
    if _EDITOR is None or _EDITOR_KEY != key:
        release_editor()
        _EDITOR = Editor(config)
        _EDITOR_KEY = key
    return _EDITOR


def release_editor():
    global _EDITOR, _EDITOR_KEY
    _EDITOR, _EDITOR_KEY = None, None
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def invert_case(case, config):
    """Real cases: invert once and cache. Synthetic cases: cache seeded noise/source."""
    return get_editor(config).prepare_initial(case, config)[1]


def edit_case(case, method, config):
    return get_editor(config).edit(case, method, config)
