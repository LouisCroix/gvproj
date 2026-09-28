import abc
import ast
import json
from dataclasses import replace
from types import SimpleNamespace
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pytest
import torch
from transformers import CLIPTokenizer

import adaptive_p2p as ap


@pytest.fixture(scope="session")
def tokenizer():
    return CLIPTokenizer.from_pretrained(ap.ROOT / "tests" / "fixtures" / "tokenizer", local_files_only=True)


@pytest.fixture
def cfg():
    return ap.EditConfig(steps=10, device="cpu", dtype="float32")


def original_controller(tokenizer):
    """Execute only AST-selected definitions from the original notebook, never its cells."""
    ns = dict(torch=torch, np=np, abc=abc, List=List, Union=Union, Tuple=Tuple,
              Dict=Dict, Optional=Optional, LOW_RESOURCE=False, MAX_NUM_WORDS=77,
              device=torch.device("cpu"), tokenizer=tokenizer, LocalBlend=object,
              seq_aligner=ap.load_aligner())
    utils = ast.parse((ap.ROOT / "prompt-to-prompt" / "ptp_utils.py").read_text())
    definitions = [n for n in utils.body if isinstance(n, ast.FunctionDef) and n.name in
                   {"get_word_inds", "update_alpha_time_word", "get_time_words_attention_alpha"}]
    exec(compile(ast.Module(definitions, type_ignores=[]), "original_ptp_utils", "exec"), ns)
    ns["ptp_utils"] = SimpleNamespace(get_time_words_attention_alpha=ns["get_time_words_attention_alpha"])
    notebook = json.loads((ap.ROOT / "prompt-to-prompt" / "prompt-to-prompt_stable.ipynb").read_text())
    for cell in notebook["cells"]:
        source = "".join(cell.get("source", []))
        if cell["cell_type"] == "code" and "class AttentionReplace" in source:
            tree = ast.parse(source)
            selected = [n for n in tree.body if isinstance(n, ast.ClassDef) and n.name in
                        {"AttentionControl", "AttentionStore", "AttentionControlEdit", "AttentionReplace"}]
            exec(compile(ast.Module(selected, type_ignores=[]), "original_p2p_notebook", "exec"), ns)
    return ns["AttentionReplace"]


@pytest.mark.parametrize("target", ["a blue car beside a red car", "a rhinoceros car beside a red car"])
@pytest.mark.parametrize("tau", [0., .4, .6, .8, 1.])
def test_baseline_matches_original_all_steps(tokenizer, cfg, target, tau):
    case = ap.prepare_case("a red car beside a red car", target)
    mapper, _, _, indices = ap.alignment(case, tokenizer)
    config = replace(cfg, tau=tau)
    edited_word = target.split()[1]
    oracle = original_controller(tokenizer)([case.source_prompt, case.target_prompt], config.steps,
        {"default_": 1., edited_word: tau}, config.self_replace_steps)
    actual = ap.AttentionController(mapper, indices, config)
    generator = torch.Generator().manual_seed(11)
    for step in range(cfg.steps):
        actual.begin_step(step)
        oracle.cur_step = step
        for cross, positions, keys in ((True, 16, 77), (False, 16, 16), (False, 257, 257)):
            probs = torch.randn(4, 2, positions, keys, generator=generator).softmax(-1)
            cond = probs[2:].clone().reshape(4, positions, keys)
            expected = oracle.forward(cond, cross, "down").reshape(2, 2, positions, keys)
            result = actual.apply(probs, cross)
            torch.testing.assert_close(result[2:], expected, rtol=0, atol=0)
            torch.testing.assert_close(result[:3], probs[:3], rtol=0, atol=0)
        actual.end_step()


def test_repeated_word_uses_position_and_unequal_bpe(tokenizer):
    case = ap.prepare_case("a red car beside a red car", "a blue car beside a red car")
    _, _, _, indices = ap.alignment(case, tokenizer)
    assert indices == ap.load_aligner().get_word_inds(case.target_prompt, 1, tokenizer).tolist()
    case2 = ap.prepare_case("a cat on grass", "a hippopotamus on grass")
    mapper, _, src, tgt = ap.alignment(case2, tokenizer)
    assert len(src) != len(tgt)
    torch.testing.assert_close(mapper[src].sum(-1), torch.ones(len(src)))


def test_input_validation(tokenizer):
    with pytest.raises(ValueError, match="equal"):
        ap.prepare_case("a cat", "a red cat")
    with pytest.raises(ValueError, match="exceeds"):
        ap.alignment(ap.prepare_case("cat " * 100, "dog " * 100), tokenizer)


def test_adaptive_uses_previous_step_and_shared_alpha(cfg):
    config = replace(cfg, tau=1.)
    controller = ap.AttentionController(torch.eye(77), [2, 3], config, "soft_adaptive", .4)
    controller.begin_step(0)
    assert controller.weight == pytest.approx(.8)
    data = torch.rand(4, 2, 16, 77).softmax(-1)
    before = controller.weight
    controller.apply(data, True)
    controller.apply(data, True)
    assert controller.weight == before
    controller.end_step()
    ema = controller.ema
    controller.begin_step(1)
    assert controller.weight == pytest.approx(ap.cosine_window(1, 10, 1.) * (.8 - .5 * ema))
    assert len(controller._stats) == 0


def test_endpoints_no_edit_and_js(cfg):
    data = torch.rand(4, 2, 16, 77).softmax(-1)
    for strength in (0., 1.):
        ctrl = ap.AttentionController(torch.eye(77), list(range(77)), replace(cfg, fixed_strength=strength), "soft_fixed")
        ctrl.begin_step(0)
        result = ctrl.apply(data, True)
        torch.testing.assert_close(result[3], data[2] if strength else data[3])
    ctrl = ap.AttentionController(torch.eye(77), [], cfg, no_edit=True)
    ctrl.begin_step(0)
    assert ctrl.apply(data, True) is data
    assert ap.normalized_js(torch.ones(4), torch.ones(4)) == 0
    assert ap.normalized_js(torch.tensor([1., 0.]), torch.tensor([0., 1.])).item() > .999
    assert ap.cosine_window(9, 10, .6) == 0


def test_processor_against_explicit_reference(cfg):
    from diffusers.models.attention_processor import Attention
    torch.manual_seed(3)
    attn = Attention(query_dim=16, cross_attention_dim=12, heads=2, dim_head=8)
    controller = ap.AttentionController(torch.eye(77), [2], cfg, "soft_adaptive", .2)
    controller.begin_step(0)
    processor = ap.ControlledAttentionProcessor(SimpleNamespace(controller=controller))
    hidden, context = torch.randn(4, 16, 16), torch.randn(4, 77, 12)
    q, k, v = [attn.head_to_batch_dim(x) for x in (attn.to_q(hidden), attn.to_k(context), attn.to_v(context))]
    probs = attn.get_attention_scores(q, k)
    edited = controller.apply(probs.reshape(4, 2, 16, 77), True).reshape_as(probs)
    expected = attn.to_out[1](attn.to_out[0](attn.batch_to_head_dim(torch.bmm(edited, v))))
    result = processor(attn, hidden, context)
    torch.testing.assert_close(result, expected)


def test_small_unet_sampling_all_methods(cfg):
    """Real diffusers modules, random weights: tests plumbing, not image quality."""
    from diffusers import UNet2DConditionModel
    unet = UNet2DConditionModel(sample_size=8, in_channels=4, out_channels=4,
        down_block_types=("CrossAttnDownBlock2D",), up_block_types=("CrossAttnUpBlock2D",),
        block_out_channels=(16,), layers_per_block=1, cross_attention_dim=12,
        attention_head_dim=2, norm_num_groups=4).eval()
    runtime = SimpleNamespace(controller=None)
    unet.set_attn_processor({name: ap.ControlledAttentionProcessor(runtime) for name in unet.attn_processors})
    latent = torch.randn(4, 4, 8, 8)
    context = torch.randn(4, 77, 12)
    sources = []
    with torch.inference_mode():
        for method in ap.METHODS:
            ctrl = ap.AttentionController(torch.eye(77), [2], cfg, method, .3)
            ctrl.begin_step(0)
            runtime.controller = ctrl
            prediction = unet(latent, 900, encoder_hidden_states=context).sample
            ctrl.end_step()
            assert torch.isfinite(prediction).all()
            assert ctrl.logs[0]["statistic_layers"] > 0
            sources.append(prediction[:3])
    for prediction in sources[1:]:
        torch.testing.assert_close(prediction, sources[0], rtol=0, atol=0)


def test_ddim_inverse_timestep_compatibility(cfg):
    editor = ap.Editor.__new__(ap.Editor)
    editor.device = torch.device("cpu")
    forward, inverse = editor.scheduler(cfg), editor.scheduler(cfg, inverse=True)
    torch.testing.assert_close(forward.timesteps.flip(0), inverse.timesteps)
    assert forward.config.set_alpha_to_one == inverse.config.set_alpha_to_one is False

