# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Compare complete K3 MLA outputs with the pinned Hugging Face implementation."""

import math
import weakref
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from _torch.attention.multi_gpu.helix_test_utils import CACHE_TYPE_SELFKONLY, setup_kv_and_metadata
from _torch.modeling.kimi_k3_mla_reference import KimiMLAAttention
from _torch.modeling.test_kimi_k3_checkpoint import load_mla_checkpoint
from utils.util import skip_pre_blackwell

from tensorrt_llm._torch.attention.backends.interface import KVCacheParams
from tensorrt_llm._torch.attention.backends.utils import get_attention_backend
from tensorrt_llm._torch.configs.kimi_linear import KimiLinearConfig
from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._torch.models.modeling_kimi_linear import KimiLinearForCausalLM
from tensorrt_llm._torch.models.modeling_utils import get_registered_model_class
from tensorrt_llm._torch.utils import model_extra_attrs
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.models.modeling_utils import QuantAlgo, QuantConfig


def _checkpoint_weights(reference: KimiMLAAttention, fp8: bool) -> dict:
    projections = {}
    for name, parameter in reference.named_parameters():
        if parameter.ndim != 2:
            continue
        weight = parameter.detach().clone()
        pair = {"weight": weight}
        if fp8:
            rows, cols = weight.shape
            padded = F.pad(weight.float(), (0, -cols % 128, 0, -rows % 128))
            blocks = padded.view(math.ceil(rows / 128), 128, math.ceil(cols / 128), 128)
            scale = blocks.abs().amax(dim=(1, 3)).clamp_min(1e-8) / 448.0
            expanded = scale.repeat_interleave(128, 0).repeat_interleave(128, 1)[:rows, :cols]
            pair = {
                "weight": (weight.float() / expanded).to(torch.float8_e4m3fn),
                "weight_scale": scale,
            }
            # Both implementations see the same stored weights. Only TRT-LLM
            # quantizes activations and uses FP8 GEMM/BMM kernels.
            with torch.no_grad():
                parameter.copy_((pair["weight"].float() * expanded).to(parameter.dtype))
        projections[name.removesuffix(".weight")] = pair
    return projections


def _assert_matches_reference(actual: torch.Tensor, expected: torch.Tensor, fp8: bool) -> None:
    # DeepGEMM weight resmoothing and activation rounding compound across the
    # A/B projections and absorption (about 7% relative L2 on this fixture).
    # Bound both elementwise outliers and the overall error relative to the signal.
    atol, rtol = (0.1, 0.1) if fp8 else (0.005, 0.03)
    torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)
    relative_l2 = (
        torch.linalg.vector_norm(actual.float() - expected.float())
        / torch.linalg.vector_norm(expected.float()).clamp_min(1e-8)
    ).item()
    assert relative_l2 < (0.08 if fp8 else 0.01), f"relative L2 error: {relative_l2}"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize(
    "fp8", [False, pytest.param(True, marks=skip_pre_blackwell)], ids=["bf16", "fp8"]
)
@pytest.mark.parametrize("gate", [False, True], ids=["no_gate", "gate"])
def test_k3_mla_matches_huggingface(fp8: bool, gate: bool) -> None:
    torch.manual_seed(42)
    mapping = Mapping(world_size=1, rank=0, tp_size=1)
    config = KimiLinearConfig(
        vocab_size=128,
        hidden_size=256,
        intermediate_size=256,
        num_hidden_layers=1,
        attn_res_block_size=1,
        torch_dtype=torch.bfloat16,
        linear_attn_config={"kda_layers": [], "full_attn_layers": [1], "num_heads": 32},
        num_attention_heads=32,
        num_key_value_heads=32,
        q_lora_rank=1536,
        kv_lora_rank=512,
        qk_nope_head_dim=128,
        qk_rope_head_dim=64,
        v_head_dim=128,
        rms_norm_eps=1e-6,
        mla_use_nope=True,
        mla_use_output_gate=gate,
        max_position_embeddings=512,
    )
    config._attn_implementation = "eager"
    reference = KimiMLAAttention(config, layer_idx=0).bfloat16().eval()
    weights = _checkpoint_weights(reference, fp8)
    model_config = ModelConfig(
        pretrained_config=config,
        mapping=mapping,
        quant_config=QuantConfig(),
        quant_config_dict={
            f"model.layers.0.self_attn.{name}": QuantConfig(
                quant_algo=QuantAlgo.FP8_BLOCK_SCALES, group_size=128
            )
            for name in weights
        }
        if fp8
        else None,
        use_cute_dsl_blockscaling_mm=True,
        use_cute_dsl_blockscaling_bmm=True,
    )
    with torch.device("cuda"):
        model = KimiLinearForCausalLM(model_config).eval()
    projections = {
        f"{name}.{suffix}": value
        for name, pair in weights.items()
        for suffix, value in pair.items()
    }
    projections.update(
        {name: value for name, value in reference.named_parameters() if value.ndim == 1}
    )
    load_mla_checkpoint(model, projections, "")
    runtime = model.model.layers[0].self_attn
    for module in runtime.modules():
        if callable(getattr(module, "post_load_weights", None)):
            module.post_load_weights()
    reference.cuda()

    batch, context_length = 2, 128
    generator = torch.Generator(device="cuda").manual_seed(1234)
    hidden = torch.randn(
        batch,
        context_length + 1,
        config.hidden_size,
        dtype=torch.bfloat16,
        device="cuda",
        generator=generator,
    )
    causal_mask = torch.full(
        (context_length + 1, context_length + 1), float("-inf"), device="cuda"
    ).triu(1)[None, None]
    expected = reference(hidden, attention_mask=causal_mask)

    context = hidden[:, :context_length].reshape(-1, config.hidden_size).contiguous()
    scenario = SimpleNamespace(
        ctx_len=context_length,
        batch=batch,
        kv_cache_tokens_per_block=64,
        num_layers=1,
        kv_cache_dtype=torch.bfloat16,
    )
    manager, metadata = setup_kv_and_metadata(
        scenario,
        mapping,
        cache_type=CACHE_TYPE_SELFKONLY,
        num_kv_heads=1,
        head_dim=config.kv_lora_rank + config.qk_rope_head_dim,
    )
    try:
        extra_attrs = {
            "attention_metadata": weakref.ref(metadata),
            "mla_layers": {runtime.mixer.layer_idx_str: weakref.ref(runtime.mixer)},
        }
        with model_extra_attrs(extra_attrs):
            actual = runtime(context, metadata).view(batch, context_length, -1)
            _assert_matches_reference(actual, expected[:, :context_length], fp8)

        for request_id in range(batch):
            assert manager.kv_cache_map[request_id].capacity >= context_length + 1
        metadata = get_attention_backend("TRTLLM").Metadata(
            seq_lens=torch.ones(batch, dtype=torch.int),
            request_ids=list(range(batch)),
            max_num_requests=batch,
            num_contexts=0,
            prompt_lens=[context_length] * batch,
            max_num_tokens=batch,
            kv_cache_manager=manager,
            kv_cache_params=KVCacheParams(
                use_cache=True, num_cached_tokens_per_seq=[context_length] * batch
            ),
            mapping=mapping,
            enable_context_mla_with_cached_kv=True,
        )
        metadata.prepare()
        extra_attrs["attention_metadata"] = weakref.ref(metadata)
        with model_extra_attrs(extra_attrs):
            actual = runtime(hidden[:, -1].contiguous(), metadata)
        _assert_matches_reference(actual, expected[:, -1], fp8)
    finally:
        manager.shutdown()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@torch.no_grad()
def test_kimi_linear_forward_matches_reference(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TLLM_FMHA_LIBS", "fallback")
    config = KimiLinearConfig(
        architectures=["KimiLinearForCausalLM"],
        vocab_size=128,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        hidden_size=256,
        intermediate_size=256,
        num_hidden_layers=1,
        attn_res_block_size=1,
        torch_dtype=torch.bfloat16,
        linear_attn_config={"kda_layers": [], "full_attn_layers": [1], "num_heads": 32},
        num_attention_heads=32,
        num_key_value_heads=32,
        q_lora_rank=1536,
        kv_lora_rank=512,
        qk_nope_head_dim=128,
        qk_rope_head_dim=64,
        v_head_dim=128,
        rms_norm_eps=1e-6,
        mla_use_nope=True,
        mla_use_output_gate=True,
        max_position_embeddings=512,
    )
    config._attn_implementation = "eager"
    model_config = ModelConfig(
        pretrained_config=config,
        attn_backend="TRTLLM",
        max_num_tokens=4,
        max_seq_len=4,
    )
    model_class = get_registered_model_class("KimiLinearForCausalLM")
    assert model_class is KimiLinearForCausalLM

    with torch.random.fork_rng():
        torch.manual_seed(42)
        reference = KimiMLAAttention(config, layer_idx=0).bfloat16().eval()
        with torch.device("cuda"):
            model = model_class(model_config).eval()
        projections = {
            name: parameter.detach().clone() for name, parameter in reference.named_parameters()
        }
        load_mla_checkpoint(model, projections, "")
        layer = model.model.layers[0]
        for parameter in model.parameters():
            if parameter.ndim == 1:
                parameter.fill_(1)
        model.model.embed_tokens.weight.normal_(std=0.2)
        model.lm_head.weight.normal_(std=0.1)
        layer.mlp.gate_up_proj.weight.normal_(std=0.04)
        layer.mlp.down_proj.weight.normal_(std=0.04)
        for module in layer.self_attn.modules():
            if callable(getattr(module, "post_load_weights", None)):
                module.post_load_weights()
        reference.cuda()

    input_ids = torch.tensor([3, 7, 11, 5], device="cuda", dtype=torch.long)
    embeddings = model.model.embed_tokens(input_ids)

    def normalize(x):
        xf = x.float()
        return (xf * torch.rsqrt(xf.square().mean(-1, keepdim=True) + config.rms_norm_eps)).to(
            x.dtype
        )

    causal_mask = torch.full((4, 4), float("-inf"), device="cuda").triu(1)[None, None]
    attention_output = reference(
        normalize(embeddings).unsqueeze(0), attention_mask=causal_mask
    ).squeeze(0)
    # The checkpoint helper zeroes score projections, so both residual
    # selections are uniform averages of the embedding and running stream.
    mlp_input = normalize(((embeddings.float() + attention_output.float()) * 0.5).bfloat16())
    gate, up = F.linear(mlp_input, layer.mlp.gate_up_proj.weight).float().chunk(2, dim=-1)
    activated = (gate.tanh() * gate.sigmoid() * up).bfloat16()
    mlp_output = F.linear(activated, layer.mlp.down_proj.weight)
    prefix = attention_output + mlp_output
    expected_hidden = normalize(((embeddings.float() + prefix.float()) * 0.5).bfloat16())
    expected_logits = F.linear(expected_hidden, model.lm_head.weight).float()
    bypass_logits = F.linear(normalize(embeddings), model.lm_head.weight).float()

    metadata = get_attention_backend("TRTLLM").Metadata(
        seq_lens=torch.tensor([4], dtype=torch.int),
        request_ids=[0],
        max_num_requests=1,
        num_contexts=1,
        max_num_tokens=4,
        mapping=model_config.mapping,
    )
    metadata.max_seq_len = 4
    extra_attrs = dict(model_config.extra_attrs)
    extra_attrs["attention_metadata"] = weakref.ref(metadata)
    with model_extra_attrs(extra_attrs):
        metadata.prepare()
        logits = model(
            attn_metadata=metadata,
            input_ids=input_ids,
            return_context_logits=True,
        )

    assert logits.shape == (4, config.vocab_size)
    assert logits.dtype == torch.float32
    assert torch.isfinite(logits).all()
    torch.testing.assert_close(logits, expected_logits, atol=0.03, rtol=0.03)
    assert not torch.allclose(logits, bypass_logits, atol=0.03, rtol=0.03)
