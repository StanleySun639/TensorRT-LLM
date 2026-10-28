# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from transformers import Qwen2Config
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VisionPatchEmbed

from tensorrt_llm._torch.attention.backends.trtllm import TrtllmAttentionMetadata
from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._torch.models import modeling_qwen2vl
from tensorrt_llm._torch.models.modeling_qwen import Qwen2ForProcessRewardModel, QwenDecoderLayer
from tensorrt_llm._torch.models.modeling_qwen2vl import (
    Qwen2_5_VisionModel,
    Qwen2_5_VLVisionAttention,
)
from tensorrt_llm._torch.models.modeling_utils import get_registered_model_class


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@torch.inference_mode()
def test_qwen2_process_reward_token_scores(monkeypatch):
    monkeypatch.setenv("TLLM_FMHA_LIBS", "fallback")
    config = Qwen2Config(
        architectures=["Qwen2ForProcessRewardModel"],
        vocab_size=64,
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        max_position_embeddings=32,
        rope_scaling=None,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        tie_word_embeddings=False,
        torch_dtype=torch.bfloat16,
    )
    assert get_registered_model_class("Qwen2ForProcessRewardModel") is Qwen2ForProcessRewardModel

    def initialize_weights(model):
        for name, parameter in model.named_parameters():
            if name.endswith("norm.weight"):
                parameter.fill_(1)
            else:
                parameter.normal_(mean=0, std=0.1)

    with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
        torch.manual_seed(42)
        model = (
            Qwen2ForProcessRewardModel(
                ModelConfig(
                    pretrained_config=config,
                    attn_backend="TRTLLM",
                    max_seq_len=32,
                    max_num_tokens=7,
                )
            )
            .cuda()
            .eval()
        )
        initialize_weights(model)

    assert isinstance(model, Qwen2ForProcessRewardModel)
    assert len(model.model.layers) == 1
    assert isinstance(model.model.layers[0], QwenDecoderLayer)
    assert model.num_labels == 2

    def prepare_inputs(sequences):
        lengths = [sequence.numel() for sequence in sequences]
        position_ids = torch.cat([torch.arange(length, device="cuda") for length in lengths])
        metadata = TrtllmAttentionMetadata(
            max_num_requests=len(sequences),
            max_num_tokens=sum(lengths),
            num_contexts=len(sequences),
            seq_lens=torch.tensor(lengths, dtype=torch.int32),
            request_ids=list(range(len(sequences))),
            position_ids=position_ids,
        )
        metadata.max_seq_len = 32
        metadata.prepare()
        return dict(
            attn_metadata=metadata,
            input_ids=torch.cat(sequences),
            position_ids=position_ids,
        )

    first_sequence = torch.tensor([1, 7, 11, 13], device="cuda")
    second_sequence = torch.tensor([1, 17, 19], device="cuda")
    output = model(**prepare_inputs([first_sequence, second_sequence]))
    assert output.is_nested
    assert output.size(0) == 2
    first_score = output[0]
    second_score = output[1]
    assert first_score.shape == (4, 2)
    assert second_score.shape == (3, 2)
    assert first_score.dtype == torch.bfloat16
    assert second_score.dtype == torch.bfloat16
    assert first_score.is_cuda
    assert second_score.is_cuda
    assert torch.isfinite(first_score).all()
    assert torch.isfinite(second_score).all()

    first_single_output = model(**prepare_inputs([first_sequence]))
    second_single_output = model(**prepare_inputs([second_sequence]))
    torch.testing.assert_close(first_score, first_single_output[0], atol=1e-2, rtol=2e-3)
    torch.testing.assert_close(second_score, second_single_output[0], atol=1e-2, rtol=2e-3)

    # Batch agreement alone would also accept a decoder that did nothing.
    first_embeddings = model.model.embed_tokens(first_sequence)
    second_embeddings = model.model.embed_tokens(second_sequence)
    first_bypassed_score = model.score(model.model.norm(first_embeddings))
    second_bypassed_score = model.score(model.model.norm(second_embeddings))
    assert not torch.allclose(first_score, first_bypassed_score, atol=1e-2, rtol=2e-3)
    assert not torch.allclose(second_score, second_bypassed_score, atol=1e-2, rtol=2e-3)

    changed_output = model(
        **prepare_inputs([torch.tensor([23, 29, 31, 13], device="cuda"), second_sequence])
    )
    changed_first_score = changed_output[0]
    changed_second_score = changed_output[1]
    assert not torch.allclose(first_score[-1], changed_first_score[-1], atol=1e-2, rtol=2e-3)
    torch.testing.assert_close(second_score, changed_second_score, atol=1e-2, rtol=2e-3)


@pytest.mark.cpu_only
def test_qwen2_5_vision_patch_projection_matches_conv3d(monkeypatch) -> None:
    patch_embed = Qwen2_5_VisionPatchEmbed(
        patch_size=4,
        temporal_patch_size=2,
        in_channels=3,
        embed_dim=8,
    )
    pixel_values = torch.randn(6, 3 * 2 * 4 * 4)
    expected = patch_embed(pixel_values)
    # The model must reuse the Conv3d parameters as a GEMM, not run the Conv3d.
    patch_embed.proj.register_forward_pre_hook(
        lambda module, args: pytest.fail("patch projection ran the Conv3d")
    )

    vision = Qwen2_5_VisionModel.__new__(Qwen2_5_VisionModel)
    torch.nn.Module.__init__(vision)
    vision.patch_embed = patch_embed
    vision.spatial_merge_unit = 1
    vision._rope_position_ids_buffer = None
    vision.full_attn_metadata = object()
    vision.window_attn_metadata = object()
    vision._full_attn_max_seq_len = 6
    vision._window_attn_max_seq_len = 6
    vision.fullatt_block_indexes = []
    vision.blocks = torch.nn.ModuleList()
    vision.merger = torch.nn.Identity()
    vision.get_rotary_pos_emb_window_data = lambda grid: (
        [torch.empty(6, 0)],
        [torch.empty(6, 0)],
        [torch.arange(6)],
        [6],
    )
    vision.prepare_attn_metadata = lambda seq_lens, metadata, **kwargs: metadata
    monkeypatch.setattr(
        modeling_qwen2vl,
        "async_tensor_h2d",
        lambda tensor, *, dtype, device: tensor.to(dtype=dtype, device=device),
    )

    actual = vision(pixel_values, torch.tensor([[1, 2, 3]]))

    torch.testing.assert_close(actual, expected)


def _make_vision_attention(
    fused_qkv: torch.Tensor, *, head_dim: int, support_fused_qkv: bool
) -> tuple[Qwen2_5_VLVisionAttention, dict[str, object]]:
    attention = Qwen2_5_VLVisionAttention.__new__(Qwen2_5_VLVisionAttention)
    torch.nn.Module.__init__(attention)
    attention.head_dim = head_dim
    attention.q_size = fused_qkv.shape[-1] // 3
    attention.kv_size = attention.q_size
    attention.support_fused_qkv = support_fused_qkv
    attention.layer_idx = 0
    attention.qkv_proj = lambda hidden_states: fused_qkv
    forwarded = {}

    def forward_impl(**kwargs) -> torch.Tensor:
        forwarded.update(kwargs)
        return kwargs["q"]

    attention.forward_impl = forward_impl
    attention.o_proj = lambda output, layer_idx: output
    return attention, forwarded


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    "returned_view",
    ["equivalent", "fresh_q", "fresh_k", "fresh_v", "offset", "stride", "dtype", "mutated_stride"],
)
def test_qwen2_5_vision_attention_checks_rotated_qkv_views(returned_view: str) -> None:
    fused_qkv = torch.randn(3, 24)
    attention, forwarded = _make_vision_attention(fused_qkv, head_dim=4, support_fused_qkv=True)
    returned_qkv = []

    def apply_rope(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        position_ids: torch.Tensor | None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None,
        packed_qk: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if returned_view == "equivalent":
            q, k, v = q.view_as(q), k.view_as(k), v.view_as(v)
        elif returned_view == "fresh_q":
            q = q + 1
        elif returned_view == "fresh_k":
            k = k + 1
        elif returned_view == "fresh_v":
            v = v + 1
        elif returned_view == "offset":
            q = q.as_strided(q.shape, q.stride(), q.storage_offset() + 1)
        elif returned_view == "stride":
            q = q.as_strided(q.shape, (q.stride(0), 0))
        elif returned_view == "dtype":
            q = q.view(torch.int32)
        elif returned_view == "mutated_stride":
            q.as_strided_(q.shape, (q.stride(0), 0))
        returned_qkv.extend((q, k, v))
        return q, k, v

    attention.apply_rope = apply_rope
    output = attention.forward(torch.empty(3, 8), attn_metadata=object())

    if returned_view == "equivalent":
        assert output is fused_qkv
    else:
        assert output is not fused_qkv
        torch.testing.assert_close(output, torch.cat(returned_qkv, dim=-1))
    assert forwarded["q"] is output
    assert forwarded["k"] is None
    assert forwarded["v"] is None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.skipif(not modeling_qwen2vl.IS_FLASHINFER_AVAILABLE, reason="FlashInfer required")
@pytest.mark.parametrize("support_fused_qkv", [True, False])
def test_qwen2_5_vision_attention_flashinfer_inplace_cuda(
    monkeypatch: pytest.MonkeyPatch, support_fused_qkv: bool
) -> None:
    head_dim = 64
    fused_qkv = torch.randn(3, 6 * head_dim, device="cuda", dtype=torch.float16)
    original_qkv = fused_qkv.clone()
    attention, forwarded = _make_vision_attention(
        fused_qkv, head_dim=head_dim, support_fused_qkv=support_fused_qkv
    )
    q, k, v = attention.split_qkv(fused_qkv)
    original_v = v.clone()
    cos = torch.randn(3, head_dim // 2, device="cuda", dtype=torch.float32)
    sin = torch.randn_like(cos)
    expected_q = (
        modeling_qwen2vl.RotaryEmbedding.apply_rotary_pos_emb(
            q.float().view(1, 3, 2, head_dim), cos, sin, unsqueeze_dim=1
        )
        .reshape_as(q)
        .to(q.dtype)
    )
    expected_k = (
        modeling_qwen2vl.RotaryEmbedding.apply_rotary_pos_emb(
            k.float().view(1, 3, 2, head_dim), cos, sin, unsqueeze_dim=1
        )
        .reshape_as(k)
        .to(k.dtype)
    )
    original_rope = modeling_qwen2vl.flashinfer_apply_rope_with_cos_sin_cache_inplace
    kernel_calls = []

    def checked_rope(*args, **kwargs) -> None:
        original_rope(*args, **kwargs)
        kernel_calls.append(True)

    monkeypatch.setattr(modeling_qwen2vl, "_flash_attn_apply_rotary", None)
    monkeypatch.setattr(
        modeling_qwen2vl, "flashinfer_apply_rope_with_cos_sin_cache_inplace", checked_rope
    )
    output = attention.forward(
        torch.empty(3, 2 * head_dim, device="cuda", dtype=torch.float16),
        attn_metadata=object(),
        position_ids=torch.arange(3, device="cuda", dtype=torch.int32),
        position_embeddings=(cos, sin),
    )

    assert len(kernel_calls) == 1
    torch.testing.assert_close(q, expected_q, atol=1e-3, rtol=1e-3)
    torch.testing.assert_close(k, expected_k, atol=1e-3, rtol=1e-3)
    torch.testing.assert_close(v, original_v)
    if support_fused_qkv:
        assert output is fused_qkv
        assert forwarded["k"] is None
        assert forwarded["v"] is None
    else:
        assert output.is_set_to(q)
        assert forwarded["k"].is_set_to(k)
        assert forwarded["v"].is_set_to(v)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        attention.forward(
            torch.empty(3, 2 * head_dim, device="cuda", dtype=torch.float16),
            attn_metadata=object(),
            position_ids=torch.arange(3, device="cuda", dtype=torch.int32),
            position_embeddings=(cos, sin),
        )
    assert len(kernel_calls) == 2
    fused_qkv.copy_(original_qkv)
    graph.replay()
    torch.testing.assert_close(q, expected_q, atol=1e-3, rtol=1e-3)
    torch.testing.assert_close(k, expected_k, atol=1e-3, rtol=1e-3)
    torch.testing.assert_close(v, original_v)
