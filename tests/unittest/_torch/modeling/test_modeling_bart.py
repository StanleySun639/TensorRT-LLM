# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Unit tests for the BartForConditionalGeneration PyTorch architecture."""

import pytest
import torch
from transformers import BartConfig

from tensorrt_llm._torch.attention.backends.utils import get_attention_backend
from tensorrt_llm._torch.metadata import KVCacheParams
from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._torch.models.modeling_bart import BartForConditionalGeneration
from tensorrt_llm._torch.models.modeling_utils import get_registered_model_class
from tensorrt_llm._torch.pyexecutor.resource_manager import KVCacheManager
from tensorrt_llm.bindings.executor import KvCacheConfig
from tensorrt_llm.mapping import Mapping


def _tiny_bart_config() -> BartConfig:
    return BartConfig(
        vocab_size=128,
        d_model=32,
        encoder_layers=2,
        decoder_layers=2,
        encoder_attention_heads=4,
        decoder_attention_heads=4,
        encoder_ffn_dim=64,
        decoder_ffn_dim=64,
        max_position_embeddings=64,
        activation_function="gelu",
        tie_word_embeddings=True,
        torch_dtype=torch.float16,
    )


def _make_kv_manager(config: BartConfig, num_layers: int, is_cross: bool):
    head_dim = config.d_model // config.encoder_attention_heads
    cache_type = (
        torch.classes.tensorrt_llm.KvCacheType.CROSS
        if is_cross
        else torch.classes.tensorrt_llm.KvCacheType.SELF
    )
    return KVCacheManager(
        kv_cache_config=KvCacheConfig(max_tokens=256),
        kv_cache_type=cache_type,
        num_layers=num_layers,
        num_kv_heads=config.decoder_attention_heads,
        head_dim=head_dim,
        tokens_per_block=32,
        max_seq_len=config.max_position_embeddings,
        max_batch_size=1,
        mapping=Mapping(world_size=1, rank=0, tp_size=1),
        dtype=torch.float16,
    )


def _build_metadata(backend_cls, seq_len, kv_manager):
    md = backend_cls.Metadata(
        max_num_requests=1,
        max_num_tokens=seq_len,
        kv_cache_manager=kv_manager,
        mapping=Mapping(world_size=1, rank=0, tp_size=1),
        runtime_features=None,
    )
    md.seq_lens = torch.tensor([seq_len], dtype=torch.int32)
    md.num_contexts = 1
    md.request_ids = [0]
    md.prompt_lens = [seq_len]
    md.kv_cache_params = KVCacheParams(use_cache=True, num_cached_tokens_per_seq=[0])
    md.prepare()
    return md


def _make_decoder_run(config: BartConfig, decoder_input_ids: torch.Tensor):
    """Build fresh KV managers, metadata, and position ids for one forward.

    Each metamorphic run needs its own KV caches so encoder K/V from a prior
    run does not leak into the next; the returned managers are shut down by the
    caller after the forward completes.
    """
    device = "cuda"
    backend_cls = get_attention_backend("TRTLLM")
    dec_len = decoder_input_ids.numel()
    num_layers = config.decoder_layers

    self_kv = _make_kv_manager(config, num_layers, is_cross=False)
    cross_kv = _make_kv_manager(config, num_layers, is_cross=True)

    decoder_attn_metadata = _build_metadata(backend_cls, dec_len, self_kv)
    return self_kv, cross_kv, decoder_attn_metadata, device, dec_len


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="BART TRTLLM attention backend requires a GPU"
)
def test_bart_for_conditional_generation_construction_and_forward():
    torch.manual_seed(0)
    config = _tiny_bart_config()

    resolved = get_registered_model_class("BartForConditionalGeneration")
    assert resolved is BartForConditionalGeneration

    model_config = ModelConfig(pretrained_config=config)
    model_config.attn_backend = "TRTLLM"
    model = BartForConditionalGeneration(model_config).to("cuda").eval()
    assert isinstance(model, BartForConditionalGeneration)

    with torch.no_grad():
        for p in model.parameters():
            p.copy_(torch.randn_like(p) * 0.02)

    decoder_input_ids = torch.tensor([1, 5, 9, 13], dtype=torch.int32)
    enc_len = 4
    encoder_hidden_a = torch.randn(enc_len, config.d_model, dtype=torch.float16)
    encoder_hidden_b = torch.randn(enc_len, config.d_model, dtype=torch.float16)

    # Run A: real top-level forward through the constructed architecture.
    self_kv, cross_kv, attn_md, device, dec_len = _make_decoder_run(config, decoder_input_ids)
    cross_md = attn_md.create_cross_metadata(
        encoder_seq_lens=[enc_len],
        cross_kv_cache_manager=cross_kv,
    )
    position_ids = (
        torch.arange(dec_len, dtype=torch.int32, device=device) + model.model.position_id_offset
    )
    with torch.inference_mode():
        logits_a = model.forward(
            attn_metadata=attn_md,
            input_ids=decoder_input_ids.to(device),
            position_ids=position_ids,
            encoder_hidden_states=encoder_hidden_a.to(device),
            cross_attn_metadata=cross_md,
        )
    self_kv.shutdown()
    cross_kv.shutdown()

    # Run B: same decoder inputs, different encoder hidden states.
    self_kv, cross_kv, attn_md, device, dec_len = _make_decoder_run(config, decoder_input_ids)
    cross_md = attn_md.create_cross_metadata(
        encoder_seq_lens=[enc_len],
        cross_kv_cache_manager=cross_kv,
    )
    position_ids = (
        torch.arange(dec_len, dtype=torch.int32, device=device) + model.model.position_id_offset
    )
    with torch.inference_mode():
        logits_b = model.forward(
            attn_metadata=attn_md,
            input_ids=decoder_input_ids.to(device),
            position_ids=position_ids,
            encoder_hidden_states=encoder_hidden_b.to(device),
            cross_attn_metadata=cross_md,
        )
    self_kv.shutdown()
    cross_kv.shutdown()

    assert logits_a.shape == (1, config.vocab_size)
    assert logits_b.shape == (1, config.vocab_size)
    assert torch.isfinite(logits_a).all()
    assert torch.isfinite(logits_b).all()

    # Distinct encoder hidden states must reach the decoder logits through
    # cross-attention and lm_head, proving the real forward tensor path ran.
    assert not torch.allclose(logits_a, logits_b, atol=1e-3)
