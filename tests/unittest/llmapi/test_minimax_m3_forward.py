# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from types import SimpleNamespace

import pytest
import torch

from tensorrt_llm._torch.attention.backends.trtllm import TrtllmAttention, TrtllmAttentionMetadata
from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._torch.models.modeling_minimaxm3 import (
    MiniMaxM3DecoderLayer,
    MiniMaxM3VLForConditionalGeneration,
)
from tensorrt_llm._torch.models.modeling_utils import get_registered_model_class
from tensorrt_llm.llmapi import MiniMaxM3SparseAttentionConfig


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@torch.inference_mode()
def test_registered_conditional_generation_text_forward():
    architecture = "MiniMaxM3SparseForConditionalGeneration"
    model_class = get_registered_model_class(architecture)
    assert model_class is MiniMaxM3VLForConditionalGeneration

    # The wrapper explicitly supports text-only checkpoints without a vision config.
    config = SimpleNamespace(
        model_type="minimax_m3",
        architectures=[architecture],
        hidden_size=512,
        intermediate_size=256,
        dense_intermediate_size=256,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=128,
        vocab_size=64,
        pad_token_id=None,
        image_token_index=62,
        video_token_index=63,
        max_position_embeddings=64,
        rms_norm_eps=1e-6,
        use_gemma_norm=False,
        attention_output_gate=False,
        rope_theta=5000000,
        rotary_dim=64,
        partial_rotary_factor=0.5,
        hidden_act="swigluoai",
        use_qk_norm=True,
        qk_norm_type="per_head",
        tie_word_embeddings=False,
        shared_intermediate_size=3072,
        num_local_experts=128,
        num_experts_per_tok=4,
        n_shared_experts=1,
        scoring_func="sigmoid",
        use_routing_bias=True,
        moe_layer_freq=[0],
        num_mtp_modules=0,
        swiglu_alpha=1.702,
        swiglu_limit=7.0,
        routed_scaling_factor=2.0,
        sparse_attention_config={
            "use_sparse_attention": True,
            "sparse_index_dim": 128,
            "sparse_num_index_heads": 4,
            "sparse_topk_blocks": 16,
            "sparse_block_size": 128,
            "sparse_init_block": 0,
            "sparse_local_block": 1,
            "sparse_score_type": "max",
            "sparse_disable_index_value": [0],
            "sparse_attention_freq": [0],
        },
        torch_dtype=torch.bfloat16,
    )
    model_config = ModelConfig(
        pretrained_config=config,
        sparse_attention_config=MiniMaxM3SparseAttentionConfig(implementation="triton"),
        max_num_tokens=3,
        max_seq_len=3,
    )
    model = model_class(model_config).cuda().eval()
    assert isinstance(model, MiniMaxM3VLForConditionalGeneration)
    assert model.vision_tower is None
    assert len(model.model.layers) == 1
    assert isinstance(model.model.layers[0], MiniMaxM3DecoderLayer)
    assert type(model.model.layers[0].self_attn.attn) is TrtllmAttention
    torch.testing.assert_close(
        model.mm_token_ids,
        torch.tensor([62, 63], dtype=torch.int32, device="cuda"),
    )

    generator = torch.Generator(device="cuda").manual_seed(153)
    for parameter in model.parameters():
        if parameter.ndim == 1:
            parameter.fill_(1)
        else:
            parameter.normal_(mean=0, std=0.02, generator=generator)

    metadata = TrtllmAttentionMetadata(
        max_num_requests=1,
        max_num_tokens=3,
        mapping=model_config.mapping,
        num_contexts=1,
        seq_lens=torch.tensor([3], dtype=torch.int32),
        request_ids=[0],
    )
    metadata.max_seq_len = 3
    metadata.prepare()
    assert metadata.kv_cache_manager is None
    assert not metadata.kv_cache_params.use_cache

    input_ids = torch.tensor([1, 2, 3], dtype=torch.long, device="cuda")
    embeddings = model.model.embed_tokens(input_ids)
    logits = model(attn_metadata=metadata, input_ids=input_ids, return_context_logits=True)
    embedded_logits = model(
        attn_metadata=metadata,
        input_ids=None,
        inputs_embeds=embeddings.clone(),
        return_context_logits=True,
    )
    assert logits.shape == (3, config.vocab_size)
    assert logits.dtype == torch.float32
    assert logits.is_cuda
    assert torch.isfinite(logits).all()
    torch.testing.assert_close(logits, embedded_logits, atol=1e-2, rtol=1e-2)

    bypassed_logits = model.lm_head(model.model.norm(embeddings)).float()
    assert not torch.allclose(logits, bypassed_logits, atol=1e-2, rtol=1e-2)

    changed_ids = input_ids.clone()
    changed_ids[0] = 4
    changed_logits = model(
        attn_metadata=metadata, input_ids=changed_ids, return_context_logits=True
    )
    assert torch.isfinite(changed_logits).all()
    assert not torch.allclose(logits[-1], changed_logits[-1], atol=1e-2, rtol=1e-2)
