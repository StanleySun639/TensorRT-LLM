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

import pytest
import torch
from transformers import PretrainedConfig

import tensorrt_llm  # noqa: F401  (loads C++ custom ops used by the model)
from tensorrt_llm._torch.attention.backends import TrtllmAttentionMetadata
from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._torch.models.modeling_deepseekv3 import DeepseekV3ForCausalLM
from tensorrt_llm._torch.models.modeling_utils import get_model_architecture
from tensorrt_llm.mapping import Mapping

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="DeepseekV32 MLA forward requires CUDA"
)


def _tiny_v32_config() -> PretrainedConfig:
    # Minimal DeepSeek-V3.2 (deepseek_v32) shape: dense-only (first_k_dense
    # beyond the layer count) so no MoE / distributed all-reduce path is
    # touched; MLA head dims kept small but valid.
    return PretrainedConfig(
        architectures=["DeepseekV32ForCausalLM"],
        model_type="deepseek_v32",
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        moe_intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        n_routed_experts=4,
        n_shared_experts=1,
        num_experts_per_tok=2,
        moe_layer_freq=1,
        first_k_dense_replace=8,
        q_lora_rank=32,
        kv_lora_rank=32,
        qk_nope_head_dim=16,
        qk_rope_head_dim=16,
        v_head_dim=16,
        rms_norm_eps=1e-6,
        rope_theta=10000.0,
        max_position_embeddings=64,
        num_nextn_predict_layers=0,
        torch_dtype="bfloat16",
        tie_word_embeddings=False,
    )


def _build_registered_v32(config: PretrainedConfig):
    # Resolve through the production registry: architectures[0] is
    # 'DeepseekV32ForCausalLM', registered onto DeepseekV3ForCausalLM.
    model_cls, arch = get_model_architecture(config)
    assert arch == "DeepseekV32ForCausalLM"
    assert model_cls is DeepseekV3ForCausalLM
    model_config = ModelConfig(
        pretrained_config=config,
        mapping=Mapping(world_size=1, rank=0, tp_size=1),
        attn_backend="TRTLLM",
    )
    model = model_cls(model_config).to("cuda").eval()
    return model, model_config


def _context_metadata(model_config: ModelConfig, seq_len: int):
    metadata = TrtllmAttentionMetadata(
        max_num_requests=1,
        max_num_tokens=seq_len,
        max_num_sequences=1,
        kv_cache_manager=None,
        mapping=model_config.mapping,
        seq_lens=torch.tensor([seq_len], dtype=torch.int32),
        num_contexts=1,
        seq_lens_kv=torch.tensor([seq_len], dtype=torch.int32),
        max_seq_len=seq_len,
    )
    metadata.request_ids = [0]
    metadata.prompt_lens = [seq_len]
    metadata.prepare()
    return metadata


def _forward_logits(model, model_config, input_ids=None, inputs_embeds=None):
    seq_len = (input_ids if input_ids is not None else inputs_embeds).shape[0]
    position_ids = torch.arange(seq_len, dtype=torch.int32, device="cuda").unsqueeze(0)
    metadata = _context_metadata(model_config, seq_len)
    with torch.inference_mode():
        return model.forward(
            attn_metadata=metadata,
            input_ids=input_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            return_context_logits=True,
        )


def _no_decoder_reference(model, input_ids):
    # Deterministic decoder-bypassed baseline built from the SAME live model
    # weights: real token embedding -> real final norm (model.norm) -> real LM
    # head / logits projection, with every DeepseekV3DecoderLayer / MLA
    # transformation skipped. This is not a model double: it reuses the
    # model's own weight tensors and its real head linear op. model.norm is an
    # RMSNorm whose forward returns (hidden, residual) when given a residual,
    # so pass residual=None and take the normalized hidden.
    with torch.inference_mode():
        hidden = model.model.embed_tokens(input_ids)
        normed, _ = model.model.norm(hidden, None)
        return model.logits_processor.forward(
            normed, model.lm_head, attn_metadata=None, return_context_logits=True
        )


class TestDeepseekV32ForCausalLM:
    def test_construct_registered_v32_and_forward(self):
        config = _tiny_v32_config()
        model, model_config = _build_registered_v32(config)
        assert isinstance(model, DeepseekV3ForCausalLM)
        assert len(model.model.layers) == config.num_hidden_layers

        seq_len = 8
        ids = torch.arange(seq_len, dtype=torch.int32, device="cuda") % config.vocab_size
        logits = _forward_logits(model, model_config, input_ids=ids)

        assert logits.shape[0] == seq_len
        assert logits.shape[-1] == config.vocab_size
        assert torch.isfinite(logits).all()

        # Top-level metamorphic oracle: feeding the equivalent inputs_embeds
        # (produced by the model's own real embed_tokens) through the same real
        # top-level forward must yield the same logits as feeding input_ids. A
        # forward that returned a fixed/zero result independent of its real
        # internal path cannot satisfy this jointly with the shifted-input
        # inequality below.
        with torch.inference_mode():
            embeds = model.model.embed_tokens(ids)
        logits_from_embeds = _forward_logits(model, model_config, inputs_embeds=embeds)
        torch.testing.assert_close(logits.float(), logits_from_embeds.float(), rtol=1e-2, atol=1e-2)

        shifted = (ids + 5) % config.vocab_size
        logits_shifted = _forward_logits(model, model_config, input_ids=shifted)
        assert not torch.allclose(logits.float(), logits_shifted.float())

        # Decoder-bypassed negative control from the same weights. Confirm it is
        # itself input/position sensitive so the not-allclose comparison below
        # is discriminating for the real decoder/MLA path.
        ref = _no_decoder_reference(model, ids)
        assert ref.shape == logits.shape
        assert torch.isfinite(ref).all()
        assert not torch.allclose(ref.float()[0], ref.float()[-1])
        ref_shifted = _no_decoder_reference(model, shifted)
        assert not torch.allclose(ref.float(), ref_shifted.float())

        # Layer-sensitivity control: the real top-level forward must diverge
        # from the embed->model.norm->head baseline. An identity/bypassed
        # decoder or MLA stack collapses the forward onto this baseline and
        # fails here.
        assert not torch.allclose(logits.float(), ref.float())
