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
"""Unit tests for the Phi3 model (PyTorch backend)."""

import unittest

import torch
import transformers
from transformers import Phi3Config

from tensorrt_llm._torch.attention.backends import utils as attention_utils
from tensorrt_llm._torch.metadata import KVCacheParams
from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._torch.models.modeling_phi3 import (Phi3DecoderLayer,
                                                      Phi3ForCausalLM,
                                                      Phi3Model)
from tensorrt_llm._torch.models.modeling_utils import MODEL_CLASS_MAPPING
from tensorrt_llm._torch.pyexecutor.resource_manager import KVCacheManager
from tensorrt_llm.bindings.executor import KvCacheConfig
from tensorrt_llm.mapping import Mapping

PHI3_SMALL_CONFIG = {
    "architectures": ["Phi3ForCausalLM"],
    "model_type": "phi3",
    "vocab_size": 256,
    "hidden_size": 128,
    "intermediate_size": 256,
    "num_hidden_layers": 2,
    "num_attention_heads": 4,
    "num_key_value_heads": 4,
    "max_position_embeddings": 512,
    "rms_norm_eps": 1e-5,
    "attention_bias": False,
    "tie_word_embeddings": False,
    "torch_dtype": "bfloat16",
}


def _make_model_config():
    cfg = Phi3Config(**PHI3_SMALL_CONFIG)
    mapping = Mapping(world_size=1, tp_size=1, rank=0)
    return ModelConfig(pretrained_config=cfg,
                       mapping=mapping,
                       attn_backend="TRTLLM",
                       max_seq_len=PHI3_SMALL_CONFIG["max_position_embeddings"])


def _build_context_metadata(model_config, num_tokens):
    config = model_config.pretrained_config
    head_dim = config.hidden_size // config.num_attention_heads
    max_seq_len = PHI3_SMALL_CONFIG["max_position_embeddings"]

    metadata_cls = attention_utils.get_attention_backend(
        model_config.attn_backend).Metadata

    kv_cache_manager = KVCacheManager(
        kv_cache_config=KvCacheConfig(max_tokens=4096),
        kv_cache_type=0,
        num_layers=config.num_hidden_layers,
        num_kv_heads=config.num_key_value_heads,
        head_dim=head_dim,
        tokens_per_block=128,
        max_seq_len=max_seq_len,
        max_batch_size=1,
        mapping=Mapping(world_size=1, tp_size=1, rank=0),
        dtype=torch.float16,
    )
    request_ids = [0]
    token_nums = [num_tokens]
    kv_cache_manager.add_dummy_requests(request_ids, token_nums)

    attn_metadata = metadata_cls(
        seq_lens=torch.tensor([num_tokens], dtype=torch.int32),
        num_contexts=1,
        kv_cache_params=KVCacheParams(
            use_cache=True,
            num_cached_tokens_per_seq=[0],
        ),
        kv_cache_manager=kv_cache_manager,
        request_ids=request_ids,
        prompt_lens=token_nums,
        max_num_requests=1,
        max_num_tokens=num_tokens,
    )
    attn_metadata.prepare()
    return kv_cache_manager, attn_metadata


@unittest.skipIf(not torch.cuda.is_available(),
                 "Phi3 forward requires a CUDA device")
class TestPhi3ForCausalLM(unittest.TestCase):

    def test_construction_and_forward(self):
        model_config = _make_model_config()
        config = model_config.pretrained_config
        device = torch.device("cuda")
        dtype = config.torch_dtype

        # The supported-architecture invariant: production registration must
        # map the architecture name to Phi3ForCausalLM. Removing
        # @register_auto_model("Phi3ForCausalLM") fails this assertion.
        self.assertIs(MODEL_CLASS_MAPPING["Phi3ForCausalLM"], Phi3ForCausalLM)

        # Construct the exact registered architecture directly.
        model = Phi3ForCausalLM(model_config)

        self.assertIsInstance(model, Phi3ForCausalLM)
        self.assertIsInstance(model.model, Phi3Model)
        self.assertEqual(len(model.model.layers), config.num_hidden_layers)
        for layer in model.model.layers:
            self.assertIsInstance(layer, Phi3DecoderLayer)
        self.assertEqual(model.config.vocab_size, config.vocab_size)
        self.assertIsNotNone(model.lm_head)

        model = model.to(device).to(dtype)
        model.eval()

        # Independent HuggingFace reference with the same config; equivalent
        # weights are copied into it to give an output-sensitive parity oracle.
        torch.manual_seed(0)
        ref = transformers.Phi3ForCausalLM(Phi3Config(**PHI3_SMALL_CONFIG))
        ref = ref.to(torch.float32)
        ref.eval()

        ref_sd = ref.state_dict()
        with torch.no_grad():
            model.model.embed_tokens.weight.copy_(
                ref_sd["model.embed_tokens.weight"].to(device).to(dtype))
            model.lm_head.weight.copy_(
                ref_sd["lm_head.weight"].to(device).to(dtype))
            model.model.norm.weight.copy_(
                ref_sd["model.norm.weight"].to(device).to(dtype))
            for i, layer in enumerate(model.model.layers):
                p = f"model.layers.{i}."
                if (p + "self_attn.qkv_proj.weight") in ref_sd:
                    qkv = ref_sd[p + "self_attn.qkv_proj.weight"]
                else:
                    qkv = torch.cat([
                        ref_sd[p + "self_attn.q_proj.weight"],
                        ref_sd[p + "self_attn.k_proj.weight"],
                        ref_sd[p + "self_attn.v_proj.weight"],
                    ],
                                    dim=0)
                layer.self_attn.qkv_proj.weight.copy_(qkv.to(device).to(dtype))
                layer.self_attn.o_proj.weight.copy_(
                    ref_sd[p + "self_attn.o_proj.weight"].to(device).to(dtype))
                if (p + "mlp.gate_up_proj.weight") in ref_sd:
                    gate_up = ref_sd[p + "mlp.gate_up_proj.weight"]
                else:
                    gate_up = torch.cat([
                        ref_sd[p + "mlp.gate_proj.weight"],
                        ref_sd[p + "mlp.up_proj.weight"],
                    ],
                                        dim=0)
                layer.mlp.gate_up_proj.weight.copy_(
                    gate_up.to(device).to(dtype))
                layer.mlp.down_proj.weight.copy_(
                    ref_sd[p + "mlp.down_proj.weight"].to(device).to(dtype))
                layer.input_layernorm.weight.copy_(
                    ref_sd[p + "input_layernorm.weight"].to(device).to(dtype))
                layer.post_attention_layernorm.weight.copy_(ref_sd[
                    p + "post_attention_layernorm.weight"].to(device).to(dtype))

        prefix_len = 6
        extra_len = 4
        full = torch.randint(0,
                             config.vocab_size, (prefix_len + extra_len, ),
                             dtype=torch.int32)
        prefix = full[:prefix_len].clone()

        prefix_kv, prefix_meta = _build_context_metadata(
            model_config, prefix_len)
        try:
            with torch.inference_mode():
                prefix_logits = model.forward(
                    input_ids=prefix.to(device),
                    position_ids=torch.arange(prefix_len,
                                              dtype=torch.int32,
                                              device=device).unsqueeze(0),
                    attn_metadata=prefix_meta,
                    return_context_logits=True,
                )
        finally:
            prefix_kv.shutdown()

        full_kv, full_meta = _build_context_metadata(model_config,
                                                     prefix_len + extra_len)
        try:
            with torch.inference_mode():
                full_logits = model.forward(
                    input_ids=full.to(device),
                    position_ids=torch.arange(prefix_len + extra_len,
                                              dtype=torch.int32,
                                              device=device).unsqueeze(0),
                    attn_metadata=full_meta,
                    return_context_logits=True,
                )
        finally:
            full_kv.shutdown()

        self.assertEqual(tuple(prefix_logits.shape),
                         (prefix_len, config.vocab_size))
        self.assertEqual(tuple(full_logits.shape),
                         (prefix_len + extra_len, config.vocab_size))
        self.assertTrue(torch.isfinite(prefix_logits).all())
        self.assertTrue(torch.isfinite(full_logits).all())
        self.assertGreater(prefix_logits.float().std().item(), 0.0)
        self.assertGreater(full_logits.float().std().item(), 0.0)

        with torch.inference_mode():
            ref_out = ref(input_ids=full.long().unsqueeze(0)).logits[0]

        # Parity against the independent transformers Phi3ForCausalLM with
        # equivalent weights. Identity/wrong-constant substitution in
        # attention, MLP, norm, or the LM head diverges from this reference.
        torch.testing.assert_close(full_logits.float().cpu(),
                                   ref_out.float(),
                                   rtol=6e-2,
                                   atol=6e-2)


if __name__ == "__main__":
    unittest.main()
