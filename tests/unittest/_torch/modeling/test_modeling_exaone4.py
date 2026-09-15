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

import unittest
from copy import deepcopy

import torch
from transformers import Exaone4Config

from tensorrt_llm._torch.attention_backend.utils import get_attention_backend
from tensorrt_llm._torch.metadata import KVCacheParams
from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._torch.models.modeling_exaone4 import Exaone4DecoderLayer, Exaone4ForCausalLM
from tensorrt_llm._torch.models.modeling_utils import MODEL_CLASS_MAPPING
from tensorrt_llm._torch.pyexecutor.resource_manager import KVCacheManager
from tensorrt_llm.bindings import DataType
from tensorrt_llm.bindings.executor import KvCacheConfig
from tensorrt_llm.bindings.internal.batch_manager import CacheType
from tensorrt_llm.mapping import Mapping

# Tiny random-init config so the registered architecture fits on a single GPU
# without any weight download while still building every decoder layer.
EXAONE4_TEST_CONFIG = {
    "architectures": ["Exaone4ForCausalLM"],
    "attention_dropout": 0.0,
    "bos_token_id": 1,
    "dtype": "bfloat16",
    "eos_token_id": 2,
    "hidden_act": "silu",
    "hidden_size": 64,
    "initializer_range": 0.02,
    "intermediate_size": 128,
    "max_position_embeddings": 512,
    "model_type": "exaone4",
    "num_attention_heads": 4,
    "num_hidden_layers": 2,
    "num_key_value_heads": 2,
    "rms_norm_eps": 1e-05,
    "rope_theta": 10000.0,
    "sliding_window": None,
    "sliding_window_pattern": 4,
    "torch_dtype": "bfloat16",
    "use_cache": True,
    "vocab_size": 256,
}

_TOKENS_PER_BLOCK = 16
_NUM_BLOCKS = 8


@unittest.skipIf(not torch.cuda.is_available(), "needs a CUDA device")
class TestExaone4ForCausalLM(unittest.TestCase):
    def _build_model_config(self):
        hf_config = Exaone4Config(**deepcopy(EXAONE4_TEST_CONFIG))
        return ModelConfig(
            pretrained_config=hf_config,
            attn_backend="TRTLLM",
            mapping=Mapping(world_size=1, tp_size=1, rank=0),
        )

    def _build_kv_cache_manager(self, config):
        head_dim = config.hidden_size // config.num_attention_heads
        return KVCacheManager(
            kv_cache_config=KvCacheConfig(max_tokens=_NUM_BLOCKS * _TOKENS_PER_BLOCK),
            kv_cache_type=CacheType.SELF,
            num_layers=config.num_hidden_layers,
            num_kv_heads=config.num_key_value_heads,
            head_dim=head_dim,
            tokens_per_block=_TOKENS_PER_BLOCK,
            max_seq_len=EXAONE4_TEST_CONFIG["max_position_embeddings"],
            max_batch_size=1,
            mapping=Mapping(world_size=1, tp_size=1, rank=0),
            dtype=DataType.BF16,
        )

    def _make_attn_metadata(self, kv_cache_manager, num_tokens):
        request_ids = [0]
        token_nums = [num_tokens]
        kv_cache_manager.add_dummy_requests(request_ids, token_nums)
        metadata_cls = get_attention_backend("TRTLLM").Metadata
        attn_metadata = metadata_cls(
            seq_lens=torch.tensor([num_tokens], dtype=torch.int),
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
        return attn_metadata

    def _run_forward(self, model, model_config, input_ids, position_ids):
        num_tokens = input_ids.size(0)
        kv_cache_manager = self._build_kv_cache_manager(model_config.pretrained_config)
        try:
            attn_metadata = self._make_attn_metadata(kv_cache_manager, num_tokens)
            with torch.inference_mode():
                # Call the constructed top-level Exaone4ForCausalLM forward
                # directly on the instance so the registered architecture's
                # own tensor path executes end to end.
                return model.forward(
                    input_ids=input_ids,
                    position_ids=position_ids,
                    attn_metadata=attn_metadata,
                )
        finally:
            kv_cache_manager.shutdown()

    def _decoder_bypassed_logits(self, model, input_ids):
        # Reproduce the top-level forward while skipping every decoder layer:
        # use the model's own embedding, its own final norm, and its own
        # logits path over the raw embeddings. If the real decoder stack were
        # an identity/no-op, the full forward would equal this baseline.
        with torch.inference_mode():
            inner = model.model
            embeds = inner.embed_tokens(input_ids).to(inner.dtype)
            hidden_states = inner.norm(embeds)
            return model.logits_processor.forward(
                hidden_states,
                model.lm_head,
                attn_metadata=None,
            )

    def test_construction_and_forward(self):
        device = torch.device("cuda")

        # Resolve the architecture through the production registry rather than
        # the direct import, so dropping @register_auto_model would fail here.
        registered_cls = MODEL_CLASS_MAPPING["Exaone4ForCausalLM"]
        self.assertIs(registered_cls, Exaone4ForCausalLM)

        model_config = self._build_model_config()
        model = registered_cls(model_config)

        self.assertIsInstance(model, Exaone4ForCausalLM)
        self.assertEqual(model.config.vocab_size, EXAONE4_TEST_CONFIG["vocab_size"])
        self.assertEqual(model.model.embed_tokens.embedding_dim, EXAONE4_TEST_CONFIG["hidden_size"])
        self.assertEqual(len(model.model.layers), EXAONE4_TEST_CONFIG["num_hidden_layers"])
        for layer in model.model.layers:
            self.assertIsInstance(layer, Exaone4DecoderLayer)

        model = model.to(device).eval()

        input_ids = torch.tensor([3, 7, 11, 19, 23], dtype=torch.int32, device=device)
        num_tokens = input_ids.size(0)
        position_ids = torch.arange(num_tokens, dtype=torch.int32, device=device).unsqueeze(0)

        logits = self._run_forward(model, model_config, input_ids, position_ids)

        self.assertEqual(
            tuple(logits.shape),
            (num_tokens, EXAONE4_TEST_CONFIG["vocab_size"]),
        )
        self.assertEqual(logits.dtype, torch.float32)
        self.assertTrue(torch.isfinite(logits).all())

        # Decoder-bypassed baseline: same embeddings + same final norm + same
        # logits path, but no Exaone4DecoderLayer executed. Nothing is mocked;
        # a load-bearing attention/MLP decoder stack must move the real logits
        # away from this baseline.
        baseline_logits = self._decoder_bypassed_logits(model, input_ids)
        self.assertEqual(tuple(baseline_logits.shape), tuple(logits.shape))
        self.assertTrue(torch.isfinite(baseline_logits).all())
        self.assertFalse(torch.allclose(logits, baseline_logits, atol=1e-2))


if __name__ == "__main__":
    unittest.main()
