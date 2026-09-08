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

from unittest.mock import MagicMock, patch

import pytest
import torch
from torch import nn
from transformers import BartConfig

from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._torch.models.modeling_bart import (
    BartDecoder,
    BartEncoder,
    BartForConditionalGeneration,
    BartModel,
)
from tensorrt_llm._torch.modules.embedding import LMHead
from tensorrt_llm._torch.modules.logits_processor import LogitsProcessor
from tensorrt_llm.mapping import Mapping


def _make_tiny_bart_config():
    return BartConfig(
        vocab_size=128,
        d_model=64,
        encoder_layers=2,
        decoder_layers=2,
        encoder_attention_heads=2,
        decoder_attention_heads=2,
        encoder_ffn_dim=128,
        decoder_ffn_dim=128,
        max_position_embeddings=32,
        activation_function="gelu",
        torch_dtype=torch.float32,
        tie_word_embeddings=False,
        scale_embedding=False,
    )


def _make_model_config(hf_config):
    mapping = Mapping(world_size=1, rank=0, tp_size=1, pp_size=1)
    return ModelConfig(
        pretrained_config=hf_config,
        mapping=mapping,
    )


def _identity_forward(self, *args, **kwargs):
    hidden = kwargs.get("hidden_states", args[1] if len(args) > 1 else args[0])
    return hidden


def _cross_identity_forward(self, *args, **kwargs):
    hidden = kwargs.get("hidden_states", args[0])
    return hidden


def _logits_processor_forward(self, hidden_states, lm_head, attn_metadata, return_context_logits):
    return lm_head(hidden_states)


class TestBartForConditionalGeneration:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.hf_config = _make_tiny_bart_config()
        self.model_config = _make_model_config(self.hf_config)

    def test_construction_and_topology(self):
        with torch.device("cuda"):
            model = BartForConditionalGeneration(self.model_config)

        assert isinstance(model, nn.Module)
        assert isinstance(model.model, BartModel)
        assert isinstance(model.lm_head, LMHead)
        assert isinstance(model.logits_processor, LogitsProcessor)

        assert isinstance(model.model.encoder, BartEncoder)
        assert isinstance(model.model.decoder, BartDecoder)

        assert len(model.model.encoder.layers) == self.hf_config.encoder_layers
        assert len(model.model.decoder.layers) == self.hf_config.decoder_layers

    @patch(
        "tensorrt_llm._torch.models.modeling_bart.BartSelfAttention.forward",
        _identity_forward,
    )
    @patch(
        "tensorrt_llm._torch.models.modeling_bart.BartCrossAttention.forward",
        _cross_identity_forward,
    )
    @patch(
        "tensorrt_llm._torch.models.modeling_bart.LogitsProcessor.forward",
        _logits_processor_forward,
    )
    def test_forward_output_shape(self):
        device = torch.device("cuda")
        with torch.device(device):
            model = BartForConditionalGeneration(self.model_config)
        model.eval()

        seq_len = 4
        enc_len = 6

        input_ids = torch.randint(0, self.hf_config.vocab_size, (seq_len,), device=device)
        encoder_input_ids = torch.randint(0, self.hf_config.vocab_size, (enc_len,), device=device)
        position_ids = (torch.arange(seq_len, device=device) + 2).int()
        encoder_position_ids = (torch.arange(enc_len, device=device) + 2).int()

        attn_metadata = MagicMock()
        encoder_attn_metadata = MagicMock()
        cross_attn_metadata = MagicMock()

        with torch.no_grad():
            output = model(
                attn_metadata=attn_metadata,
                input_ids=input_ids,
                position_ids=position_ids,
                encoder_input_ids=encoder_input_ids,
                encoder_position_ids=encoder_position_ids,
                encoder_attn_metadata=encoder_attn_metadata,
                cross_attn_metadata=cross_attn_metadata,
            )

        assert isinstance(output, torch.Tensor)
        assert output.shape[-1] == self.hf_config.vocab_size
        assert output.shape[0] == seq_len
