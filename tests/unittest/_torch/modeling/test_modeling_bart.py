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

import pytest
import torch
from torch import nn
from transformers import BartConfig

pytestmark = pytest.mark.cpu_only


def _make_bart_config(
    vocab_size=128,
    d_model=64,
    encoder_layers=2,
    decoder_layers=2,
    encoder_attention_heads=4,
    decoder_attention_heads=4,
    encoder_ffn_dim=128,
    decoder_ffn_dim=128,
    max_position_embeddings=32,
    activation_function="gelu",
    tie_word_embeddings=True,
    scale_embedding=False,
    torch_dtype=torch.float32,
):
    return BartConfig(
        vocab_size=vocab_size,
        d_model=d_model,
        encoder_layers=encoder_layers,
        decoder_layers=decoder_layers,
        encoder_attention_heads=encoder_attention_heads,
        decoder_attention_heads=decoder_attention_heads,
        encoder_ffn_dim=encoder_ffn_dim,
        decoder_ffn_dim=decoder_ffn_dim,
        max_position_embeddings=max_position_embeddings,
        activation_function=activation_function,
        tie_word_embeddings=tie_word_embeddings,
        scale_embedding=scale_embedding,
        torch_dtype=torch_dtype,
    )


def test_bart_for_conditional_generation_construction_and_structure():
    from tensorrt_llm._torch.model_config import ModelConfig
    from tensorrt_llm._torch.models.modeling_bart import (
        BartDecoderLayer,
        BartEncoder,
        BartEncoderLayer,
        BartForConditionalGeneration,
        BartModel,
    )
    from tensorrt_llm._torch.modules.embedding import LMHead
    from tensorrt_llm._torch.modules.logits_processor import LogitsProcessor
    from tensorrt_llm.mapping import Mapping

    hf_config = _make_bart_config(
        encoder_layers=2,
        decoder_layers=3,
        max_position_embeddings=32,
        tie_word_embeddings=True,
        scale_embedding=False,
    )

    mapping = Mapping(world_size=1, rank=0, tp_size=1, pp_size=1)
    model_config = ModelConfig(
        pretrained_config=hf_config,
        mapping=mapping,
    )

    with torch.device("meta"):
        model = BartForConditionalGeneration(model_config)

    assert isinstance(model, nn.Module)

    assert isinstance(model.model, BartModel)
    assert isinstance(model.model.encoder, BartEncoder)

    assert len(model.model.encoder.layers) == 2
    for layer in model.model.encoder.layers:
        assert isinstance(layer, BartEncoderLayer)

    assert len(model.model.decoder.layers) == 3
    for layer in model.model.decoder.layers:
        assert isinstance(layer, BartDecoderLayer)

    assert isinstance(model.lm_head, LMHead)
    assert isinstance(model.logits_processor, LogitsProcessor)

    assert model.config is hf_config

    assert model.infer_max_seq_len() == 32

    assert model.lm_head.weight is model.model.shared_embedding.weight

    assert model.model.embed_scale == 1.0
