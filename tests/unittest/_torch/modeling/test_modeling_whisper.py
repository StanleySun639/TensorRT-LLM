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

"""Unit tests for modeling_whisper.py.

CPU-only parity tests pinning ``WhisperLogMelFrontend`` (the engine-side GPU
log-mel front-end) to the HF reference (``_torch_extract_fbank_features``);
drift here silently corrupts transcripts.
"""

import numpy as np
import pytest
import torch
from transformers import WhisperConfig, WhisperFeatureExtractor

from tensorrt_llm._torch.attention.backends.utils import get_attention_backend
from tensorrt_llm._torch.metadata import KVCacheParams
from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._torch.models.modeling_utils import get_model_architecture
from tensorrt_llm._torch.models.modeling_whisper import (
    WhisperForConditionalGeneration,
    WhisperLogMelFrontend,
)


def _synthetic_waveform_batch(n_samples: int, seed: int = 1234) -> np.ndarray:
    """[3, n_samples] fp32 batch: full window, ~1/3 window, and near-silence,
    zero-padded to a common length (the request contract)."""
    rng = np.random.default_rng(seed)
    time = np.arange(n_samples, dtype=np.float32)
    batch = np.zeros((3, n_samples), dtype=np.float32)

    full = 0.4 * np.sin(2 * np.pi * 440.0 / 16000.0 * time)
    full += 0.2 * np.sin(2 * np.pi * 1333.0 / 16000.0 * time)
    full += 0.05 * rng.standard_normal(n_samples)
    batch[0] = full.astype(np.float32)

    short = n_samples // 3
    batch[1, :short] = (0.3 * rng.standard_normal(short)).astype(np.float32)

    batch[2, :1600] = 1e-4  # hard-zero tail exercises the log floor
    return batch


def _reference_log_mel(extractor: WhisperFeatureExtractor, batch: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(extractor._torch_extract_fbank_features(batch, device="cpu"))


@pytest.mark.parametrize("num_mel_bins", [80, 128])
def test_log_mel_frontend_matches_hf(num_mel_bins):
    """Default-parameter parity (no preprocessor config: whisper-tiny=80,
    large-v3=128 mel bins)."""
    config = WhisperConfig(num_mel_bins=num_mel_bins)
    frontend = WhisperLogMelFrontend(config)
    extractor = WhisperFeatureExtractor(feature_size=num_mel_bins)

    batch = _synthetic_waveform_batch(extractor.n_samples)
    ours = frontend(torch.from_numpy(batch))
    reference = _reference_log_mel(extractor, batch)

    assert ours.shape == (3, num_mel_bins, extractor.nb_max_frames)
    torch.testing.assert_close(ours, reference, atol=1e-4, rtol=1e-4)


def test_log_mel_frontend_reads_preprocessor_config(tmp_path):
    """Non-default STFT parameters (hop_length) and dither must be read from
    the checkpoint's preprocessor_config.json, not assumed."""
    extractor = WhisperFeatureExtractor(feature_size=80, hop_length=320, dither=0.02)
    extractor.save_pretrained(tmp_path)

    config = WhisperConfig(num_mel_bins=80)
    config._name_or_path = str(tmp_path)
    frontend = WhisperLogMelFrontend(config)

    assert frontend.hop_length == 320
    assert frontend.dither == pytest.approx(0.02)

    # Both implementations draw the dither noise from torch's default
    # generator with an identical shape, so seeding both sides identically
    # makes the (random) dither path exactly comparable.
    batch = _synthetic_waveform_batch(extractor.n_samples)
    torch.manual_seed(0)
    ours = frontend(torch.from_numpy(batch))
    torch.manual_seed(0)
    reference = _reference_log_mel(extractor, batch)

    torch.testing.assert_close(ours, reference, atol=1e-4, rtol=1e-4)


def test_log_mel_frontend_feature_size_mismatch_falls_back(tmp_path):
    """A preprocessor config contradicting config.num_mel_bins (broken
    checkpoint) must not win over the model config: the conv stem's input
    channel count comes from num_mel_bins."""
    WhisperFeatureExtractor(feature_size=80).save_pretrained(tmp_path)

    config = WhisperConfig(num_mel_bins=128)
    config._name_or_path = str(tmp_path)
    frontend = WhisperLogMelFrontend(config)

    assert frontend._mel_filters_np.shape[1] == 128
    batch = _synthetic_waveform_batch(WhisperFeatureExtractor().n_samples)
    assert frontend(torch.from_numpy(batch)).shape[1] == 128


def test_log_mel_frontend_does_not_mutate_input():
    """The waveform buffer belongs to the request; dither must not be added
    in place."""
    extractor = WhisperFeatureExtractor(feature_size=80)
    config = WhisperConfig(num_mel_bins=80)
    frontend = WhisperLogMelFrontend(config)
    frontend.dither = 0.02

    batch = torch.from_numpy(_synthetic_waveform_batch(extractor.n_samples))
    snapshot = batch.clone()
    frontend(batch)
    torch.testing.assert_close(batch, snapshot, atol=0.0, rtol=0.0)


def _tiny_whisper_model_config() -> "ModelConfig":
    config = WhisperConfig(
        vocab_size=64,
        num_mel_bins=80,
        d_model=32,
        encoder_layers=1,
        encoder_attention_heads=2,
        decoder_layers=1,
        decoder_attention_heads=2,
        encoder_ffn_dim=64,
        decoder_ffn_dim=64,
        max_source_positions=16,
        max_target_positions=16,
        torch_dtype=torch.float16,
        architectures=["WhisperForConditionalGeneration"],
    )
    return ModelConfig(
        pretrained_config=config,
        attn_backend="TRTLLM",
        is_generation=True,
        is_encoder_decoder=True,
    )


def _decoder_forward_logits(
    model: WhisperForConditionalGeneration,
    token_ids: list[int],
    encoder_hidden_states: torch.Tensor,
) -> torch.Tensor:
    device = torch.device("cuda")
    seq_len = len(token_ids)
    encoder_len = encoder_hidden_states.shape[0]

    metadata_cls = get_attention_backend(model.model_config.attn_backend).Metadata
    attn_metadata = metadata_cls(
        max_num_requests=1,
        max_num_tokens=seq_len,
        max_num_sequences=1,
        kv_cache_manager=None,
        seq_lens=torch.tensor([seq_len], dtype=torch.int32),
        seq_lens_kv=torch.tensor([seq_len], dtype=torch.int32),
        num_contexts=1,
        kv_cache_params=KVCacheParams(use_cache=False),
        max_seq_len=seq_len,
    )
    attn_metadata.prepare()

    cross_attn_metadata = attn_metadata.create_cross_metadata([encoder_len])
    cross_attn_metadata.max_seq_len = encoder_len
    cross_attn_metadata.prepare()

    input_ids = torch.tensor(token_ids, dtype=torch.int32, device=device)
    position_ids = torch.arange(seq_len, dtype=torch.int32, device=device).unsqueeze(0)

    return model.forward(
        attn_metadata=attn_metadata,
        input_ids=input_ids,
        position_ids=position_ids,
        encoder_hidden_states=encoder_hidden_states,
        cross_attn_metadata=cross_attn_metadata,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_whisper_for_conditional_generation_construction_and_forward():
    model_config = _tiny_whisper_model_config()

    resolved_cls, _ = get_model_architecture(model_config.pretrained_config)
    assert resolved_cls is WhisperForConditionalGeneration

    torch.manual_seed(0)
    model = WhisperForConditionalGeneration(model_config).to("cuda").eval()

    assert model.model.decoder is not None
    assert model.model.encoder is not None
    assert model.lm_head.weight is model.model.decoder.embed_tokens.weight

    config = model_config.pretrained_config

    def _encoder_context(seed: int) -> torch.Tensor:
        gen = torch.Generator(device="cuda").manual_seed(seed)
        return torch.randn(
            config.max_source_positions,
            config.d_model,
            dtype=config.torch_dtype,
            device="cuda",
            generator=gen,
        )

    encoder_context = _encoder_context(seed=11)
    encoder_context_other = _encoder_context(seed=22)

    tokens_a = [1, 2, 3]
    tokens_b = [5, 6, 7]

    with torch.inference_mode():
        logits_a = _decoder_forward_logits(model, tokens_a, encoder_context)
        logits_b = _decoder_forward_logits(model, tokens_b, encoder_context)
        logits_a_other_ctx = _decoder_forward_logits(model, tokens_a, encoder_context_other)

    assert logits_a.shape[-1] == config.vocab_size
    assert logits_a.dtype == torch.float32
    assert torch.isfinite(logits_a).all()
    assert torch.isfinite(logits_b).all()
    assert torch.isfinite(logits_a_other_ctx).all()
    assert not torch.allclose(logits_a, logits_b)
    assert not torch.allclose(logits_a, logits_a_other_ctx)
