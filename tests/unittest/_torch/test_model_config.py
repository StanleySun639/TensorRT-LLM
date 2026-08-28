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

import errno
import json
import struct
import tempfile
import threading
import types

import filelock
import pytest
import torch

from tensorrt_llm._torch import model_config as model_config_module
from tensorrt_llm._torch.model_config import (
    _DEEPSEEK_V4_ROUTED_EXPERT_WEIGHT,
    ModelConfig,
    hf_remote_code_lock,
)
from tensorrt_llm._torch.pyexecutor.model_loader import (
    validate_and_set_kv_cache_quant,
    validate_encoder_decoder_kv_cache_config,
)
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.models.modeling_utils import QuantAlgo, QuantConfig

pytestmark = pytest.mark.cpu_only


def make_pretrained_config(
    *,
    num_attention_heads: int = 16,
    num_key_value_heads=8,
    head_dim: int | None = None,
    num_hidden_layers: int = 1,
    vocab_size: int = 3000,
    is_encoder_decoder: bool = False,
):
    # A minimal config object that provides the attributes used by
    # ModelConfig.get_bindings_model_config().
    hidden_size = head_dim * num_attention_heads
    intermediate_size = hidden_size * 4

    return types.SimpleNamespace(
        architectures=["DummyArchitecture"],
        num_attention_heads=num_attention_heads,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_key_value_heads=num_key_value_heads,
        head_dim=head_dim,
        num_hidden_layers=num_hidden_layers,
        vocab_size=vocab_size,
        torch_dtype=torch.float16,
        is_encoder_decoder=is_encoder_decoder,
    )


@pytest.mark.parametrize(
    "num_key_value_heads",
    [
        pytest.param(8, id="kv_heads_scalar"),
        pytest.param([8, 20], id="kv_heads_per_layer_varied"),
    ],
)
@pytest.mark.parametrize("enable_attention_dp", [False, True])
@pytest.mark.parametrize(
    "mapping_kwargs",
    [
        # Same tp/cp sizes, but different ways of setting attention TP:
        # - No explicit attn_tp_size: Mapping infers it.
        # - Explicit attn_tp_size: Mapping uses the provided value.
        dict(world_size=8, tp_size=4, cp_size=2),
        dict(world_size=4, tp_size=2, cp_size=2, attn_tp_size=4),
    ],
)
def test_get_bindings_model_config_attention_dp_attn_tp_override(
    enable_attention_dp, mapping_kwargs, num_key_value_heads
):
    mapping = Mapping(enable_attention_dp=enable_attention_dp, **mapping_kwargs)
    cfg = make_pretrained_config(
        # Keep values consistent:
        # hidden_size = num_attention_heads * head_dim.
        num_attention_heads=16,
        head_dim=4,
        num_key_value_heads=num_key_value_heads,
        num_hidden_layers=2,
    )
    model_config = ModelConfig(pretrained_config=cfg, mapping=mapping)

    tokens_per_block = 32
    bindings_cfg = model_config.get_bindings_model_config(tokens_per_block=tokens_per_block)

    # bindings hidden_size is sharded by attn_tp_size and attn_cp_size.
    attn_tp_size = mapping.attn_tp_size if not mapping.enable_attention_dp else 1
    attn_cp_size = mapping.attn_cp_size

    def ceil_div(a, b):
        return (a + b - 1) // b

    assert bindings_cfg.num_heads == ceil_div(cfg.num_attention_heads, attn_tp_size * attn_cp_size)
    # bindings hidden_size is sharded by attn_tp_size.
    assert bindings_cfg.hidden_size == ceil_div(cfg.hidden_size, attn_tp_size)
    if isinstance(cfg.num_key_value_heads, (list, tuple)):
        expected_num_kv_heads_per_layer = [
            ceil_div(kv, attn_tp_size * attn_cp_size) for kv in cfg.num_key_value_heads
        ]
        assert list(bindings_cfg.num_kv_heads_per_layer) == expected_num_kv_heads_per_layer
        assert bindings_cfg.num_kv_heads(0) == expected_num_kv_heads_per_layer[0]
    else:
        assert bindings_cfg.num_kv_heads(0) == ceil_div(
            cfg.num_key_value_heads, attn_tp_size * attn_cp_size
        )

    # tp_size-dependent value (uses mapping.tp_size, not attn_tp_size).
    assert bindings_cfg.mlp_hidden_size == ceil_div(cfg.intermediate_size, mapping.tp_size)
    assert bindings_cfg.tokens_per_block == tokens_per_block


def _make_model_config_with_kv_quant(kv_cache_quant_algo):
    return ModelConfig(quant_config=QuantConfig(kv_cache_quant_algo=kv_cache_quant_algo))


def _make_kv_cache_config(
    *, use_kv_cache_manager_v2: bool = False, cross_kv_cache_fraction: float | None = None
):
    return types.SimpleNamespace(
        use_kv_cache_manager_v2=use_kv_cache_manager_v2,
        cross_kv_cache_fraction=cross_kv_cache_fraction,
    )


def test_validate_and_set_kv_cache_quant_auto_uses_checkpoint():
    model_config = _make_model_config_with_kv_quant(QuantAlgo.FP8)
    validate_and_set_kv_cache_quant(model_config, "auto")
    assert model_config.quant_config.kv_cache_quant_algo == QuantAlgo.FP8


def test_validate_and_set_kv_cache_quant_explicit_dtype_overrides():
    model_config = _make_model_config_with_kv_quant(QuantAlgo.FP8)
    validate_and_set_kv_cache_quant(model_config, "nvfp4")
    assert model_config.quant_config.kv_cache_quant_algo == QuantAlgo.NVFP4


def test_validate_and_set_kv_cache_quant_rejects_invalid_dtype():
    model_config = _make_model_config_with_kv_quant(QuantAlgo.FP8)
    with pytest.raises(ValueError, match="Accepted types are"):
        validate_and_set_kv_cache_quant(model_config, "invalid_dtype")


def _make_mixed_precision_model_config():
    """MIXED_PRECISION checkpoint shape: global config plus per-layer entries
    whose kv_cache_quant_algo comes only from hf_quant_config.json (None here)."""
    return ModelConfig(
        quant_config=QuantConfig(quant_algo=QuantAlgo.MIXED_PRECISION, kv_cache_quant_algo=None),
        quant_config_dict={
            "model.layers.0.attention": QuantConfig(quant_algo=QuantAlgo.FP8),
            "model.layers.0.mixer.experts": QuantConfig(quant_algo=QuantAlgo.NVFP4),
        },
    )


def test_validate_and_set_kv_cache_quant_propagates_to_quant_config_dict():
    """Explicit kv_cache_config.dtype must override the per-layer QuantConfigs
    too, otherwise the KV pool (sized from the global config) and attention
    modules (built from per-layer configs) disagree on KV element size."""
    model_config = _make_mixed_precision_model_config()
    validate_and_set_kv_cache_quant(model_config, "fp8")
    assert model_config.quant_config.kv_cache_quant_algo == QuantAlgo.FP8
    for layer_quant_config in model_config.quant_config_dict.values():
        assert layer_quant_config.kv_cache_quant_algo == QuantAlgo.FP8


def test_validate_and_set_kv_cache_quant_auto_keeps_quant_config_dict():
    model_config = _make_mixed_precision_model_config()
    validate_and_set_kv_cache_quant(model_config, "auto")
    for layer_quant_config in model_config.quant_config_dict.values():
        assert layer_quant_config.kv_cache_quant_algo is None


def _write_safetensors_header(checkpoint_dir, tensor_dtype, tensor_shape):
    shard_name = "model-00001-of-00001.safetensors"
    header = {
        _DEEPSEEK_V4_ROUTED_EXPERT_WEIGHT: {
            "dtype": tensor_dtype,
            "shape": tensor_shape,
            "data_offsets": [0, 0],
        }
    }
    encoded_header = json.dumps(header).encode("utf-8")

    with open(checkpoint_dir / shard_name, "wb") as f:
        f.write(struct.pack("<Q", len(encoded_header)))
        f.write(encoded_header)

    with open(checkpoint_dir / "model.safetensors.index.json", "w") as f:
        json.dump({"weight_map": {_DEEPSEEK_V4_ROUTED_EXPERT_WEIGHT: shard_name}}, f)


@pytest.mark.parametrize(
    "tensor_dtype,tensor_shape,expected_layout,expected_is_base",
    [
        pytest.param("I8", [2048, 2048], "mxfp4", False, id="mxfp4"),
        pytest.param("U8", [2048, 2048], "nvfp4", False, id="nvfp4"),
        pytest.param("F8_E4M3", [2048, 4096], None, True, id="base-fp8"),
    ],
)
def test_deepseek_v4_base_checkpoint_detection(
    tmp_path, tensor_dtype, tensor_shape, expected_layout, expected_is_base
):
    _write_safetensors_header(tmp_path, tensor_dtype, tensor_shape)

    assert ModelConfig._detect_deepseek_v4_routed_moe_layout(str(tmp_path)) == expected_layout
    assert ModelConfig._is_deepseek_v4_base_checkpoint(str(tmp_path)) is expected_is_base


def test_deepseek_v4_missing_compress_ratios_raises(tmp_path, monkeypatch):
    """DeepSeek-V4 load must fail fast with a clear error when neither the
    checkpoint config nor a user ``sparse_attention_config`` provides
    ``compress_ratios``.

    Regression test: previously ``compress_ratios`` could stay ``None`` and the
    internal normalization comprehension raised an opaque
    ``TypeError: 'NoneType' object is not iterable`` mid-load.
    """
    from tensorrt_llm._torch import model_config as model_config_module
    from tensorrt_llm._torch.configs.deepseekv4 import DeepseekV4Config

    pretrained_config = DeepseekV4Config(
        architectures=["DeepseekV4ForCausalLM"],
        compress_ratios=None,
        num_hidden_layers=4,
    )

    # Avoid touching the filesystem for the HF config load; the empty tmp_path
    # makes the real ``_is_deepseek_v4_base_checkpoint`` probe return False.
    monkeypatch.setattr(
        model_config_module, "load_pretrained_config", lambda *args, **kwargs: pretrained_config
    )

    with pytest.raises(ValueError, match="compress_ratios"):
        ModelConfig.from_pretrained(str(tmp_path))


def test_model_config_sets_is_encoder_decoder_from_pretrained_config():
    model_config = ModelConfig(
        pretrained_config=make_pretrained_config(
            head_dim=4,
            is_encoder_decoder=True,
        )
    )

    assert model_config.is_encoder_decoder is True


def test_validate_encoder_decoder_kv_cache_config_accepts_v1_enc_dec():
    """V1 KVCacheManager is the default and production target for enc-dec models.

    Both V1 and V2 are supported as long as ``cross_kv_cache_fraction`` is set.
    """
    model_config = ModelConfig(
        pretrained_config=make_pretrained_config(
            head_dim=4,
            is_encoder_decoder=True,
        )
    )

    validate_encoder_decoder_kv_cache_config(
        model_config,
        _make_kv_cache_config(use_kv_cache_manager_v2=False, cross_kv_cache_fraction=0.5),
    )


def test_validate_encoder_decoder_kv_cache_config_requires_cross_fraction():
    model_config = ModelConfig(
        pretrained_config=make_pretrained_config(
            head_dim=4,
            is_encoder_decoder=True,
        )
    )

    with pytest.raises(ValueError, match="cross_kv_cache_fraction to be set"):
        validate_encoder_decoder_kv_cache_config(
            model_config,
            _make_kv_cache_config(use_kv_cache_manager_v2=True),
        )


def test_validate_encoder_decoder_kv_cache_config_rejects_cross_fraction_for_decoder_only():
    model_config = ModelConfig(
        pretrained_config=make_pretrained_config(
            head_dim=4,
            is_encoder_decoder=False,
        )
    )

    with pytest.raises(ValueError, match="should only be set for encoder-decoder models"):
        validate_encoder_decoder_kv_cache_config(
            model_config,
            _make_kv_cache_config(cross_kv_cache_fraction=0.5),
        )


def test_validate_encoder_decoder_kv_cache_config_accepts_v2_enc_dec():
    model_config = ModelConfig(
        pretrained_config=make_pretrained_config(
            head_dim=4,
            is_encoder_decoder=True,
        )
    )

    validate_encoder_decoder_kv_cache_config(
        model_config,
        _make_kv_cache_config(use_kv_cache_manager_v2=True, cross_kv_cache_fraction=0.5),
    )


def test_hf_remote_code_lock_filler_holds_lock_during_body(tmp_path, monkeypatch):
    monkeypatch.setattr(model_config_module, "HF_MODULES_CACHE", str(tmp_path))

    lock_file = str(tmp_path / "hf_remote_code.lock")

    # --- Filler path: first caller holds the lock during the body ---
    filler_body_count = []
    filler_lock_held = []

    with hf_remote_code_lock(timeout=5):
        filler_body_count.append(1)
        probe = filelock.FileLock(lock_file)
        try:
            probe.acquire(timeout=0)
            filler_lock_held.append(False)
            probe.release()
        except filelock.Timeout:
            filler_lock_held.append(True)

    assert len(filler_body_count) == 1
    assert filler_lock_held == [True]

    # After exit the lock must be released.
    check = filelock.FileLock(lock_file)
    check.acquire(timeout=0)
    check.release()

    # --- Waiter path: use a subclass that releases the holder only when the
    # bounded wait (timeout > 0) begins, ensuring the probe deterministically
    # observes contention. ---
    holder_ready = threading.Event()
    holder_release = threading.Event()
    real_file_lock = filelock.FileLock

    def _hold_lock():
        lk = real_file_lock(lock_file)
        lk.acquire(timeout=5)
        holder_ready.set()
        holder_release.wait(timeout=10)
        lk.release()

    t = threading.Thread(target=_hold_lock, daemon=True)
    t.start()
    assert holder_ready.wait(timeout=5)

    class _ReleaseHolderOnWait(real_file_lock):
        def acquire(self, timeout=-1, **kwargs):
            if timeout is not None and timeout > 0:
                holder_release.set()
            return super().acquire(timeout=timeout, **kwargs)

    monkeypatch.setattr(model_config_module.filelock, "FileLock", _ReleaseHolderOnWait)

    waiter_body_count = []
    waiter_lock_held = []

    with hf_remote_code_lock(timeout=5):
        waiter_body_count.append(1)
        probe2 = real_file_lock(lock_file)
        try:
            probe2.acquire(timeout=0)
            waiter_lock_held.append(False)
            probe2.release()
        except filelock.Timeout:
            waiter_lock_held.append(True)

    t.join(timeout=5)
    assert not t.is_alive()

    assert len(waiter_body_count) == 1
    assert waiter_lock_held == [False]


def test_hf_remote_code_lock_bounded_wait_timeout_degrades(tmp_path, monkeypatch):
    monkeypatch.setattr(model_config_module, "HF_MODULES_CACHE", str(tmp_path))

    acquire_calls = []
    release_calls = []

    class MockFileLock:
        def __init__(self, path, *args, **kwargs):
            self._path = path

        def acquire(self, timeout=-1, **kwargs):
            acquire_calls.append(("acquire", timeout))
            raise filelock.Timeout(self._path)

        def release(self, **kwargs):
            release_calls.append("release")

    monkeypatch.setattr(model_config_module.filelock, "FileLock", MockFileLock)

    body_executed = []

    with hf_remote_code_lock(timeout=1):
        body_executed.append(True)

    # The probe (timeout=0) raises filelock.Timeout -> _try_take_lock returns
    # False -> waiter branch -> bounded acquire(timeout=1) also raises
    # filelock.Timeout -> swallowed, body still executes, no release.
    assert len(body_executed) == 1
    assert release_calls == []
    assert len(acquire_calls) == 2
    assert acquire_calls[0] == ("acquire", 0)
    assert acquire_calls[1][0] == "acquire"
    assert acquire_calls[1][1] > 0


@pytest.mark.parametrize(
    "first_error",
    [
        pytest.param(PermissionError("cache dir denied"), id="PermissionError"),
        pytest.param(
            OSError(errno.EACCES, "access denied"),
            id="OSError-EACCES",
        ),
        pytest.param(
            OSError(errno.EPERM, "operation not permitted"),
            id="OSError-EPERM",
        ),
        pytest.param(
            OSError(errno.ENOLCK, "no locks available"),
            id="OSError-ENOLCK",
        ),
        pytest.param(
            OSError(errno.ESTALE, "stale NFS handle"),
            id="OSError-ESTALE",
        ),
        pytest.param(
            OSError(errno.EEXIST, "file exists"),
            id="OSError-EEXIST",
        ),
    ],
)
def test_hf_remote_code_lock_tempdir_fallback(tmp_path, monkeypatch, first_error):
    monkeypatch.setattr(model_config_module, "HF_MODULES_CACHE", str(tmp_path))

    call_count = [0]
    paths_used = []

    class MockFileLockInfraFail:
        def __init__(self, path, *args, **kwargs):
            self._path = path

        def acquire(self, timeout=-1, **kwargs):
            call_count[0] += 1
            paths_used.append(self._path)
            if call_count[0] == 1:
                raise first_error
            elif call_count[0] == 2:
                raise PermissionError("tempdir denied")
            return None

        def release(self, **kwargs):
            pass

    monkeypatch.setattr(model_config_module.filelock, "FileLock", MockFileLockInfraFail)

    body_executed = []

    with hf_remote_code_lock(timeout=1):
        body_executed.append(True)

    assert call_count[0] == 2
    assert len(body_executed) == 1
    assert paths_used[0] == str(tmp_path / "hf_remote_code.lock")
    assert paths_used[1].startswith(tempfile.gettempdir())

    # --- Non-infra OSError propagates ---
    call_count[0] = 0

    class MockFileLockNonInfra:
        def __init__(self, path, *args, **kwargs):
            self._path = path

        def acquire(self, timeout=-1, **kwargs):
            call_count[0] += 1
            err = OSError("non-infra error")
            err.errno = errno.ENOENT
            raise err

        def release(self, **kwargs):
            pass

    monkeypatch.setattr(model_config_module.filelock, "FileLock", MockFileLockNonInfra)

    with pytest.raises(OSError, match="non-infra error"):
        with hf_remote_code_lock(timeout=1):
            pass
