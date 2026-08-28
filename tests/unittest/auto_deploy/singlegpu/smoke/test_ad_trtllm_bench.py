# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import subprocess
import tempfile
from pathlib import Path

import pytest
import yaml
from _model_test_utils import get_small_model_config
from click.testing import CliRunner

from tensorrt_llm.bench.tuning.dataclasses import ModelConfig
from tensorrt_llm.bench.tuning.heuristics import (
    BYTES_PER_ELEM,
    calc_engine_setting,
    finetune_setting,
)
from tensorrt_llm.commands.bench import main
from tensorrt_llm.llmapi.llm_utils import QuantConfig

_PIECEWISE_COMPILE_BACKENDS = {"torch-cudagraph", "torch-opt"}


def run_benchmark(
    model_name: str, model_path: str, dataset_path: str, extra_llm_api_options_path: str
):
    runner = CliRunner()

    args = [
        "--model",
        model_name,
    ]

    # Only pass --model_path if it's a local filesystem path
    if model_path.startswith("/"):
        args.extend(["--model_path", model_path])

    args.extend(
        [
            "throughput",
            "--backend",
            "_autodeploy",
            "--dataset",
            dataset_path,
            "--iteration_log",
            "iteration_log.log",
            "--extra_llm_api_options",
            f"{extra_llm_api_options_path}",
        ]
    )
    result = runner.invoke(main, args, catch_exceptions=False)
    assert result.exit_code == 0

    with open("iteration_log.log", "r") as f:
        lines = f.readlines()
    assert len(lines) > 0
    # TODO: add more checks


def prepare_dataset(root_dir: str, temp_dir: str, model_path_or_name: str):
    _DATASET_NAME = "synthetic_128_128.txt"
    dataset_path = Path(temp_dir, _DATASET_NAME)

    # Generate a small dataset to run a test - matching workload configuration
    command = [
        "trtllm-bench",
        "--model",
        model_path_or_name,
        "prepare-dataset",
        "--output",
        f"{dataset_path}",
        "token-norm-dist",
        "--input-mean",
        "128",
        "--output-mean",
        "128",
        "--input-stdev",
        "0",
        "--output-stdev",
        "0",
        "--num-requests",
        "10",
    ]
    print(f"Running command: {' '.join(command)}")
    result = subprocess.run(command, cwd=str(temp_dir), capture_output=True, text=True, timeout=300)
    if result.returncode != 0:
        raise RuntimeError(f"Failed to prepare dataset: {result.stderr}")

    return dataset_path


@pytest.mark.parametrize("compile_backend", ["torch-compile", "torch-opt", "torch-cudagraph"])
@pytest.mark.parametrize("model_name", ["TinyLlama/TinyLlama-1.1B-Chat-v1.0"])
def test_trtllm_bench(llm_root, compile_backend, model_name):  # noqa: F811
    args = get_small_model_config(model_name)["args"]
    # remove kv_cache_config and max_batch_size to avoid conflicts with trtllm-bench
    args.pop("kv_cache_config", None)
    args.pop("max_batch_size", None)
    compile_model_config = {
        "stage": "compile",
        "cuda_graph_batch_sizes": [1, 2, 4, 8, 16, 32, 64, 128],
        "backend": compile_backend,
    }
    if compile_backend not in _PIECEWISE_COMPILE_BACKENDS:
        compile_model_config["piecewise_enabled"] = False

    with tempfile.TemporaryDirectory() as temp_dir:
        extra_llm_api_options_path = f"{temp_dir}/extra_llm_api_options.yaml"
        with open(extra_llm_api_options_path, "w") as f:
            yaml.dump(
                {
                    **args,
                    "transforms": {
                        "resize_kv_cache": {"enabled": False},  # rely on default estimation
                        "compile_model": compile_model_config,
                    },
                },
                f,
            )

        dataset_path = prepare_dataset(llm_root, temp_dir, args["model"])
        run_benchmark(model_name, str(args["model"]), dataset_path, extra_llm_api_options_path)


@pytest.mark.cpu_only
def test_mla_branch_yields_larger_batch_and_tokens(monkeypatch):
    device_memory_gb = 25.0
    monkeypatch.setattr(
        "tensorrt_llm.bench.tuning.heuristics.get_device_memory",
        lambda: device_memory_gb,
    )

    kv_lora_rank = 512
    qk_rope_head_dim = 64
    num_attention_layers = 60
    num_key_value_heads = 64
    head_size = 128

    mla_config = ModelConfig(
        name="deepseek-test",
        model_type="deepseek_v3",
        num_hidden_layers=num_attention_layers,
        num_attention_layers=num_attention_layers,
        num_attention_heads=64,
        num_key_value_heads=num_key_value_heads,
        head_size=head_size,
        hidden_size=7168,
        vocab_size=102400,
        param_count=2_000_000_000,
        max_position_embeddings=4096,
        kv_lora_rank=kv_lora_rank,
        qk_rope_head_dim=qk_rope_head_dim,
    )
    non_mla_config = ModelConfig(
        name="deepseek-test-nomla",
        model_type="llama",
        num_hidden_layers=num_attention_layers,
        num_attention_layers=num_attention_layers,
        num_attention_heads=64,
        num_key_value_heads=num_key_value_heads,
        head_size=head_size,
        hidden_size=7168,
        vocab_size=102400,
        param_count=2_000_000_000,
        max_position_embeddings=4096,
    )

    assert mla_config.is_mla() is True
    assert non_mla_config.is_mla() is False

    quant_config = QuantConfig()
    tp_size = 1
    pp_size = 1
    target_input_len = 511
    target_output_len = 1
    kv_cache_gpu_mem_fraction = 0.95

    mla_bs, mla_tokens = calc_engine_setting(
        model_config=mla_config,
        quant_config=quant_config,
        tp_size=tp_size,
        pp_size=pp_size,
        target_input_len=target_input_len,
        target_output_len=target_output_len,
        kv_cache_gpu_mem_fraction=kv_cache_gpu_mem_fraction,
    )
    non_mla_bs, non_mla_tokens = calc_engine_setting(
        model_config=non_mla_config,
        quant_config=quant_config,
        tp_size=tp_size,
        pp_size=pp_size,
        target_input_len=target_input_len,
        target_output_len=target_output_len,
        kv_cache_gpu_mem_fraction=kv_cache_gpu_mem_fraction,
    )

    byte_per_elem = BYTES_PER_ELEM.get(quant_config.quant_algo, 2)
    byte_per_kv_elem = BYTES_PER_ELEM.get(quant_config.kv_cache_quant_algo, 2)
    target_seq_len = target_input_len + target_output_len
    engine_size = mla_config.param_count * byte_per_elem / (1024**3)
    cache_memory = (device_memory_gb - engine_size) * mla_config.cache_memory_fraction(
        kv_cache_gpu_mem_fraction
    )

    expected_mla_gb_per_token = (
        num_attention_layers * (kv_lora_rank + qk_rope_head_dim) * byte_per_kv_elem / (1024**3)
    )
    expected_mla_requests = cache_memory / (expected_mla_gb_per_token * target_seq_len)
    expected_mla_setting = finetune_setting(
        expected_mla_requests,
        target_input_len,
        target_output_len,
        pp_size,
    )

    adjusted_num_kv_heads = max(tp_size, num_key_value_heads)
    expected_non_mla_gb_per_token = (
        2 * num_attention_layers * adjusted_num_kv_heads * head_size * byte_per_kv_elem / (1024**3)
    )
    expected_non_mla_requests = cache_memory / (expected_non_mla_gb_per_token * target_seq_len)
    expected_non_mla_setting = finetune_setting(
        expected_non_mla_requests,
        target_input_len,
        target_output_len,
        pp_size,
    )

    assert (mla_bs, mla_tokens) == expected_mla_setting
    assert expected_mla_setting == (640, 32768)
    assert (non_mla_bs, non_mla_tokens) == expected_non_mla_setting
    assert expected_non_mla_setting == (64, 11264)
    assert mla_bs > non_mla_bs
    assert mla_tokens > non_mla_tokens
