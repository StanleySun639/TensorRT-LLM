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
"""Unit tests for DeepseekV3DecoderLayer.forward_mlp fusion-op selection."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import torch

from tensorrt_llm._torch.distributed import AllReduceFusionOp, AllReduceParams


def _make_mock_layer(has_nvfp4: bool):
    """Create a minimal mock of DeepseekV3DecoderLayer for forward_mlp testing."""
    layer = MagicMock()

    # fusion_config with PRE_MLP_FUSION enabled, POST_MLP_FUSION disabled
    layer.fusion_config = SimpleNamespace(
        PRE_MLP_FUSION=True,
        POST_MLP_FUSION=False,
    )

    # mlp.gate_up_proj.has_nvfp4 controls the branch
    layer.mlp = MagicMock()
    layer.mlp.gate_up_proj = MagicMock()
    layer.mlp.gate_up_proj.has_nvfp4 = has_nvfp4
    layer.mlp.gate_up_proj.input_scale = torch.tensor(1.0)

    # post_attention_layernorm
    layer.post_attention_layernorm = MagicMock()
    layer.post_attention_layernorm.weight = torch.ones(16)
    layer.post_attention_layernorm.variance_epsilon = 1e-6

    # next_layer_layernorm
    layer.next_layer_layernorm = MagicMock()
    layer.next_layer_layernorm.weight = torch.ones(16)
    layer.next_layer_layernorm.variance_epsilon = 1e-6

    # mlp_tp_size
    layer.mlp_tp_size = 1

    # layer_idx
    layer.layer_idx = 0

    return layer


def test_forward_mlp_pre_mlp_fusion_fallback_without_nvfp4():
    """When PRE_MLP_FUSION is enabled and gate_up_proj.has_nvfp4 is False,
    allreduce must use RESIDUAL_RMS_NORM (not RESIDUAL_RMS_NORM_QUANT_NVFP4).
    When has_nvfp4 is True, it must use RESIDUAL_RMS_NORM_QUANT_NVFP4."""
    from tensorrt_llm._torch.models.modeling_deepseekv3 import DeepseekV3DecoderLayer

    # --- Test case 1: has_nvfp4 = False => RESIDUAL_RMS_NORM ---
    layer = _make_mock_layer(has_nvfp4=False)

    hidden_states = torch.randn(4, 16)
    residual = torch.randn(4, 16)

    # allreduce returns (hidden_states, residual) for the non-nvfp4 path
    layer.allreduce = MagicMock(return_value=(hidden_states.clone(), residual.clone()))

    # mlp returns a tensor
    layer.mlp.return_value = hidden_states.clone()

    # next_layer_layernorm returns (hidden, residual)
    layer.next_layer_layernorm.return_value = (hidden_states.clone(), residual.clone())

    # Call the unbound method with our mock as self
    DeepseekV3DecoderLayer.forward_mlp(
        layer, hidden_states=hidden_states, residual=residual, spec_metadata=None
    )

    # Verify allreduce was called
    assert layer.allreduce.called, "allreduce should be called when PRE_MLP_FUSION is enabled"

    # Get the AllReduceParams from the first allreduce call
    first_call_kwargs = layer.allreduce.call_args_list[0]
    all_reduce_params = first_call_kwargs[1]["all_reduce_params"]

    assert isinstance(all_reduce_params, AllReduceParams)
    assert all_reduce_params.fusion_op == AllReduceFusionOp.RESIDUAL_RMS_NORM, (
        f"Expected RESIDUAL_RMS_NORM when has_nvfp4=False, got {all_reduce_params.fusion_op}"
    )

    # --- Test case 2: has_nvfp4 = True => RESIDUAL_RMS_NORM_QUANT_NVFP4 ---
    layer2 = _make_mock_layer(has_nvfp4=True)

    # allreduce returns (act_fp4, act_sf, residual) for the nvfp4 path
    act_fp4 = torch.randn(4, 16)
    act_sf = torch.randn(4, 1)
    layer2.allreduce = MagicMock(return_value=(act_fp4, act_sf, residual.clone()))

    # mlp returns a tensor
    layer2.mlp.return_value = hidden_states.clone()

    # next_layer_layernorm returns (hidden, residual)
    layer2.next_layer_layernorm.return_value = (hidden_states.clone(), residual.clone())

    DeepseekV3DecoderLayer.forward_mlp(
        layer2, hidden_states=hidden_states, residual=residual, spec_metadata=None
    )

    # Verify allreduce was called
    assert layer2.allreduce.called, "allreduce should be called when PRE_MLP_FUSION is enabled"

    # Get the AllReduceParams from the first allreduce call
    first_call_kwargs2 = layer2.allreduce.call_args_list[0]
    all_reduce_params2 = first_call_kwargs2[1]["all_reduce_params"]

    assert isinstance(all_reduce_params2, AllReduceParams)
    assert all_reduce_params2.fusion_op == AllReduceFusionOp.RESIDUAL_RMS_NORM_QUANT_NVFP4, (
        f"Expected RESIDUAL_RMS_NORM_QUANT_NVFP4 when has_nvfp4=True, "
        f"got {all_reduce_params2.fusion_op}"
    )
