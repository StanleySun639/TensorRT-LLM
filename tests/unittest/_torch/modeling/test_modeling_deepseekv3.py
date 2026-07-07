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

"""Unit tests for DeepseekV3DecoderLayer.forward_mlp fusion op selection."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import torch

from tensorrt_llm._torch.distributed import AllReduceFusionOp
from tensorrt_llm._torch.utils import Fp4QuantizedTensor


def test_forward_mlp_pre_mlp_fusion_no_nvfp4_fallback():
    """Test that forward_mlp uses RESIDUAL_RMS_NORM when PRE_MLP_FUSION is
    enabled but gate_up_proj.has_nvfp4 is False, and does NOT wrap the result
    in Fp4QuantizedTensor."""
    from tensorrt_llm._torch.models.modeling_deepseekv3 import \
        DeepseekV3DecoderLayer

    layer = MagicMock()

    # fusion_config
    layer.fusion_config = SimpleNamespace(
        PRE_MLP_FUSION=True,
        POST_MLP_FUSION=False,
    )

    # mlp.gate_up_proj.has_nvfp4 = False (the key condition for the new branch)
    gate_up_proj_mock = MagicMock()
    gate_up_proj_mock.has_nvfp4 = False
    gate_up_proj_mock.input_scale = torch.tensor(1.0)

    # post_attention_layernorm
    layer.post_attention_layernorm = MagicMock()
    layer.post_attention_layernorm.weight = torch.ones(16)
    layer.post_attention_layernorm.variance_epsilon = 1e-5

    # next_layer_layernorm returns (hidden, residual)
    final_hidden = torch.randn(4, 16)
    final_residual = torch.randn(4, 16)
    layer.next_layer_layernorm = MagicMock(
        return_value=(final_hidden, final_residual))

    # mlp_tp_size
    layer.mlp_tp_size = 1

    # allreduce should return (hidden_states, residual) for RESIDUAL_RMS_NORM path
    mock_hidden = torch.randn(4, 16)
    mock_residual = torch.randn(4, 16)
    layer.allreduce = MagicMock(return_value=(mock_hidden, mock_residual))

    # mlp callable returns a tensor; also attach gate_up_proj attribute
    mlp_output = torch.randn(4, 16)
    mlp_mock = MagicMock(return_value=mlp_output)
    mlp_mock.gate_up_proj = gate_up_proj_mock
    layer.mlp = mlp_mock

    hidden_states = torch.randn(4, 16)
    residual = torch.randn(4, 16)

    # Call the unbound method with our mock as self.
    # NOTE: This pattern breaks if forward_mlp is ever decorated/wrapped.
    result = DeepseekV3DecoderLayer.forward_mlp(layer, hidden_states, residual,
                                                None)

    # --- Assertion 1: allreduce called with RESIDUAL_RMS_NORM ---
    layer.allreduce.assert_called_once()
    ar_call = layer.allreduce.call_args
    # The source always passes all_reduce_params as a keyword argument.
    all_reduce_params = ar_call.kwargs.get('all_reduce_params')

    assert all_reduce_params is not None, \
        "allreduce must be called with all_reduce_params keyword argument"
    assert all_reduce_params.fusion_op == AllReduceFusionOp.RESIDUAL_RMS_NORM, \
        (f"Expected RESIDUAL_RMS_NORM but got {all_reduce_params.fusion_op}; "
         "the fallback branch for unquantized dense layers must use "
         "RESIDUAL_RMS_NORM")

    # --- Assertion 2: self.mlp is called exactly once with a plain tensor ---
    assert layer.mlp.call_count == 1, \
        f"Expected self.mlp to be called exactly once, got {layer.mlp.call_count}"
    mlp_call = layer.mlp.call_args
    mlp_input = mlp_call.args[0] if mlp_call.args else mlp_call.kwargs.get(
        'hidden_states')
    assert not isinstance(mlp_input, Fp4QuantizedTensor), \
        "MLP input should NOT be Fp4QuantizedTensor when has_nvfp4 is False"
    assert isinstance(mlp_input, torch.Tensor), \
        f"MLP input should be a plain torch.Tensor, got {type(mlp_input)}"
    assert torch.equal(mlp_input, mock_hidden), \
        "MLP input should be the hidden_states returned by allreduce"

    # --- Assertion 3: final return value matches next_layer_layernorm output ---
    assert isinstance(result, tuple) and len(result) == 2, \
        f"forward_mlp should return a 2-tuple, got {type(result)}"
    assert torch.equal(result[0], final_hidden), \
        "First element of return should be hidden_states from next_layer_layernorm"
    assert torch.equal(result[1], final_residual), \
        "Second element of return should be residual from next_layer_layernorm"
