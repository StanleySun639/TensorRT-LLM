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

from unittest.mock import MagicMock

import torch

from tensorrt_llm._torch.distributed import AllReduceFusionOp, AllReduceParams


def _make_mock_layer(has_nvfp4: bool):
    """Create a mock DeepseekV3DecoderLayer with the relevant attributes for forward_mlp testing."""
    from tensorrt_llm._torch.models.modeling_deepseekv3 import DeepseekV3DecoderLayer

    layer = object.__new__(DeepseekV3DecoderLayer)

    # fusion_config with PRE_MLP_FUSION enabled, POST_MLP_FUSION disabled
    fusion_config = MagicMock()
    fusion_config.PRE_MLP_FUSION = True
    fusion_config.POST_MLP_FUSION = False
    layer.fusion_config = fusion_config

    # mlp mock
    mlp = MagicMock()
    gate_up_proj = MagicMock()
    gate_up_proj.has_nvfp4 = has_nvfp4
    gate_up_proj.input_scale = torch.tensor(1.0)
    mlp.gate_up_proj = gate_up_proj
    # mlp() returns a tensor
    mlp.return_value = torch.zeros(4, 16)
    layer.mlp = mlp

    # mlp_tp_size
    layer.mlp_tp_size = 1

    # post_attention_layernorm mock
    post_attention_layernorm = MagicMock()
    post_attention_layernorm.weight = torch.ones(16)
    post_attention_layernorm.variance_epsilon = 1e-6
    layer.post_attention_layernorm = post_attention_layernorm

    # next_layer_layernorm mock
    next_layer_layernorm = MagicMock()
    next_layer_layernorm.weight = torch.ones(16)
    next_layer_layernorm.variance_epsilon = 1e-6
    next_layer_layernorm.return_value = (torch.zeros(4, 16), torch.zeros(4, 16))
    layer.next_layer_layernorm = next_layer_layernorm

    # allreduce mock - we'll track calls to this
    layer.allreduce = MagicMock()

    return layer


def test_forward_mlp_pre_mlp_fusion_no_nvfp4_fallback():
    """Test that forward_mlp uses RESIDUAL_RMS_NORM when has_nvfp4=False
    and RESIDUAL_RMS_NORM_QUANT_NVFP4 when has_nvfp4=True under PRE_MLP_FUSION."""
    from tensorrt_llm._torch.models.modeling_deepseekv3 import DeepseekV3DecoderLayer

    # Test case 1: has_nvfp4 = False -> should use RESIDUAL_RMS_NORM
    layer_no_fp4 = _make_mock_layer(has_nvfp4=False)
    # allreduce returns (hidden_states, residual) for RESIDUAL_RMS_NORM path
    layer_no_fp4.allreduce.return_value = (torch.zeros(4, 16), torch.zeros(4, 16))

    hidden_states = torch.randn(4, 16)
    residual = torch.randn(4, 16)

    DeepseekV3DecoderLayer.forward_mlp(
        layer_no_fp4, hidden_states=hidden_states, residual=residual, spec_metadata=None
    )

    # Verify allreduce was called with RESIDUAL_RMS_NORM (not QUANT_NVFP4)
    assert layer_no_fp4.allreduce.call_count >= 1
    first_call_kwargs = layer_no_fp4.allreduce.call_args_list[0]
    all_reduce_params = first_call_kwargs[1]["all_reduce_params"]
    assert isinstance(all_reduce_params, AllReduceParams)
    assert all_reduce_params.fusion_op == AllReduceFusionOp.RESIDUAL_RMS_NORM

    # Test case 2: has_nvfp4 = True -> should use RESIDUAL_RMS_NORM_QUANT_NVFP4
    layer_fp4 = _make_mock_layer(has_nvfp4=True)
    # allreduce returns (act_fp4, act_sf, residual) for QUANT_NVFP4 path
    layer_fp4.allreduce.return_value = (torch.zeros(4, 16), torch.zeros(4, 4), torch.zeros(4, 16))

    hidden_states = torch.randn(4, 16)
    residual = torch.randn(4, 16)

    DeepseekV3DecoderLayer.forward_mlp(
        layer_fp4, hidden_states=hidden_states, residual=residual, spec_metadata=None
    )

    # Verify allreduce was called with RESIDUAL_RMS_NORM_QUANT_NVFP4
    assert layer_fp4.allreduce.call_count >= 1
    first_call_kwargs = layer_fp4.allreduce.call_args_list[0]
    all_reduce_params = first_call_kwargs[1]["all_reduce_params"]
    assert isinstance(all_reduce_params, AllReduceParams)
    assert all_reduce_params.fusion_op == AllReduceFusionOp.RESIDUAL_RMS_NORM_QUANT_NVFP4


if __name__ == "__main__":
    test_forward_mlp_pre_mlp_fusion_no_nvfp4_fallback()
