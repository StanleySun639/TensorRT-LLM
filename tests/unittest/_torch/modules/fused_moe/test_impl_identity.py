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

from tensorrt_llm._torch.modules.fused_moe.impl_contract import MoEEligibility, MoERejectReason
from tensorrt_llm._torch.modules.fused_moe.impl_identity import (
    MoEImplDescriptor,
    MoEImplId,
    MoEImplRegistry,
)
from tensorrt_llm._torch.modules.fused_moe.interface import MoESchedulerKind

pytestmark = pytest.mark.cpu_only


def test_moe_impl_id_parse_and_validate():
    # Valid construction and canonical round-trip
    impl_id = MoEImplId(
        provider="trtllm_native",
        technique="cutlass",
        quant="w4a8_mxfp4_mxfp8",
        kernel_name="blockscale_splitk",
    )
    assert impl_id.canonical() == "trtllm_native.cutlass.w4a8_mxfp4_mxfp8.blockscale_splitk"
    assert str(impl_id) == impl_id.canonical()

    # Uppercase rejected
    with pytest.raises(ValueError):
        MoEImplId(provider="TrtLLM", technique="cutlass", quant="bf16", kernel_name="dense")

    # Empty string rejected
    with pytest.raises(ValueError):
        MoEImplId(provider="", technique="cutlass", quant="bf16", kernel_name="dense")

    # Leading underscore rejected
    with pytest.raises(ValueError):
        MoEImplId(provider="_leading", technique="cutlass", quant="bf16", kernel_name="dense")

    # Dot in field rejected
    with pytest.raises(ValueError):
        MoEImplId(provider="has.dot", technique="cutlass", quant="bf16", kernel_name="dense")

    # MoEImplId.parse round-trip for valid input
    canonical_str = "deepgemm.cutedsl.nvfp4.densegemm"
    parsed = MoEImplId.parse(canonical_str)
    assert parsed.canonical() == canonical_str
    assert parsed == MoEImplId(
        provider="deepgemm", technique="cutedsl", quant="nvfp4", kernel_name="densegemm"
    )

    # MoEImplId.parse rejects 3 fields
    with pytest.raises(ValueError, match="expected 4"):
        MoEImplId.parse("only.three.fields")

    # MoEImplId.parse rejects 5 fields
    with pytest.raises(ValueError, match="expected 4"):
        MoEImplId.parse("one.two.three.four.five")


def test_moe_impl_registry_duplicate_rejection():
    registry = MoEImplRegistry()

    test_id = MoEImplId(provider="testprov", technique="testtec", quant="bf16", kernel_name="kern1")
    desc = MoEImplDescriptor(
        identity=test_id,
        scheduler_kind=MoESchedulerKind.EXTERNAL_COMM,
    )

    class FakeImplA:
        descriptor = desc

    result = registry.register(FakeImplA)
    assert result is FakeImplA
    assert registry.lookup(test_id) is FakeImplA
    assert len(registry) == 1

    # Duplicate raises ValueError
    class FakeImplB:
        descriptor = desc

    with pytest.raises(ValueError, match="duplicate"):
        registry.register(FakeImplB)

    # Missing descriptor raises TypeError
    class NoDescriptor:
        pass

    with pytest.raises(TypeError, match="MoEImplDescriptor"):
        registry.register(NoDescriptor)

    # Lookup returns None for unregistered id
    unknown_id = MoEImplId(provider="other", technique="other", quant="fp8", kernel_name="unknown")
    assert registry.lookup(unknown_id) is None


def test_moe_eligibility_invariant_enforcement():
    # eligible=True without reject_reason succeeds
    e = MoEEligibility(eligible=True)
    assert e.eligible is True
    assert e.reject_reason is None

    # eligible=False with a MoERejectReason succeeds
    e2 = MoEEligibility(eligible=False, reject_reason=MoERejectReason.SM_UNSUPPORTED)
    assert e2.eligible is False
    assert e2.reject_reason is MoERejectReason.SM_UNSUPPORTED

    # eligible=True with reject_reason raises ValueError
    with pytest.raises(ValueError, match="eligible=True must not carry a reject_reason"):
        MoEEligibility(eligible=True, reject_reason=MoERejectReason.SM_UNSUPPORTED)

    # eligible=False without reject_reason raises ValueError
    with pytest.raises(ValueError, match="a rejection must name a MoERejectReason"):
        MoEEligibility(eligible=False)
