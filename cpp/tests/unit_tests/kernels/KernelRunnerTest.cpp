// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <gtest/gtest.h>

#include "tensorrt_llm/kernels/trtllmGenKernels/gemm/KernelRunner.h"

namespace
{

namespace tk = tensorrt_llm::kernels;

TEST(TrtllmGenGemmRunnerTest, ValidProblemDimsPassedToGemmData)
{
    int32_t const m = 64;
    int32_t const n = 128;
    int32_t const k = 256;

    auto const nonTransposeData = tk::makeTrtllmGenGemmProblemDimensions(m, n, k, /*transposeMmaOutput=*/false);
    EXPECT_EQ(nonTransposeData.mM, m);
    EXPECT_EQ(nonTransposeData.mN, n);
    EXPECT_EQ(nonTransposeData.mK, k);
    EXPECT_EQ(nonTransposeData.mValidM, m);
    EXPECT_EQ(nonTransposeData.mValidN, n);
    EXPECT_EQ(nonTransposeData.mValidK, k);
    EXPECT_EQ(nonTransposeData.mRank, 0);
    EXPECT_EQ(nonTransposeData.mWorldSize, 1);

    auto const transposeData = tk::makeTrtllmGenGemmProblemDimensions(m, n, k, /*transposeMmaOutput=*/true);
    EXPECT_EQ(transposeData.mM, n);
    EXPECT_EQ(transposeData.mN, m);
    EXPECT_EQ(transposeData.mK, k);
    EXPECT_EQ(transposeData.mValidM, n);
    EXPECT_EQ(transposeData.mValidN, m);
    EXPECT_EQ(transposeData.mValidK, k);
    EXPECT_EQ(transposeData.mRank, 0);
    EXPECT_EQ(transposeData.mWorldSize, 1);
}

} // namespace
