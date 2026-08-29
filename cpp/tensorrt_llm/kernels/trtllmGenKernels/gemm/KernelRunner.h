/*
 * Copyright (c) 2020-2026, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include <cstdint>
#include <cuda.h>
#include <optional>
#include <vector>

#include "tensorrt_llm/common/config.h"

#include "trtllmGen_gemm_export/trtllm/gen/DtypeDecl.h"

TRTLLM_NAMESPACE_BEGIN

namespace kernels
{

struct TrtllmGenGemmProblemDimensions
{
    int32_t mM{0};
    int32_t mValidM{0};
    int32_t mN{0};
    int32_t mValidN{0};
    int32_t mK{0};
    int32_t mValidK{0};
    int32_t mRank{0};
    int32_t mWorldSize{0};
};

inline TrtllmGenGemmProblemDimensions makeTrtllmGenGemmProblemDimensions(
    int32_t m, int32_t n, int32_t k, bool transposeMmaOutput)
{
    TrtllmGenGemmProblemDimensions problemDimensions;
    problemDimensions.mM = transposeMmaOutput ? n : m;
    problemDimensions.mN = transposeMmaOutput ? m : n;
    problemDimensions.mK = k;
    problemDimensions.mValidM = transposeMmaOutput ? n : m;
    problemDimensions.mValidN = transposeMmaOutput ? m : n;
    problemDimensions.mValidK = k;
    problemDimensions.mRank = 0;
    problemDimensions.mWorldSize = 1;
    return problemDimensions;
}

struct TrtllmGenGemmRunnerOptions
{
    gemm::trtllm::gen::Dtype eltTypeA;
    gemm::trtllm::gen::Dtype eltTypeB{gemm::trtllm::gen::Dtype::Void};
    gemm::trtllm::gen::Dtype outputType;
    bool deepSeekFp8{false};
    bool transposeMmaOutput{false};
};

class TrtllmGenGemmRunner
{
public:
    explicit TrtllmGenGemmRunner(TrtllmGenGemmRunnerOptions const& options);

    [[nodiscard]] size_t getWorkspaceSizeInBytes(int32_t m, int32_t n, int32_t k);

    void run(int32_t m, int32_t n, int32_t k, void const* a, float const* aScale, void const* b, float const* bScale,
        void* c, float* cScale, float* cScalePtr, void* workspace, CUstream stream, int device);

    void run(int32_t m, int32_t n, int32_t k, void const* a, void const* b, void* c, float* cScale, void* workspace,
        CUstream stream, int device);

private:
    void selectGemmConfig(int32_t m, int32_t n, int32_t k);

private:
    TrtllmGenGemmRunnerOptions mOptions;
    std::optional<int> mSelectedConfigIndex;
    std::vector<int32_t> mPassingConfigIndices;
};
} // namespace kernels

TRTLLM_NAMESPACE_END
