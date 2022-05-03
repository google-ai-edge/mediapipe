// Copyright 2019 The MediaPipe Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "mediapipe/calculators/core/begin_loop_calculator.h"

#include <vector>

#include "mediapipe/framework/formats/detection.pb.h"
#include "mediapipe/framework/formats/landmark.pb.h"
#include "mediapipe/framework/formats/matrix.h"
#include "mediapipe/framework/formats/rect.pb.h"

namespace mediapipe {

// A calculator to process std::vector<NormalizedLandmarkList>.
typedef BeginLoopCalculator<std::vector<::mediapipe::NormalizedLandmarkList>>
    BeginLoopNormalizedLandmarkListVectorCalculator;
REGISTER_CALCULATOR(BeginLoopNormalizedLandmarkListVectorCalculator);

// A calculator to process std::vector<int>.
typedef BeginLoopCalculator<std::vector<int>> BeginLoopIntCalculator;
REGISTER_CALCULATOR(BeginLoopIntCalculator);

// A calculator to process std::vector<NormalizedRect>.
typedef BeginLoopCalculator<std::vector<::mediapipe::NormalizedRect>>
    BeginLoopNormalizedRectCalculator;
REGISTER_CALCULATOR(BeginLoopNormalizedRectCalculator);

// A calculator to process std::vector<Detection>.
typedef BeginLoopCalculator<std::vector<::mediapipe::Detection>>
    BeginLoopDetectionCalculator;
REGISTER_CALCULATOR(BeginLoopDetectionCalculator);

// A calculator to process std::vector<Matrix>.
typedef BeginLoopCalculator<std::vector<Matrix>> BeginLoopMatrixCalculator;
REGISTER_CALCULATOR(BeginLoopMatrixCalculator);

// A calculator to process std::vector<std::vector<Matrix>>.
typedef BeginLoopCalculator<std::vector<std::vector<Matrix>>>
    BeginLoopMatrixVectorCalculator;
REGISTER_CALCULATOR(BeginLoopMatrixVectorCalculator);

// A calculator to process std::vector<uint64_t>.
typedef BeginLoopCalculator<std::vector<uint64_t>> BeginLoopUint64tCalculator;
REGISTER_CALCULATOR(BeginLoopUint64tCalculator);

}  // namespace mediapipe
