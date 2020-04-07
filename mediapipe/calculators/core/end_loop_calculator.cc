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

#include "mediapipe/calculators/core/end_loop_calculator.h"

#include <vector>

#include "mediapipe/framework/formats/classification.pb.h"
#include "mediapipe/framework/formats/landmark.pb.h"
#include "mediapipe/framework/formats/rect.pb.h"
#include "mediapipe/util/render_data.pb.h"
#include "tensorflow/lite/interpreter.h"

namespace mediapipe {

typedef EndLoopCalculator<std::vector<::mediapipe::NormalizedRect>>
    EndLoopNormalizedRectCalculator;
REGISTER_CALCULATOR(EndLoopNormalizedRectCalculator);

typedef EndLoopCalculator<std::vector<::mediapipe::NormalizedLandmarkList>>
    EndLoopNormalizedLandmarkListVectorCalculator;
REGISTER_CALCULATOR(EndLoopNormalizedLandmarkListVectorCalculator);

typedef EndLoopCalculator<std::vector<bool>> EndLoopBooleanCalculator;
REGISTER_CALCULATOR(EndLoopBooleanCalculator);

typedef EndLoopCalculator<std::vector<::mediapipe::RenderData>>
    EndLoopRenderDataCalculator;
REGISTER_CALCULATOR(EndLoopRenderDataCalculator);

typedef EndLoopCalculator<std::vector<::mediapipe::ClassificationList>>
    EndLoopClassificationListCalculator;
REGISTER_CALCULATOR(EndLoopClassificationListCalculator);

typedef EndLoopCalculator<std::vector<TfLiteTensor>> EndLoopTensorCalculator;
REGISTER_CALCULATOR(EndLoopTensorCalculator);

}  // namespace mediapipe
