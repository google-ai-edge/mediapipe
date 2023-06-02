// Copyright 2022 The MediaPipe Authors.
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

#include "mediapipe/calculators/core/get_vector_item_calculator.h"

#include "mediapipe/framework/formats/classification.pb.h"
#include "mediapipe/framework/formats/detection.pb.h"
#include "mediapipe/framework/formats/landmark.pb.h"
#include "mediapipe/framework/formats/rect.pb.h"

namespace mediapipe {
namespace api2 {

using GetLandmarkListVectorItemCalculator =
    GetVectorItemCalculator<mediapipe::LandmarkList>;
REGISTER_CALCULATOR(GetLandmarkListVectorItemCalculator);

using GetNormalizedLandmarkListVectorItemCalculator =
    GetVectorItemCalculator<mediapipe::NormalizedLandmarkList>;
REGISTER_CALCULATOR(GetNormalizedLandmarkListVectorItemCalculator);

using GetClassificationListVectorItemCalculator =
    GetVectorItemCalculator<mediapipe::ClassificationList>;
REGISTER_CALCULATOR(GetClassificationListVectorItemCalculator);

using GetDetectionVectorItemCalculator =
    GetVectorItemCalculator<mediapipe::Detection>;
REGISTER_CALCULATOR(GetDetectionVectorItemCalculator);

using GetNormalizedRectVectorItemCalculator =
    GetVectorItemCalculator<NormalizedRect>;
REGISTER_CALCULATOR(GetNormalizedRectVectorItemCalculator);

using GetRectVectorItemCalculator = GetVectorItemCalculator<Rect>;
REGISTER_CALCULATOR(GetRectVectorItemCalculator);

}  // namespace api2
}  // namespace mediapipe
