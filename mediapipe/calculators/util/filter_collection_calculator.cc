
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

#include "mediapipe/calculators/util/filter_collection_calculator.h"

#include <vector>

#include "mediapipe/framework/formats/classification.pb.h"
#include "mediapipe/framework/formats/landmark.pb.h"
#include "mediapipe/framework/formats/rect.pb.h"

namespace mediapipe {

typedef FilterCollectionCalculator<std::vector<::mediapipe::NormalizedRect>>
    FilterNormalizedRectCollectionCalculator;
REGISTER_CALCULATOR(FilterNormalizedRectCollectionCalculator);

typedef FilterCollectionCalculator<
    std::vector<::mediapipe::NormalizedLandmarkList>>
    FilterLandmarkListCollectionCalculator;
REGISTER_CALCULATOR(FilterLandmarkListCollectionCalculator);

typedef FilterCollectionCalculator<std::vector<::mediapipe::ClassificationList>>
    FilterClassificationListCollectionCalculator;
REGISTER_CALCULATOR(FilterClassificationListCollectionCalculator);

}  // namespace mediapipe
