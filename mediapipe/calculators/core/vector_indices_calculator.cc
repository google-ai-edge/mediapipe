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

#include "mediapipe/calculators/core/vector_indices_calculator.h"

#include "mediapipe/framework/formats/landmark.pb.h"

namespace mediapipe {
namespace api2 {

using IntVectorIndicesCalculator = VectorIndicesCalculator<int>;
REGISTER_CALCULATOR(IntVectorIndicesCalculator);

using Uint64tVectorIndicesCalculator = VectorIndicesCalculator<uint64_t>;
REGISTER_CALCULATOR(Uint64tVectorIndicesCalculator);

using NormalizedLandmarkListVectorIndicesCalculator =
    VectorIndicesCalculator<mediapipe::NormalizedLandmarkList>;
REGISTER_CALCULATOR(NormalizedLandmarkListVectorIndicesCalculator);

}  // namespace api2
}  // namespace mediapipe
