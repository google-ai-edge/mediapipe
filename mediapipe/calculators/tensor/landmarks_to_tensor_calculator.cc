// Copyright 2021 The MediaPipe Authors.
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

#include "mediapipe/calculators/tensor/landmarks_to_tensor_calculator.h"

#include <memory>

#include "mediapipe/calculators/tensor/landmarks_to_tensor_calculator.pb.h"
#include "mediapipe/framework/api2/node.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/formats/landmark.pb.h"
#include "mediapipe/framework/formats/tensor.h"
#include "mediapipe/framework/port/ret_check.h"

namespace mediapipe {
namespace api2 {

namespace {

float GetAttribute(
    const Landmark& landmark,
    const LandmarksToTensorCalculatorOptions::Attribute& attribute) {
  switch (attribute) {
    case LandmarksToTensorCalculatorOptions::X:
      return landmark.x();
    case LandmarksToTensorCalculatorOptions::Y:
      return landmark.y();
    case LandmarksToTensorCalculatorOptions::Z:
      return landmark.z();
    case LandmarksToTensorCalculatorOptions::VISIBILITY:
      return landmark.visibility();
    case LandmarksToTensorCalculatorOptions::PRESENCE:
      return landmark.presence();
  }
}

}  // namespace

class LandmarksToTensorCalculatorImpl
    : public NodeImpl<LandmarksToTensorCalculator> {
 public:
  absl::Status Open(CalculatorContext* cc) override {
    options_ = cc->Options<LandmarksToTensorCalculatorOptions>();
    RET_CHECK(options_.attributes_size() > 0)
        << "At least one attribute must be specified";
    return absl::OkStatus();
  }

  absl::Status Process(CalculatorContext* cc) override {
    if (kInLandmarkList(cc).IsEmpty()) {
      return absl::OkStatus();
    }

    // Get input landmarks.
    const auto& in_landmarks = *kInLandmarkList(cc);

    // Determine tensor shape.
    const int n_landmarks = in_landmarks.landmark_size();
    const int n_attributes = options_.attributes_size();
    auto tensor_shape = options_.flatten()
                            ? Tensor::Shape{1, n_landmarks * n_attributes}
                            : Tensor::Shape{1, n_landmarks, n_attributes};

    // Create empty tesnor.
    Tensor tensor(Tensor::ElementType::kFloat32, tensor_shape);
    auto* buffer = tensor.GetCpuWriteView().buffer<float>();

    // Fill tensor with landmark attributes.
    for (int i = 0; i < n_landmarks; ++i) {
      for (int j = 0; j < n_attributes; ++j) {
        buffer[i * n_attributes + j] =
            GetAttribute(in_landmarks.landmark(i), options_.attributes(j));
      }
    }

    // Return vector with a single tensor.
    auto result = std::vector<Tensor>();
    result.push_back(std::move(tensor));
    kOutTensors(cc).Send(std::move(result));

    return absl::OkStatus();
  }

 private:
  LandmarksToTensorCalculatorOptions options_;
};
MEDIAPIPE_NODE_IMPLEMENTATION(LandmarksToTensorCalculatorImpl);

}  // namespace api2
}  // namespace mediapipe
