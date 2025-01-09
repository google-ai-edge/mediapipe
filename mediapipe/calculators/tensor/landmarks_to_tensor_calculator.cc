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
#include <optional>
#include <type_traits>
#include <vector>

#include "mediapipe/calculators/tensor/landmarks_to_tensor_calculator.pb.h"
#include "mediapipe/framework/api2/node.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/formats/landmark.pb.h"
#include "mediapipe/framework/formats/tensor.h"
#include "mediapipe/framework/memory_manager.h"
#include "mediapipe/framework/memory_manager_service.h"
#include "mediapipe/framework/port/ret_check.h"

namespace mediapipe {
namespace api2 {

namespace {

// Returns the scale attribute should be multiplied by.
float GetAttributeScale(
    const LandmarksToTensorCalculatorOptions::Attribute& attribute,
    const std::pair<int, int>& image_size) {
  switch (attribute) {
    case LandmarksToTensorCalculatorOptions::X:
    case LandmarksToTensorCalculatorOptions::Z:
      return image_size.first;
    case LandmarksToTensorCalculatorOptions::Y:
      return image_size.second;
    case LandmarksToTensorCalculatorOptions::VISIBILITY:
    case LandmarksToTensorCalculatorOptions::PRESENCE:
      return 1.0f;
  }
}

template <typename LandmarkType>
float GetAttribute(
    const LandmarkType& landmark,
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

template <typename LandmarksT>
Tensor ConvertLandmarksToTensor(
    const LandmarksT& landmarks, const std::vector<float>& attribute_scales,
    const LandmarksToTensorCalculatorOptions& options,
    MemoryManager* memory_manager) {
  // Determine tensor shape.
  const int n_landmarks = landmarks.landmark_size();
  const int n_attributes = options.attributes_size();
  auto tensor_shape = options.flatten()
                          ? Tensor::Shape{1, n_landmarks * n_attributes}
                          : Tensor::Shape{1, n_landmarks, n_attributes};

  // Create empty tesnor.
  Tensor tensor(Tensor::ElementType::kFloat32, tensor_shape, memory_manager);
  auto* buffer = tensor.GetCpuWriteView().buffer<float>();

  // Fill tensor with landmark attributes.
  for (int i = 0; i < n_landmarks; ++i) {
    for (int j = 0; j < n_attributes; ++j) {
      float value = GetAttribute(landmarks.landmark(i), options.attributes(j));
      float scale = attribute_scales[j];
      buffer[i * n_attributes + j] = value * scale;
    }
  }

  return tensor;
}

}  // namespace

class LandmarksToTensorCalculatorImpl
    : public NodeImpl<LandmarksToTensorCalculator> {
 public:
  absl::Status Open(CalculatorContext* cc) override {
    if (cc->Service(kMemoryManagerService).IsAvailable()) {
      memory_manager_ = &cc->Service(kMemoryManagerService).GetObject();
    }
    options_ = cc->Options<LandmarksToTensorCalculatorOptions>();
    RET_CHECK(options_.attributes_size() > 0)
        << "At least one attribute must be specified";

    RET_CHECK(kInLandmarkList(cc).IsConnected() ^
              kInNormLandmarkList(cc).IsConnected())
        << "Exactly one landmarks input should be provided";
    RET_CHECK_EQ(kInNormLandmarkList(cc).IsConnected(),
                 kImageSize(cc).IsConnected())
        << "Image size should be provided only for normalized landmarks";

    return absl::OkStatus();
  }

  absl::Status Process(CalculatorContext* cc) override {
    // Get attribute scales depending on whether landmarks are normalized or
    // not.
    std::vector<float> attribute_scales;
    if (kInLandmarkList(cc).IsConnected()) {
      for (int j = 0; j < options_.attributes_size(); ++j) {
        attribute_scales.push_back(1.0f);
      }
    } else {
      RET_CHECK(!kImageSize(cc).IsEmpty());
      auto image_size = kImageSize(cc).Get();
      for (int j = 0; j < options_.attributes_size(); ++j) {
        attribute_scales.push_back(
            GetAttributeScale(options_.attributes(j), image_size));
      }
    }

    // Convert landmarks to tensor.
    auto result = std::vector<Tensor>();
    if (kInLandmarkList(cc).IsConnected()) {
      if (kInLandmarkList(cc).IsEmpty()) {
        return absl::OkStatus();
      }
      Tensor tensor =
          ConvertLandmarksToTensor(kInLandmarkList(cc).Get(), attribute_scales,
                                   options_, memory_manager_);
      result.push_back(std::move(tensor));
    } else {
      if (kInNormLandmarkList(cc).IsEmpty()) {
        return absl::OkStatus();
      }
      Tensor tensor =
          ConvertLandmarksToTensor(kInNormLandmarkList(cc).Get(),
                                   attribute_scales, options_, memory_manager_);
      result.push_back(std::move(tensor));
    }

    kOutTensors(cc).Send(std::move(result));

    return absl::OkStatus();
  }

  static absl::Status UpdateContract(CalculatorContract* cc) {
    cc->UseService(kMemoryManagerService).Optional();
    return absl::OkStatus();
  }

 private:
  LandmarksToTensorCalculatorOptions options_;
  // Enable pooling of AHWBs in Tensor instances.
  MemoryManager* memory_manager_ = nullptr;
};
MEDIAPIPE_NODE_IMPLEMENTATION(LandmarksToTensorCalculatorImpl);

}  // namespace api2
}  // namespace mediapipe
