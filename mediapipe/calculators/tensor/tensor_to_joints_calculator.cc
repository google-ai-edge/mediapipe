// Copyright 2023 The MediaPipe Authors.
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

#include "mediapipe/calculators/tensor/tensor_to_joints_calculator.h"

#include <utility>

#include "mediapipe/calculators/tensor/tensor_to_joints_calculator.pb.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/formats/body_rig.pb.h"
#include "mediapipe/framework/formats/tensor.h"
#include "mediapipe/framework/port/ret_check.h"

namespace mediapipe {
namespace api2 {
namespace {

// Number of values in 6D representation of rotation.
constexpr int kRotation6dSize = 6;

}  // namespace

class TensorToJointsCalculatorImpl
    : public mediapipe::api2::NodeImpl<TensorToJointsCalculator> {
 public:
  absl::Status Open(CalculatorContext* cc) override {
    const auto& options = cc->Options<TensorToJointsCalculatorOptions>();

    // Get number of joints.
    RET_CHECK_GE(options.num_joints(), 0);
    num_joints_ = options.num_joints();

    // Get start index.
    start_index_ = options.start_index();

    return absl::OkStatus();
  }

  absl::Status Process(CalculatorContext* cc) override {
    // Skip if Tensor is empty.
    if (kInTensor(cc).IsEmpty()) {
      return absl::OkStatus();
    }

    // Get raw floats from the Tensor.
    const Tensor& tensor = kInTensor(cc).Get();
    RET_CHECK_EQ(tensor.shape().num_elements(),
                 num_joints_ * kRotation6dSize + start_index_)
        << "Unexpected number of values in Tensor";
    const float* raw_floats = tensor.GetCpuReadView().buffer<float>();

    // Convert raw floats into Joint rotations.
    JointList joints;
    for (int joint_idx = 0; joint_idx < num_joints_; ++joint_idx) {
      Joint* joint = joints.add_joint();
      for (int idx_6d = 0; idx_6d < kRotation6dSize; ++idx_6d) {
        joint->add_rotation_6d(
            raw_floats[start_index_ + joint_idx * kRotation6dSize + idx_6d]);
      }
    }

    kOutJoints(cc).Send(std::move(joints));
    return absl::OkStatus();
  }

 private:
  int num_joints_ = 0;
  int start_index_ = 0;
};
MEDIAPIPE_NODE_IMPLEMENTATION(TensorToJointsCalculatorImpl);

}  // namespace api2
}  // namespace mediapipe
