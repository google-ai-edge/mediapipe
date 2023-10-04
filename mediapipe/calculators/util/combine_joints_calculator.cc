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

#include "mediapipe/calculators/util/combine_joints_calculator.h"

#include <utility>

#include "mediapipe/calculators/util/combine_joints_calculator.pb.h"
#include "mediapipe/framework/api2/node.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/formats/body_rig.pb.h"
#include "mediapipe/framework/port/ret_check.h"

namespace mediapipe {
namespace api2 {

namespace {}  // namespace

class CombineJointsCalculatorImpl : public NodeImpl<CombineJointsCalculator> {
 public:
  absl::Status Open(CalculatorContext* cc) override {
    options_ = cc->Options<CombineJointsCalculatorOptions>();
    RET_CHECK_GE(options_.num_joints(), 0);
    RET_CHECK_GT(kInJoints(cc).Count(), 0);
    RET_CHECK_EQ(kInJoints(cc).Count(), options_.joints_mapping_size());
    RET_CHECK(options_.has_default_joint());
    for (const auto& mapping : options_.joints_mapping()) {
      for (int idx : mapping.idx()) {
        RET_CHECK_GE(idx, 0);
        RET_CHECK_LT(idx, options_.num_joints());
      }
    }
    return absl::OkStatus();
  }

  absl::Status Process(CalculatorContext* cc) override {
    // Initialize output joints with default values.
    JointList out_joints;
    for (int i = 0; i < options_.num_joints(); ++i) {
      *out_joints.add_joint() = options_.default_joint();
    }

    // Override default joints with provided joints.
    for (int i = 0; i < kInJoints(cc).Count(); ++i) {
      // Skip empty joint streams.
      if (kInJoints(cc)[i].IsEmpty()) {
        continue;
      }

      const JointList& in_joints = kInJoints(cc)[i].Get();
      const auto& mapping = options_.joints_mapping(i);
      RET_CHECK_EQ(in_joints.joint_size(), mapping.idx_size());
      for (int j = 0; j < in_joints.joint_size(); ++j) {
        *out_joints.mutable_joint(mapping.idx(j)) = in_joints.joint(j);
      }
    }

    kOutJoints(cc).Send(std::move(out_joints));
    return absl::OkStatus();
  }

 private:
  CombineJointsCalculatorOptions options_;
};
MEDIAPIPE_NODE_IMPLEMENTATION(CombineJointsCalculatorImpl);

}  // namespace api2
}  // namespace mediapipe
