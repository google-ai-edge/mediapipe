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

#include "mediapipe/calculators/util/align_hand_to_pose_in_world_calculator.h"

#include <utility>

#include "absl/status/status.h"
#include "mediapipe/calculators/util/align_hand_to_pose_in_world_calculator.pb.h"
#include "mediapipe/framework/api2/node.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/formats/landmark.pb.h"
#include "mediapipe/framework/port/ret_check.h"

namespace mediapipe::api2 {

namespace {}  // namespace

class AlignHandToPoseInWorldCalculatorImpl
    : public NodeImpl<AlignHandToPoseInWorldCalculator> {
 public:
  absl::Status Open(CalculatorContext* cc) override {
    const auto& options =
        cc->Options<mediapipe::AlignHandToPoseInWorldCalculatorOptions>();
    hand_wrist_idx_ = options.hand_wrist_idx();
    pose_wrist_idx_ = options.pose_wrist_idx();
    return absl::OkStatus();
  }

  absl::Status Process(CalculatorContext* cc) override {
    // Check that landmarks are not empty.
    if (kInHandLandmarks(cc).IsEmpty()) {
      return absl::OkStatus();
    }

    // Check that pose landmarks are provided.
    RET_CHECK(!kInPoseLandmarks(cc).IsEmpty());

    const auto& in_hand_landmarks = kInHandLandmarks(cc).Get();
    const auto& in_pose_landmarks = kInPoseLandmarks(cc).Get();

    // Get hand and pose wrists.
    RET_CHECK(hand_wrist_idx_ <= in_hand_landmarks.landmark_size());
    const auto& hand_wrist = in_hand_landmarks.landmark(hand_wrist_idx_);
    RET_CHECK(pose_wrist_idx_ <= in_pose_landmarks.landmark_size());
    const auto& pose_wrist = in_pose_landmarks.landmark(pose_wrist_idx_);

    LandmarkList out_hand_landmarks;
    for (int i = 0; i < in_hand_landmarks.landmark_size(); ++i) {
      const auto& in_landmark = in_hand_landmarks.landmark(i);
      Landmark* out_landmark = out_hand_landmarks.add_landmark();
      *out_landmark = in_landmark;
      out_landmark->set_x(in_landmark.x() - hand_wrist.x() + pose_wrist.x());
      out_landmark->set_y(in_landmark.y() - hand_wrist.y() + pose_wrist.y());
      out_landmark->set_z(in_landmark.z() - hand_wrist.z() + pose_wrist.z());
    }

    kOutHandLandmarks(cc).Send(std::move(out_hand_landmarks));

    return absl::OkStatus();
  }

 private:
  int hand_wrist_idx_;
  int pose_wrist_idx_;
};
MEDIAPIPE_NODE_IMPLEMENTATION(AlignHandToPoseInWorldCalculatorImpl);

}  // namespace mediapipe::api2
