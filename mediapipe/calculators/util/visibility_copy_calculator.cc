// Copyright 2025 The MediaPipe Authors.
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
#include "mediapipe/calculators/util/visibility_copy_calculator.h"

#include "absl/status/status.h"
#include "mediapipe/calculators/util/visibility_copy_calculator.pb.h"
#include "mediapipe/framework/api3/calculator.h"
#include "mediapipe/framework/api3/calculator_context.h"
#include "mediapipe/framework/formats/landmark.pb.h"

namespace mediapipe::api3 {

class VisibilityCopyNodeImpl
    : public Calculator<VisibilityCopyNode, VisibilityCopyNodeImpl> {
 public:
  absl::Status Open(CalculatorContext<VisibilityCopyNode>& cc) final;
  absl::Status Process(CalculatorContext<VisibilityCopyNode>& cc) final;

 private:
  template <class From, class To, class Output>
  absl::Status CopyVisibility(const From& from, const To& to, Output& out);

  bool copy_visibility_;
  bool copy_presence_;
};

absl::Status VisibilityCopyNodeImpl::Open(
    CalculatorContext<VisibilityCopyNode>& cc) {
  const auto& options = cc.options.Get();
  copy_visibility_ = options.copy_visibility();
  copy_presence_ = options.copy_presence();
  return absl::OkStatus();
}

absl::Status VisibilityCopyNodeImpl::Process(
    CalculatorContext<VisibilityCopyNode>& cc) {
  if (cc.out_landmarks_to.IsConnected()) {
    if (cc.in_landmarks_from.IsConnected()) {
      return CopyVisibility(cc.in_landmarks_from, cc.in_landmarks_to,
                            cc.out_landmarks_to);
    }
    return CopyVisibility(cc.in_norm_landmarks_from, cc.in_landmarks_to,
                          cc.out_landmarks_to);
  }

  if (cc.in_landmarks_from.IsConnected()) {
    return CopyVisibility(cc.in_landmarks_from, cc.in_norm_landmarks_to,
                          cc.out_norm_landmarks_to);
  }
  return CopyVisibility(cc.in_norm_landmarks_from, cc.in_norm_landmarks_to,
                        cc.out_norm_landmarks_to);
}

template <class From, class To, class Output>
absl::Status VisibilityCopyNodeImpl::CopyVisibility(const From& from,
                                                    const To& to, Output& out) {
  // Check that both landmarks to copy from and to copy are non empty.
  if (!from || !to) {
    return absl::OkStatus();
  }

  const auto& landmarks_from = from.GetOrDie();
  const auto& landmarks_to = to.GetOrDie();
  auto landmarks_out = absl::make_unique<typename Output::Payload>();

  for (int i = 0; i < landmarks_from.landmark_size(); ++i) {
    const auto& landmark_from = landmarks_from.landmark(i);
    const auto& landmark_to = landmarks_to.landmark(i);

    // Create output landmark and copy all fields from the `to` landmark.
    const auto& landmark_out = landmarks_out->add_landmark();
    *landmark_out = landmark_to;

    // Copy visibility and presence from the `from` landmark.
    if (copy_visibility_) {
      landmark_out->set_visibility(landmark_from.visibility());
    }
    if (copy_presence_) {
      landmark_out->set_presence(landmark_from.presence());
    }
  }

  out.Send(std::move(landmarks_out));
  return absl::OkStatus();
}

}  // namespace mediapipe::api3
