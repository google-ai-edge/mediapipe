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

#ifndef MEDIAPIPE_CALCULATORS_CORE_CONCATENATE_PROTO_LIST_CALCULATOR_H_  // NOLINT
#define MEDIAPIPE_CALCULATORS_CORE_CONCATENATE_PROTO_LIST_CALCULATOR_H_  // NOLINT

#include "mediapipe/calculators/core/concatenate_vector_calculator.pb.h"
#include "mediapipe/framework/api2/node.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/formats/body_rig.pb.h"
#include "mediapipe/framework/formats/classification.pb.h"
#include "mediapipe/framework/formats/landmark.pb.h"
#include "mediapipe/framework/port/canonical_errors.h"
#include "mediapipe/framework/port/ret_check.h"
#include "mediapipe/framework/port/status.h"

namespace mediapipe {
namespace api2 {

// Concatenate several input packets of ListType with a repeated field of
// ItemType into a single output packet of ListType following stream index
// order.
template <typename ItemType, typename ListType>
class ConcatenateListsCalculator : public Node {
 public:
  static constexpr typename Input<ListType>::Multiple kIn{""};
  static constexpr Output<ListType> kOut{""};

  MEDIAPIPE_NODE_CONTRACT(kIn, kOut);

  static absl::Status UpdateContract(CalculatorContract* cc) {
    RET_CHECK_GE(kIn(cc).Count(), 1);
    return absl::OkStatus();
  }

  absl::Status Open(CalculatorContext* cc) override {
    only_emit_if_all_present_ =
        cc->Options<::mediapipe::ConcatenateVectorCalculatorOptions>()
            .only_emit_if_all_present();
    return absl::OkStatus();
  }

  absl::Status Process(CalculatorContext* cc) override {
    if (only_emit_if_all_present_) {
      for (const auto& input : kIn(cc)) {
        if (input.IsEmpty()) return absl::OkStatus();
      }
    }

    ListType output;
    for (const auto& input : kIn(cc)) {
      if (input.IsEmpty()) continue;
      const ListType& list = *input;
      for (int j = 0; j < ListSize(list); ++j) {
        *AddItem(output) = GetItem(list, j);
      }
    }
    kOut(cc).Send(std::move(output));
    return absl::OkStatus();
  }

 protected:
  virtual int ListSize(const ListType& list) const = 0;
  virtual const ItemType GetItem(const ListType& list, int idx) const = 0;
  virtual ItemType* AddItem(ListType& list) const = 0;

 private:
  bool only_emit_if_all_present_;
};

// TODO: Move calculators to separate *.cc files

class ConcatenateNormalizedLandmarkListCalculator
    : public ConcatenateListsCalculator<NormalizedLandmark,
                                        NormalizedLandmarkList> {
 protected:
  int ListSize(const NormalizedLandmarkList& list) const override {
    return list.landmark_size();
  }
  const NormalizedLandmark GetItem(const NormalizedLandmarkList& list,
                                   int idx) const override {
    return list.landmark(idx);
  }
  NormalizedLandmark* AddItem(NormalizedLandmarkList& list) const override {
    return list.add_landmark();
  }
};
MEDIAPIPE_REGISTER_NODE(ConcatenateNormalizedLandmarkListCalculator);

class ConcatenateLandmarkListCalculator
    : public ConcatenateListsCalculator<Landmark, LandmarkList> {
 protected:
  int ListSize(const LandmarkList& list) const override {
    return list.landmark_size();
  }
  const Landmark GetItem(const LandmarkList& list, int idx) const override {
    return list.landmark(idx);
  }
  Landmark* AddItem(LandmarkList& list) const override {
    return list.add_landmark();
  }
};
MEDIAPIPE_REGISTER_NODE(ConcatenateLandmarkListCalculator);

class ConcatenateClassificationListCalculator
    : public ConcatenateListsCalculator<Classification, ClassificationList> {
 protected:
  int ListSize(const ClassificationList& list) const override {
    return list.classification_size();
  }
  const Classification GetItem(const ClassificationList& list,
                               int idx) const override {
    return list.classification(idx);
  }
  Classification* AddItem(ClassificationList& list) const override {
    return list.add_classification();
  }
};
MEDIAPIPE_REGISTER_NODE(ConcatenateClassificationListCalculator);

class ConcatenateJointListCalculator
    : public ConcatenateListsCalculator<Joint, JointList> {
 protected:
  int ListSize(const JointList& list) const override {
    return list.joint_size();
  }
  const Joint GetItem(const JointList& list, int idx) const override {
    return list.joint(idx);
  }
  Joint* AddItem(JointList& list) const override { return list.add_joint(); }
};
MEDIAPIPE_REGISTER_NODE(ConcatenateJointListCalculator);

}  // namespace api2
}  // namespace mediapipe

// NOLINTNEXTLINE
#endif  // MEDIAPIPE_CALCULATORS_CORE_CONCATENATE_PROTO_LIST_CALCULATOR_H_
