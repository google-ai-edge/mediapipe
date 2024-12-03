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

#include "mediapipe/calculators/util/landmarks_refinement_calculator.h"

#include <algorithm>
#include <set>
#include <utility>

#include "absl/log/absl_check.h"
#include "absl/memory/memory.h"
#include "mediapipe/calculators/util/landmarks_refinement_calculator.pb.h"
#include "mediapipe/framework/api2/node.h"
#include "mediapipe/framework/api2/port.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/port/proto_ns.h"
#include "mediapipe/framework/port/ret_check.h"

namespace mediapipe {

namespace api2 {

namespace {

absl::StatusOr<int> GetNumberOfRefinedLandmarks(
    const proto_ns::RepeatedPtrField<
        LandmarksRefinementCalculatorOptions::Refinement>& refinements) {
  // Gather all used indexes.
  std::set<int> idxs;
  for (int i = 0; i < refinements.size(); ++i) {
    const auto& refinement = refinements.Get(i);
    for (int i = 0; i < refinement.indexes_mapping_size(); ++i) {
      idxs.insert(refinement.indexes_mapping(i));
    }
  }

  // Check that indxes start with 0 and there is no gaps between min and max
  // indexes.
  RET_CHECK(!idxs.empty())
      << "There should be at least one landmark in indexes mapping";
  int idxs_min = *idxs.begin();
  int idxs_max = *idxs.rbegin();
  int n_idxs = idxs.size();
  RET_CHECK_EQ(idxs_min, 0)
      << "Indexes are expected to start with 0 instead of " << idxs_min;
  RET_CHECK_EQ(idxs_max, n_idxs - 1)
      << "Indexes should have no gaps but " << idxs_max - n_idxs + 1
      << " indexes are missing";

  return n_idxs;
}

void RefineXY(const proto_ns::RepeatedField<int>& indexes_mapping,
              const NormalizedLandmarkList& landmarks,
              NormalizedLandmarkList* refined_landmarks) {
  for (int i = 0; i < landmarks.landmark_size(); ++i) {
    const auto& landmark = landmarks.landmark(i);
    auto* refined_landmark =
        refined_landmarks->mutable_landmark(indexes_mapping.Get(i));
    refined_landmark->set_x(landmark.x());
    refined_landmark->set_y(landmark.y());
  }
}

float GetZAverage(const NormalizedLandmarkList& landmarks,
                  const proto_ns::RepeatedField<int>& indexes) {
  double z_sum = 0;
  for (int i = 0; i < indexes.size(); ++i) {
    z_sum += landmarks.landmark(indexes.Get(i)).z();
  }
  return z_sum / indexes.size();
}

void RefineZ(
    const proto_ns::RepeatedField<int>& indexes_mapping,
    const LandmarksRefinementCalculatorOptions::ZRefinement& z_refinement,
    const NormalizedLandmarkList& landmarks,
    NormalizedLandmarkList* refined_landmarks) {
  if (z_refinement.has_none()) {
    // Do nothing and keep Z that is already in refined landmarks.
  } else if (z_refinement.has_copy()) {
    for (int i = 0; i < landmarks.landmark_size(); ++i) {
      refined_landmarks->mutable_landmark(indexes_mapping.Get(i))
          ->set_z(landmarks.landmark(i).z());
    }
  } else if (z_refinement.has_assign_average()) {
    const float z_average =
        GetZAverage(*refined_landmarks,
                    z_refinement.assign_average().indexes_for_average());
    for (int i = 0; i < indexes_mapping.size(); ++i) {
      refined_landmarks->mutable_landmark(indexes_mapping.Get(i))
          ->set_z(z_average);
    }
  } else {
    ABSL_CHECK(false)
        << "Z refinement is either not specified or not supported";
  }
}

}  // namespace

class LandmarksRefinementCalculatorImpl
    : public NodeImpl<LandmarksRefinementCalculator> {
  absl::Status Open(CalculatorContext* cc) override {
    options_ = cc->Options<LandmarksRefinementCalculatorOptions>();

    // Validate refinements.
    for (int i = 0; i < options_.refinement_size(); ++i) {
      const auto& refinement = options_.refinement(i);
      RET_CHECK_GT(refinement.indexes_mapping_size(), 0)
          << "Refinement " << i << " has no indexes mapping";
      RET_CHECK(refinement.has_z_refinement())
          << "Refinement " << i << " has no Z refinement specified";
      RET_CHECK(refinement.z_refinement().has_none() ^
                refinement.z_refinement().has_copy() ^
                refinement.z_refinement().has_assign_average())
          << "Exactly one Z refinement should be specified";

      const auto z_refinement = refinement.z_refinement();
      if (z_refinement.has_assign_average()) {
        RET_CHECK_GT(z_refinement.assign_average().indexes_for_average_size(),
                     0)
            << "When using assign average Z refinement at least one index for "
               "averagin should be specified";
      }
    }

    // Validate indexes mapping and get total number of refined landmarks.
    MP_ASSIGN_OR_RETURN(n_refined_landmarks_,
                        GetNumberOfRefinedLandmarks(options_.refinement()));

    // Validate that number of refinements and landmark streams is the same.
    RET_CHECK_EQ(kLandmarks(cc).Count(), options_.refinement_size())
        << "There are " << options_.refinement_size() << " refinements while "
        << kLandmarks(cc).Count() << " landmark streams";

    return absl::OkStatus();
  }

  absl::Status Process(CalculatorContext* cc) override {
    // If any of the refinement landmarks is missing - refinement won't happen.
    for (const auto& landmarks_stream : kLandmarks(cc)) {
      if (landmarks_stream.IsEmpty()) {
        return absl::OkStatus();
      }
    }

    // Initialize refined landmarks list.
    auto refined_landmarks = absl::make_unique<NormalizedLandmarkList>();
    for (int i = 0; i < n_refined_landmarks_; ++i) {
      refined_landmarks->add_landmark();
    }

    // Apply input landmarks to outpu refined landmarks in provided order.
    for (int i = 0; i < kLandmarks(cc).Count(); ++i) {
      const auto& landmarks = kLandmarks(cc)[i].Get();
      const auto& refinement = options_.refinement(i);

      // Check number of landmarks in mapping and stream are the same.
      RET_CHECK_EQ(landmarks.landmark_size(), refinement.indexes_mapping_size())
          << "There are " << landmarks.landmark_size()
          << " refinement landmarks while mapping has "
          << refinement.indexes_mapping_size();

      // Refine X and Y.
      RefineXY(refinement.indexes_mapping(), landmarks,
               refined_landmarks.get());

      // Refine Z.
      RefineZ(refinement.indexes_mapping(), refinement.z_refinement(),
              landmarks, refined_landmarks.get());

      // Visibility and presence are not currently refined and are left as `0`.
    }

    kRefinedLandmarks(cc).Send(std::move(refined_landmarks));
    return absl::OkStatus();
  }

 private:
  LandmarksRefinementCalculatorOptions options_;
  int n_refined_landmarks_ = 0;
};

MEDIAPIPE_NODE_IMPLEMENTATION(LandmarksRefinementCalculatorImpl);

}  // namespace api2
}  // namespace mediapipe
