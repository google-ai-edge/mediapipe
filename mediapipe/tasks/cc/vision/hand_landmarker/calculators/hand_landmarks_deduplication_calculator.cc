/* Copyright 2022 The MediaPipe Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/
#include "mediapipe/tasks/cc/vision/hand_landmarker/calculators/hand_landmarks_deduplication_calculator.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_set.h"
#include "absl/memory/memory.h"
#include "absl/status/statusor.h"
#include "absl/types/optional.h"
#include "mediapipe/framework/api2/builder.h"
#include "mediapipe/framework/api2/node.h"
#include "mediapipe/framework/api2/port.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/formats/classification.pb.h"
#include "mediapipe/framework/formats/landmark.pb.h"
#include "mediapipe/framework/formats/rect.pb.h"
#include "mediapipe/tasks/cc/components/containers/rect.h"
#include "mediapipe/tasks/cc/vision/utils/landmarks_duplicates_finder.h"
#include "mediapipe/tasks/cc/vision/utils/landmarks_utils.h"

namespace mediapipe::api2 {
namespace {

using ::mediapipe::NormalizedRect;
using ::mediapipe::api2::Input;
using ::mediapipe::api2::Output;
using ::mediapipe::api2::builder::Source;
using ::mediapipe::tasks::components::containers::RectF;
using ::mediapipe::tasks::vision::utils::CalculateIOU;
using ::mediapipe::tasks::vision::utils::DuplicatesFinder;

float Distance(const NormalizedLandmark& lm_a, const NormalizedLandmark& lm_b,
               int width, int height) {
  return std::sqrt(std::pow((lm_a.x() - lm_b.x()) * width, 2) +
                   std::pow((lm_a.y() - lm_b.y()) * height, 2));
}

absl::StatusOr<std::vector<float>> Distances(const NormalizedLandmarkList& a,
                                             const NormalizedLandmarkList& b,
                                             int width, int height) {
  const int num = a.landmark_size();
  RET_CHECK_EQ(b.landmark_size(), num);
  std::vector<float> distances;
  distances.reserve(num);
  for (int i = 0; i < num; ++i) {
    const NormalizedLandmark& lm_a = a.landmark(i);
    const NormalizedLandmark& lm_b = b.landmark(i);
    distances.push_back(Distance(lm_a, lm_b, width, height));
  }
  return distances;
}

// Calculates a baseline distance of a hand that can be used as a relative
// measure when calculating hand to hand similarity.
//
// Calculated as maximum of distances: 0->5, 5->17, 17->0, where 0, 5, 17 key
// points are depicted below:
//
//               /Middle/
//                  |
//        /Index/   |    /Ring/
//           |      |      |   /Pinky/
//           V      V      V      |
//                                V
//          [8]   [12]   [16]
//           |      |      |    [20]
//           |      |      |      |
// /Thumb/   |      |      |      |
//    |     [7]   [11]   [15]   [19]
//    V      |      |      |      |
//           |      |      |      |
//   [4]     |      |      |      |
//    |     [6]   [10]   [14]   [18]
//    |      |      |      |      |
//    |      |      |      |      |
//   [3]     |      |      |      |
//    |     [5]----[9]---[13]---[17]
//    .      |                    |
//     \     .                    |
//      \   /                     |
//       [2]                      |
//         \                      |
//          \                     |
//           \                    |
//           [1]                  .
//             \                 /
//              \               /
//               ._____[0]_____.
//
//                      ^
//                      |
//                   /Wrist/
absl::StatusOr<float> HandBaselineDistance(
    const NormalizedLandmarkList& landmarks, int width, int height) {
  RET_CHECK_EQ(landmarks.landmark_size(), 21);  // Num of hand landmarks.
  constexpr int kWrist = 0;
  constexpr int kIndexFingerMcp = 5;
  constexpr int kPinkyMcp = 17;
  float distance = Distance(landmarks.landmark(kWrist),
                            landmarks.landmark(kIndexFingerMcp), width, height);
  distance = std::max(distance,
                      Distance(landmarks.landmark(kIndexFingerMcp),
                               landmarks.landmark(kPinkyMcp), width, height));
  distance =
      std::max(distance, Distance(landmarks.landmark(kPinkyMcp),
                                  landmarks.landmark(kWrist), width, height));
  return distance;
}

RectF CalculateBound(const NormalizedLandmarkList& list) {
  constexpr float kMinInitialValue = std::numeric_limits<float>::max();
  constexpr float kMaxInitialValue = std::numeric_limits<float>::lowest();

  // Compute min and max values on landmarks (they will form
  // bounding box)
  float bounding_box_left = kMinInitialValue;
  float bounding_box_top = kMinInitialValue;
  float bounding_box_right = kMaxInitialValue;
  float bounding_box_bottom = kMaxInitialValue;
  for (const auto& landmark : list.landmark()) {
    bounding_box_left = std::min(bounding_box_left, landmark.x());
    bounding_box_top = std::min(bounding_box_top, landmark.y());
    bounding_box_right = std::max(bounding_box_right, landmark.x());
    bounding_box_bottom = std::max(bounding_box_bottom, landmark.y());
  }

  // Populate normalized non rotated face bounding box
  return RectF{/*left=*/bounding_box_left,
               /*top=*/bounding_box_top,
               /*right=*/bounding_box_right,
               /*bottom=*/bounding_box_bottom};
}

// Uses IoU and distance of some corresponding hand landmarks to detect
// duplicate / similar hands. IoU, distance thresholds, number of landmarks to
// match are found experimentally. Evaluated:
// - manually comparing side by side, before and after deduplication applied
// - generating gesture dataset, and checking select frames in baseline and
//   "deduplicated" dataset
// - by confirming gesture training is better with use of deduplication using
//   selected thresholds
class HandDuplicatesFinder : public DuplicatesFinder {
 public:
  explicit HandDuplicatesFinder(bool start_from_the_end)
      : start_from_the_end_(start_from_the_end) {}

  absl::StatusOr<absl::flat_hash_set<int>> FindDuplicates(
      const std::vector<NormalizedLandmarkList>& multi_landmarks,
      int input_width, int input_height) override {
    absl::flat_hash_set<int> retained_indices;
    absl::flat_hash_set<int> suppressed_indices;

    const int num = multi_landmarks.size();
    std::vector<float> baseline_distances;
    baseline_distances.reserve(num);
    std::vector<RectF> bounds;
    bounds.reserve(num);
    for (const NormalizedLandmarkList& list : multi_landmarks) {
      MP_ASSIGN_OR_RETURN(
          const float baseline_distance,
          HandBaselineDistance(list, input_width, input_height));
      baseline_distances.push_back(baseline_distance);
      bounds.push_back(CalculateBound(list));
    }

    for (int index = 0; index < num; ++index) {
      const int i = start_from_the_end_ ? num - index - 1 : index;
      const float stable_distance_i = baseline_distances[i];
      bool suppressed = false;
      for (int j : retained_indices) {
        const float stable_distance_j = baseline_distances[j];

        constexpr float kAllowedBaselineDistanceRatio = 0.2f;
        const float distance_threshold =
            std::max(stable_distance_i, stable_distance_j) *
            kAllowedBaselineDistanceRatio;

        MP_ASSIGN_OR_RETURN(const std::vector<float> distances,
                            Distances(multi_landmarks[i], multi_landmarks[j],
                                      input_width, input_height));
        const int num_matched_landmarks = absl::c_count_if(
            distances,
            [&](float distance) { return distance < distance_threshold; });

        const float iou = CalculateIOU(bounds[i], bounds[j]);

        constexpr int kNumMatchedLandmarksToSuppressHand = 10;  // out of 21
        constexpr float kMinIouThresholdToSuppressHand = 0.2f;
        if (num_matched_landmarks >= kNumMatchedLandmarksToSuppressHand &&
            iou > kMinIouThresholdToSuppressHand) {
          suppressed = true;
          break;
        }
      }

      if (suppressed) {
        suppressed_indices.insert(i);
      } else {
        retained_indices.insert(i);
      }
    }
    return suppressed_indices;
  }

 private:
  const bool start_from_the_end_;
};

template <typename InputPortT>
absl::StatusOr<absl::optional<typename InputPortT::PayloadT>>
VerifyNumAndMaybeInitOutput(const InputPortT& port, CalculatorContext* cc,
                            int num_expected_size) {
  absl::optional<typename InputPortT::PayloadT> output;
  if (port(cc).IsConnected() && !port(cc).IsEmpty()) {
    RET_CHECK_EQ(port(cc).Get().size(), num_expected_size);
    typename InputPortT::PayloadT result;
    return {{result}};
  }
  return {absl::nullopt};
}
}  // namespace

std::unique_ptr<DuplicatesFinder> CreateHandDuplicatesFinder(
    bool start_from_the_end) {
  return absl::make_unique<HandDuplicatesFinder>(start_from_the_end);
}

absl::Status HandLandmarksDeduplicationCalculator::Process(
    mediapipe::CalculatorContext* cc) {
  if (kInLandmarks(cc).IsEmpty()) return absl::OkStatus();
  if (kInSize(cc).IsEmpty()) return absl::OkStatus();

  const std::vector<NormalizedLandmarkList>& in_landmarks = *kInLandmarks(cc);
  const std::pair<int, int>& image_size = *kInSize(cc);

  std::unique_ptr<DuplicatesFinder> duplicates_finder =
      CreateHandDuplicatesFinder(/*start_from_the_end=*/false);
  MP_ASSIGN_OR_RETURN(absl::flat_hash_set<int> indices_to_remove,
                      duplicates_finder->FindDuplicates(
                          in_landmarks, image_size.first, image_size.second));

  if (indices_to_remove.empty()) {
    kOutLandmarks(cc).Send(kInLandmarks(cc));
    kOutRois(cc).Send(kInRois(cc));
    kOutWorldLandmarks(cc).Send(kInWorldLandmarks(cc));
    kOutClassifications(cc).Send(kInClassifications(cc));
  } else {
    std::vector<NormalizedLandmarkList> out_landmarks;
    const int num = in_landmarks.size();

    MP_ASSIGN_OR_RETURN(absl::optional<std::vector<NormalizedRect>> out_rois,
                        VerifyNumAndMaybeInitOutput(kInRois, cc, num));
    MP_ASSIGN_OR_RETURN(
        absl::optional<std::vector<LandmarkList>> out_world_landmarks,
        VerifyNumAndMaybeInitOutput(kInWorldLandmarks, cc, num));
    MP_ASSIGN_OR_RETURN(
        absl::optional<std::vector<ClassificationList>> out_classifications,
        VerifyNumAndMaybeInitOutput(kInClassifications, cc, num));

    for (int i = 0; i < num; ++i) {
      if (indices_to_remove.find(i) != indices_to_remove.end()) continue;

      out_landmarks.push_back(in_landmarks[i]);
      if (out_rois) {
        out_rois->push_back(kInRois(cc).Get()[i]);
      }
      if (out_world_landmarks) {
        out_world_landmarks->push_back(kInWorldLandmarks(cc).Get()[i]);
      }
      if (out_classifications) {
        out_classifications->push_back(kInClassifications(cc).Get()[i]);
      }
    }

    if (!out_landmarks.empty()) {
      kOutLandmarks(cc).Send(std::move(out_landmarks));
    }
    if (out_rois && !out_rois->empty()) {
      kOutRois(cc).Send(std::move(out_rois.value()));
    }
    if (out_world_landmarks && !out_world_landmarks->empty()) {
      kOutWorldLandmarks(cc).Send(std::move(out_world_landmarks.value()));
    }
    if (out_classifications && !out_classifications->empty()) {
      kOutClassifications(cc).Send(std::move(out_classifications.value()));
    }
  }
  return absl::OkStatus();
}
MEDIAPIPE_REGISTER_NODE(HandLandmarksDeduplicationCalculator);

}  // namespace mediapipe::api2
