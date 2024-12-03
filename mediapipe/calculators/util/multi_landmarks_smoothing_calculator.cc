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

#include "mediapipe/calculators/util/multi_landmarks_smoothing_calculator.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <vector>

#include "mediapipe/calculators/util/landmarks_smoothing_calculator.pb.h"
#include "mediapipe/calculators/util/landmarks_smoothing_calculator_utils.h"
#include "mediapipe/framework/api2/node.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/formats/landmark.pb.h"
#include "mediapipe/framework/formats/rect.pb.h"
#include "mediapipe/framework/port/ret_check.h"
#include "mediapipe/framework/timestamp.h"

namespace mediapipe {
namespace api2 {

namespace {

using ::mediapipe::NormalizedRect;
using ::mediapipe::landmarks_smoothing::GetObjectScale;
using ::mediapipe::landmarks_smoothing::LandmarksToNormalizedLandmarks;
using ::mediapipe::landmarks_smoothing::MultiLandmarkFilters;
using ::mediapipe::landmarks_smoothing::NormalizedLandmarksToLandmarks;

}  // namespace

class MultiLandmarksSmoothingCalculatorImpl
    : public NodeImpl<MultiLandmarksSmoothingCalculator> {
 public:
  absl::Status Process(CalculatorContext* cc) override {
    // Check that landmarks are not empty and reset the filter if so.
    // Don't emit an empty packet for this timestamp.
    if (kInNormLandmarks(cc).IsEmpty()) {
      multi_filters_.Clear();
      return absl::OkStatus();
    }

    const auto& timestamp =
        absl::Microseconds(cc->InputTimestamp().Microseconds());

    const auto& tracking_ids = kTrackingIds(cc).Get();
    multi_filters_.ClearUnused(tracking_ids);

    const auto& in_norm_landmarks_vec = kInNormLandmarks(cc).Get();
    RET_CHECK_EQ(in_norm_landmarks_vec.size(), tracking_ids.size());

    int image_width;
    int image_height;
    std::tie(image_width, image_height) = kImageSize(cc).Get();

    std::optional<std::vector<NormalizedRect>> object_scale_roi_vec;
    if (kObjectScaleRoi(cc).IsConnected() && !kObjectScaleRoi(cc).IsEmpty()) {
      object_scale_roi_vec = kObjectScaleRoi(cc).Get();
      RET_CHECK_EQ(object_scale_roi_vec.value().size(), tracking_ids.size());
    }

    std::vector<NormalizedLandmarkList> out_norm_landmarks_vec;
    for (int i = 0; i < tracking_ids.size(); ++i) {
      LandmarkList in_landmarks;
      NormalizedLandmarksToLandmarks(in_norm_landmarks_vec[i], image_width,
                                     image_height, in_landmarks);

      std::optional<float> object_scale;
      if (object_scale_roi_vec) {
        object_scale = GetObjectScale(object_scale_roi_vec.value()[i],
                                      image_width, image_height);
      }

      MP_ASSIGN_OR_RETURN(
          auto* landmarks_filter,
          multi_filters_.GetOrCreate(
              tracking_ids[i],
              cc->Options<LandmarksSmoothingCalculatorOptions>()));

      LandmarkList out_landmarks;
      MP_RETURN_IF_ERROR(landmarks_filter->Apply(in_landmarks, timestamp,
                                                 object_scale, out_landmarks));

      NormalizedLandmarkList out_norm_landmarks;
      LandmarksToNormalizedLandmarks(out_landmarks, image_width, image_height,
                                     out_norm_landmarks);

      out_norm_landmarks_vec.push_back(std::move(out_norm_landmarks));
    }

    kOutNormLandmarks(cc).Send(std::move(out_norm_landmarks_vec));

    return absl::OkStatus();
  }

 private:
  MultiLandmarkFilters multi_filters_;
};
MEDIAPIPE_NODE_IMPLEMENTATION(MultiLandmarksSmoothingCalculatorImpl);

}  // namespace api2
}  // namespace mediapipe
