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

#include "mediapipe/calculators/util/refine_landmarks_from_heatmap_calculator.h"

#include "mediapipe/calculators/util/refine_landmarks_from_heatmap_calculator.pb.h"
#include "mediapipe/framework/calculator_framework.h"

namespace mediapipe {

namespace {

inline float Sigmoid(float value) { return 1.0f / (1.0f + std::exp(-value)); }

absl::StatusOr<std::tuple<int, int, int>> GetHwcFromDims(
    const std::vector<int>& dims) {
  if (dims.size() == 3) {
    return std::make_tuple(dims[0], dims[1], dims[2]);
  } else if (dims.size() == 4) {
    // BHWC format check B == 1
    RET_CHECK_EQ(1, dims[0]) << "Expected batch to be 1 for BHWC heatmap";
    return std::make_tuple(dims[1], dims[2], dims[3]);
  } else {
    RET_CHECK(false) << "Invalid shape size for heatmap tensor" << dims.size();
  }
}

}  // namespace

namespace api2 {

// Refines landmarks using correspond heatmap area.
//
// Input:
//   NORM_LANDMARKS - Required. Input normalized landmarks to update.
//   TENSORS - Required. Vector of input tensors. 0th element should be heatmap.
//             The rest is unused.
// Output:
//   NORM_LANDMARKS - Required. Updated normalized landmarks.
class RefineLandmarksFromHeatmapCalculatorImpl
    : public NodeImpl<RefineLandmarksFromHeatmapCalculator,
                      RefineLandmarksFromHeatmapCalculatorImpl> {
 public:
  absl::Status Open(CalculatorContext* cc) override { return absl::OkStatus(); }

  absl::Status Process(CalculatorContext* cc) override {
    // Make sure we bypass landmarks if there is no detection.
    if (kInLandmarks(cc).IsEmpty()) {
      return absl::OkStatus();
    }
    // If for some reason heatmap is missing, just return original landmarks.
    if (kInTensors(cc).IsEmpty()) {
      kOutLandmarks(cc).Send(*kInLandmarks(cc));
      return absl::OkStatus();
    }

    // Check basic prerequisites.
    const auto& input_tensors = *kInTensors(cc);
    RET_CHECK(!input_tensors.empty()) << "Empty input tensors list. First "
                                         "element is expeced to be a heatmap";

    const auto& hm_tensor = input_tensors[0];
    const auto& in_lms = *kInLandmarks(cc);
    auto hm_view = hm_tensor.GetCpuReadView();
    auto hm_raw = hm_view.buffer<float>();
    const auto& options =
        cc->Options<mediapipe::RefineLandmarksFromHeatmapCalculatorOptions>();

    ASSIGN_OR_RETURN(
        auto out_lms,
        RefineLandmarksFromHeatMap(
            in_lms, hm_raw, hm_tensor.shape().dims, options.kernel_size(),
            options.min_confidence_to_refine(), options.refine_presence(),
            options.refine_visibility()));

    kOutLandmarks(cc).Send(std::move(out_lms));
    return absl::OkStatus();
  }
};

}  // namespace api2

// Runs actual refinement
// High level algorithm:
//
// Heatmap is accepted as tensor in HWC layout where i-th channel is a heatmap
// for the i-th landmark.
//
// For each landmark we replace original value with a value calculated from the
// area in heatmap close to original landmark position (in particular are
// covered with kernel of size options.kernel_size). To calculate new coordinate
// from heatmap we calculate an weighted average inside the kernel. We update
// the landmark iff heatmap is confident in it's prediction i.e. max(heatmap) in
// kernel is at least options.min_confidence_to_refine big.
absl::StatusOr<mediapipe::NormalizedLandmarkList> RefineLandmarksFromHeatMap(
    const mediapipe::NormalizedLandmarkList& in_lms,
    const float* heatmap_raw_data, const std::vector<int>& heatmap_dims,
    int kernel_size, float min_confidence_to_refine, bool refine_presence,
    bool refine_visibility) {
  ASSIGN_OR_RETURN(auto hm_dims, GetHwcFromDims(heatmap_dims));
  auto [hm_height, hm_width, hm_channels] = hm_dims;

  RET_CHECK_EQ(in_lms.landmark_size(), hm_channels)
      << "Expected heatmap to have number of layers == to number of "
         "landmarks";

  int hm_row_size = hm_width * hm_channels;
  int hm_pixel_size = hm_channels;

  mediapipe::NormalizedLandmarkList out_lms = in_lms;
  for (int lm_index = 0; lm_index < out_lms.landmark_size(); ++lm_index) {
    int center_col = out_lms.landmark(lm_index).x() * hm_width;
    int center_row = out_lms.landmark(lm_index).y() * hm_height;
    // Point is outside of the image let's keep it intact.
    if (center_col < 0 || center_col >= hm_width || center_row < 0 ||
        center_col >= hm_height) {
      continue;
    }

    int offset = (kernel_size - 1) / 2;
    // Calculate area to iterate over. Note that we decrease the kernel on
    // the edges of the heatmap. Equivalent to zero border.
    int begin_col = std::max(0, center_col - offset);
    int end_col = std::min(hm_width, center_col + offset + 1);
    int begin_row = std::max(0, center_row - offset);
    int end_row = std::min(hm_height, center_row + offset + 1);

    float sum = 0;
    float weighted_col = 0;
    float weighted_row = 0;
    float max_confidence_value = 0;

    // Main loop. Go over kernel and calculate weighted sum of coordinates,
    // sum of weights and max weights.
    for (int row = begin_row; row < end_row; ++row) {
      for (int col = begin_col; col < end_col; ++col) {
        // We expect memory to be in HWC layout without padding.
        int idx = hm_row_size * row + hm_pixel_size * col + lm_index;
        // Right now we hardcode sigmoid activation as it will be wasteful to
        // calculate sigmoid for each value of heatmap in the model itself.  If
        // we ever have other activations it should be trivial to expand via
        // options.
        float confidence = Sigmoid(heatmap_raw_data[idx]);
        sum += confidence;
        max_confidence_value = std::max(max_confidence_value, confidence);
        weighted_col += col * confidence;
        weighted_row += row * confidence;
      }
    }
    if (max_confidence_value >= min_confidence_to_refine && sum > 0) {
      out_lms.mutable_landmark(lm_index)->set_x(weighted_col / hm_width / sum);
      out_lms.mutable_landmark(lm_index)->set_y(weighted_row / hm_height / sum);
    }
    if (refine_presence && sum > 0 &&
        out_lms.landmark(lm_index).has_presence()) {
      // We assume confidence in heatmaps describes landmark presence.
      // If landmark is not confident in heatmaps, probably it is not present.
      const float presence = out_lms.landmark(lm_index).presence();
      const float new_presence = std::min(presence, max_confidence_value);
      out_lms.mutable_landmark(lm_index)->set_presence(new_presence);
    }
    if (refine_visibility && sum > 0 &&
        out_lms.landmark(lm_index).has_visibility()) {
      // We assume confidence in heatmaps describes landmark presence.
      // As visibility = (not occluded but still present) -> that mean that if
      // landmark is not present, it is not visible as well.
      // I.e. visibility confidence cannot be bigger than presence confidence.
      const float visibility = out_lms.landmark(lm_index).visibility();
      const float new_visibility = std::min(visibility, max_confidence_value);
      out_lms.mutable_landmark(lm_index)->set_visibility(new_visibility);
    }
  }
  return out_lms;
}

}  // namespace mediapipe
