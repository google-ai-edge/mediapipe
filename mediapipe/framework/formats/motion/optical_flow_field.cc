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

#include "mediapipe/framework/formats/motion/optical_flow_field.h"

#include <math.h>

#include <cmath>

#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "mediapipe/framework/deps/mathutil.h"
#include "mediapipe/framework/formats/location.h"
#include "mediapipe/framework/port/file_helpers.h"
#include "mediapipe/framework/port/integral_types.h"
#include "mediapipe/framework/port/logging.h"
#include "mediapipe/framework/port/point2.h"
#include "mediapipe/framework/port/ret_check.h"
#include "mediapipe/framework/type_map.h"

namespace {

const float kHugeToIgnore = 1e9;

// File tags defined in Middlebury specifications to check little endian floats
const char kFloFileHeaderOnWrite[] = "PIEH";
const float kFloFileHeaderOnRead = 202021.25;

void CartesianToPolarCoordinates(const cv::Mat& cartesian, cv::Mat* magnitudes,
                                 cv::Mat* angles) {
  CHECK(magnitudes != nullptr);
  CHECK(angles != nullptr);
  cv::Mat cartesian_components[2];
  cv::split(cartesian, cartesian_components);
  cv::cartToPolar(cartesian_components[0], cartesian_components[1], *magnitudes,
                  *angles, true);
}

float MaxAbsoluteValueIgnoringHuge(const cv::Mat_<float>& values, float huge) {
  float max_val = 0.0f;
  for (int r = 0; r < values.rows; ++r) {
    for (int c = 0; c < values.cols; ++c) {
      const float abs_val = std::abs(values(r, c));
      if (abs_val < huge && max_val < abs_val) {
        max_val = abs_val;
      }
    }
  }
  return max_val;
}

cv::Mat MakeVisualizationHsv(const cv::Mat_<float>& angles,
                             const cv::Mat_<float>& magnitudes, float max_mag) {
  cv::Mat hsv(angles.size(), CV_8UC3);
  for (int r = 0; r < hsv.rows; ++r) {
    for (int c = 0; c < hsv.cols; ++c) {
      const uint8 hue = static_cast<uint8>(255.0f * angles(r, c) / 360.0f);
      uint8 saturation = 255;
      if (magnitudes(r, c) < max_mag) {
        saturation = static_cast<uint8>(255.0f * magnitudes(r, c) / max_mag);
      }
      const uint8 value = 255;

      hsv.at<cv::Vec3b>(r, c) = cv::Vec3b(hue, saturation, value);
    }
  }
  return hsv;
}

}  // namespace

namespace mediapipe {

OpticalFlowField::OpticalFlowField(const cv::Mat_<cv::Point2f>& flow) {
  flow.copyTo(flow_data_);
}

float OpticalFlowField::GetRobustMaximumMagnitude() const {
  cv::Mat angles;
  cv::Mat magnitudes;
  CartesianToPolarCoordinates(flow_data_, &magnitudes, &angles);
  return MaxAbsoluteValueIgnoringHuge(magnitudes, kHugeToIgnore);
}

cv::Mat OpticalFlowField::GetVisualizationInternal(
    float max_magnitude, bool enforce_max_magnitude) const {
  cv::Mat angles;
  cv::Mat magnitudes;
  CartesianToPolarCoordinates(flow_data_, &magnitudes, &angles);
  if (!enforce_max_magnitude) {
    // Guard against dividing by zero for the case of an all-zero flow field.
    max_magnitude =
        std::max(std::numeric_limits<float>::epsilon(),
                 MaxAbsoluteValueIgnoringHuge(magnitudes, kHugeToIgnore));
  }
  CHECK_LT(0, max_magnitude);
  cv::Mat hsv = MakeVisualizationHsv(angles, magnitudes, max_magnitude);
  cv::Mat viz;
  cv::cvtColor(hsv, viz, 71 /*cv::COLOR_HSV2RGB_FULL*/);
  return viz;
}

cv::Mat OpticalFlowField::GetVisualization() const {
  // Dummy value of 1.0 for max_magnitude will be replaced.
  return GetVisualizationInternal(1.0f, false);
}

cv::Mat OpticalFlowField::GetVisualizationSaturatedAt(
    float max_magnitude) const {
  CHECK_LT(0, max_magnitude)
      << "Specified saturation magnitude must be positive.";
  return GetVisualizationInternal(max_magnitude, true);
}

void OpticalFlowField::Allocate(int width, int height) {
  flow_data_.create(height, width);
}

void OpticalFlowField::Resize(int new_width, int new_height) {
  if (new_width == flow_data_.cols && new_height == flow_data_.rows) {
    return;
  }
  cv::Mat source_for_resize = flow_data_;
  float width_scale = new_width / static_cast<float>(source_for_resize.cols);
  float height_scale = new_height / static_cast<float>(source_for_resize.rows);
  cv::resize(source_for_resize, flow_data_, cv::Size(new_width, new_height), 0,
             0, cv::INTER_LINEAR);
  for (int r = 0; r < new_height; ++r) {
    for (int c = 0; c < new_width; ++c) {
      cv::Point2f flow_vector = flow_data_.at<cv::Point2f>(r, c);
      flow_data_.at<cv::Point2f>(r, c) = cv::Point2f(
          flow_vector.x * width_scale, flow_vector.y * height_scale);
    }
  }
}

void OpticalFlowField::CopyFromTensor(const tensorflow::Tensor& tensor) {
  CHECK_EQ(tensorflow::DT_FLOAT, tensor.dtype());
  CHECK_EQ(3, tensor.dims()) << "Tensor must be height x width x 2.";
  CHECK_EQ(2, tensor.dim_size(2)) << "Tensor must be height x width x 2.";
  const int height = tensor.dim_size(0);
  const int width = tensor.dim_size(1);
  Allocate(width, height);
  typename tensorflow::TTypes<float, 3>::ConstTensor input_flow =
      tensor.shaped<float, 3>({height, width, 2});
  for (int r = 0; r < height; ++r) {
    for (int c = 0; c < width; ++c) {
      flow_data_(r, c) = cv::Point2f(input_flow(r, c, 0), input_flow(r, c, 1));
    }
  }
}

void OpticalFlowField::SetFromProto(const OpticalFlowFieldData& proto) {
  CHECK_EQ(proto.width() * proto.height(), proto.dx_size());
  CHECK_EQ(proto.width() * proto.height(), proto.dy_size());
  flow_data_.create(proto.height(), proto.width());
  int i = 0;
  for (int r = 0; r < flow_data_.rows; ++r) {
    for (int c = 0; c < flow_data_.cols; ++c, ++i) {
      flow_data_(r, c) = cv::Point2f(proto.dx(i),   // x component
                                     proto.dy(i));  // y component
    }
  }
}

void OpticalFlowField::ConvertToProto(OpticalFlowFieldData* proto) const {
  proto->set_width(width());
  proto->set_height(height());
  proto->clear_dx();
  proto->clear_dy();

  for (int r = 0; r < flow_data_.rows; ++r) {
    for (int c = 0; c < flow_data_.cols; ++c) {
      proto->add_dx(flow_data_(r, c).x);
      proto->add_dy(flow_data_(r, c).y);
    }
  }
}

bool OpticalFlowField::FollowFlow(float x, float y, float* new_x,
                                  float* new_y) const {
  CHECK(new_x);
  CHECK(new_y);
  if (x < 0 || x > flow_data_.cols - 1 ||  // horizontal bounds
      y < 0 || y > flow_data_.rows - 1) {  // vertical bounds
    return false;
  }
  const cv::Point2f flow_vector = InterpolatedFlowAt(x, y);
  *new_x = x + flow_vector.x;
  *new_y = y + flow_vector.y;
  return true;
}

cv::Point2f OpticalFlowField::InterpolatedFlowAt(float x, float y) const {
  // Sanity bounds checks.
  CHECK_GE(x, 0);
  CHECK_GE(y, 0);
  CHECK_LE(x, flow_data_.cols - 1);
  CHECK_LE(y, flow_data_.rows - 1);

  const int x0 = static_cast<int>(std::floor(x));
  const int y0 = static_cast<int>(std::floor(y));
  // Make sure we don't try to access out of bounds pixels in the case where no
  // interpolation is needed (e.g., because x == width - 1).
  int x1 = x0 < flow_data_.cols - 1 ? x0 + 1 : x0;
  int y1 = y0 < flow_data_.rows - 1 ? y0 + 1 : y0;

  const cv::Point2f flow_top_left = flow_data_(y0, x0);
  const cv::Point2f flow_top_right = flow_data_(y0, x1);
  const cv::Point2f flow_bottom_left = flow_data_(y1, x0);
  const cv::Point2f flow_bottom_right = flow_data_(y1, x1);
  // Linearly interpolate horizontally first.
  const cv::Point2f flow_top =
      flow_top_left + (x - x0) * (flow_top_right - flow_top_left);
  const cv::Point2f flow_bottom =
      flow_bottom_left + (x - x0) * (flow_bottom_right - flow_bottom_left);
  // Linear interpolation vertically.
  return flow_top + (y - y0) * (flow_bottom - flow_top);
}

cv::Mat OpticalFlowField::ConvertToCorrespondences() const {
  // Initialize with (dx, dy).
  cv::Mat_<cv::Point2f> correspondences = flow_data_.clone();
  // Add (x, y) to each location.
  for (int y = 0; y < correspondences.rows; ++y) {
    for (int x = 0; x < correspondences.cols; ++x) {
      correspondences(y, x) += cv::Point2f(x, y);
    }
  }
  return correspondences;
}

bool OpticalFlowField::AllWithinMargin(const OpticalFlowField& other,
                                       float margin) const {
  if (other.width() != width() || other.height() != height()) {
    return false;
  }
  for (int r = 0; r < flow_data_.rows; ++r) {
    for (int c = 0; c < flow_data_.cols; ++c) {
      const cv::Point2f& this_motion = flow_data_.at<cv::Point2f>(r, c);
      const cv::Point2f& other_motion = other.flow_data().at<cv::Point2f>(r, c);
      if (!MathUtil::WithinMargin(this_motion.x, other_motion.x, margin) ||
          !MathUtil::WithinMargin(this_motion.y, other_motion.y, margin)) {
        LOG(INFO) << "First failure at" << r << " " << c;
        return false;
      }
    }
  }
  return true;
}

void OpticalFlowField::EstimateMotionConsistencyOcclusions(
    const OpticalFlowField& forward, const OpticalFlowField& backward,
    double spatial_distance_threshold, Location* occluded_mask,
    Location* disoccluded_mask) {
  CHECK_EQ(forward.width(), backward.width())
      << "Flow fields have different widths.";
  CHECK_EQ(forward.height(), backward.height())
      << "Flow fields have different heights.";
  if (occluded_mask != nullptr) {
    *occluded_mask = FindMotionInconsistentPixels(forward, backward,
                                                  spatial_distance_threshold);
  }
  if (disoccluded_mask != nullptr) {
    *disoccluded_mask = FindMotionInconsistentPixels(
        backward, forward, spatial_distance_threshold);
  }
}

Location OpticalFlowField::FindMotionInconsistentPixels(
    const OpticalFlowField& forward, const OpticalFlowField& backward,
    double spatial_distance_threshold) {
  const uint8 kOccludedPixelValue = 1;
  const double threshold_sq =
      spatial_distance_threshold * spatial_distance_threshold;
  cv::Mat occluded = cv::Mat::zeros(forward.height(), forward.width(), CV_8UC1);
  for (int x = 0; x < forward.width(); ++x) {
    for (int y = 0; y < forward.height(); ++y) {
      // Location of the point in the next frame.
      float new_x;
      float new_y;
      // Location of the point in this frame after a round-trip to the next
      // frame and back.
      float round_trip_x;
      float round_trip_y;
      forward.FollowFlow(x, y, &new_x, &new_y);
      bool in_bounds_in_next_frame =
          backward.FollowFlow(new_x, new_y, &round_trip_x, &round_trip_y);
      if (!in_bounds_in_next_frame ||
          Point2_f(x - round_trip_x, y - round_trip_y).ToVector().Norm2() >
              threshold_sq) {
        occluded.at<uint8>(y, x) = kOccludedPixelValue;
      }
    }
  }
  return Location::CreateCvMaskLocation<uint8>(occluded);
}
}  // namespace mediapipe
