// Copyright 2022 The MediaPipe Authors.
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

#include "mediapipe/framework/formats/location_opencv.h"

#include "absl/log/absl_check.h"
#include "absl/log/absl_log.h"
#include "absl/memory/memory.h"
#include "absl/strings/substitute.h"
#include "mediapipe/framework/formats/annotation/rasterization.pb.h"
#include "mediapipe/framework/formats/location.h"
#include "mediapipe/framework/port/opencv_imgproc_inc.h"
#include "mediapipe/framework/port/statusor.h"

namespace mediapipe {

namespace {
Rectangle_i MaskToRectangle(const LocationData& location_data) {
  ABSL_CHECK(location_data.mask().has_rasterization());
  const auto& rasterization = location_data.mask().rasterization();
  if (rasterization.interval_size() == 0) {
    return Rectangle_i(0, 0, 0, 0);
  }
  int xmin = std::numeric_limits<int>::max();
  int xmax = std::numeric_limits<int>::lowest();
  int ymin = std::numeric_limits<int>::max();
  int ymax = std::numeric_limits<int>::lowest();
  for (const auto& interval : rasterization.interval()) {
    xmin = std::min(xmin, interval.left_x());
    xmax = std::max(xmax, interval.right_x());
    ymin = std::min(ymin, interval.y());
    ymax = std::max(ymax, interval.y());
  }
  return Rectangle_i(xmin, ymin, xmax - xmin + 1, ymax - ymin + 1);
}

std::unique_ptr<cv::Mat> MaskToMat(const LocationData::BinaryMask& mask) {
  auto image = absl::make_unique<cv::Mat>();
  *image = cv::Mat::zeros(cv::Size(mask.width(), mask.height()), CV_32FC1);
  for (const auto& interval : mask.rasterization().interval()) {
    for (int x = interval.left_x(); x <= interval.right_x(); ++x) {
      image->at<float>(interval.y(), x) = 1.0f;
    }
  }
  return image;
}
absl::StatusOr<std::unique_ptr<cv::Mat>> RectangleToMat(
    int image_width, int image_height, const Rectangle_i& rect) {
  // These checks prevent undefined behavior caused when setting memory for
  // rectangles whose edges lie outside image edges.
  if (rect.ymin() < 0 || rect.xmin() < 0 || rect.xmax() > image_width ||
      rect.ymax() > image_height) {
    return absl::InvalidArgumentError(absl::Substitute(
        "Rectangle must be bounded by image boundaries.\nImage Width: "
        "$0\nImage Height: $1\nRectangle: [($2, $3), ($4, $5)]",
        image_width, image_height, rect.xmin(), rect.ymin(), rect.xmax(),
        rect.ymax()));
  }
  // Allocate image and set pixels of foreground mask.
  auto image = absl::make_unique<cv::Mat>();
  *image = cv::Mat::zeros(cv::Size(image_width, image_height), CV_32FC1);
  for (int y = rect.ymin(); y < rect.ymax(); ++y) {
    for (int x = rect.xmin(); x < rect.xmax(); ++x) {
      image->at<float>(y, x) = 1.0f;
    }
  }

  return std::move(image);
}
}  // namespace

Location CreateBBoxLocation(const cv::Rect& rect) {
  return Location::CreateBBoxLocation(rect.x, rect.y, rect.width, rect.height);
}

std::unique_ptr<cv::Mat> GetCvMask(const Location& location) {
  const auto location_data = location.ConvertToProto();
  ABSL_CHECK_EQ(LocationData::MASK, location_data.format());
  const auto& mask = location_data.mask();
  std::unique_ptr<cv::Mat> mat(
      new cv::Mat(mask.height(), mask.width(), CV_8UC1, cv::Scalar(0)));
  for (const auto& interval : location_data.mask().rasterization().interval()) {
    for (int x = interval.left_x(); x <= interval.right_x(); ++x) {
      mat->at<uint8_t>(interval.y(), x) = 255;
    }
  }
  return mat;
}

std::unique_ptr<cv::Mat> ConvertToCvMask(const Location& location,
                                         int image_width, int image_height) {
  const auto location_data = location.ConvertToProto();
  switch (location_data.format()) {
    case LocationData::GLOBAL:
    case LocationData::BOUNDING_BOX:
    case LocationData::RELATIVE_BOUNDING_BOX: {
      auto status_or_mat = RectangleToMat(
          image_width, image_height,
          location.ConvertToBBox<Rectangle_i>(image_width, image_height));
      if (!status_or_mat.ok()) {
        ABSL_LOG(ERROR) << status_or_mat.status().message();
        return nullptr;
      }
      return std::move(status_or_mat).value();
    }
    case LocationData::MASK: {
      return MaskToMat(location_data.mask());
    }
  }
// This should never happen; a new LocationData::Format enum was introduced
// without updating this function's switch(...) to support it.
#if !defined(MEDIAPIPE_MOBILE) && !defined(MEDIAPIPE_LITE)
  ABSL_LOG(ERROR) << "Location's LocationData has format not supported by "
                     "Location::ConvertToMask: "
                  << location_data.DebugString();
#endif
  return nullptr;
}

void EnlargeLocation(Location& location, const float factor) {
  ABSL_CHECK_GT(factor, 0.0f);
  if (factor == 1.0f) return;
  auto location_data = location.ConvertToProto();
  switch (location_data.format()) {
    case LocationData::GLOBAL: {
      // Do nothing.
      break;
    }
    case LocationData::BOUNDING_BOX: {
      auto* box = location_data.mutable_bounding_box();
      const int enlarged_int_width =
          static_cast<int>(std::round(factor * box->width()));
      const int enlarged_int_height =
          static_cast<int>(std::round(factor * box->height()));
      box->set_xmin(
          std::max(box->xmin() + box->width() / 2 - enlarged_int_width / 2, 0));
      box->set_ymin(std::max(
          box->ymin() + box->height() / 2 - enlarged_int_height / 2, 0));
      box->set_width(enlarged_int_width);
      box->set_height(enlarged_int_height);
      break;
    }
    case LocationData::RELATIVE_BOUNDING_BOX: {
      auto* box = location_data.mutable_relative_bounding_box();
      box->set_xmin(box->xmin() - ((factor - 1.0) * box->width()) / 2.0);
      box->set_ymin(box->ymin() - ((factor - 1.0) * box->height()) / 2.0);
      box->set_width(factor * box->width());
      box->set_height(factor * box->height());
      break;
    }
    case LocationData::MASK: {
      auto mask_bounding_box = MaskToRectangle(location_data);
      const float scaler = std::fabs(factor - 1.0f);
      const int dilation_width =
          static_cast<int>(std::round(scaler * mask_bounding_box.Width()));
      const int dilation_height =
          static_cast<int>(std::round(scaler * mask_bounding_box.Height()));
      if (dilation_width == 0 || dilation_height == 0) break;
      cv::Mat morph_element(dilation_height, dilation_width, CV_8U,
                            cv::Scalar(1));
      auto mask = GetCvMask(location);
      if (factor > 1.0f) {
        cv::dilate(*mask, *mask, morph_element);
      } else {
        cv::erode(*mask, *mask, morph_element);
      }
      CreateCvMaskLocation<uint8_t>(*mask).ConvertToProto(&location_data);
      break;
    }
  }
  location.SetFromProto(location_data);
}

template <typename T>
Location CreateCvMaskLocation(const cv::Mat_<T>& mask) {
  ABSL_CHECK_EQ(1, mask.channels())
      << "The specified cv::Mat mask should be single-channel.";

  LocationData location_data;
  location_data.set_format(LocationData::MASK);
  location_data.mutable_mask()->set_width(mask.cols);
  location_data.mutable_mask()->set_height(mask.rows);
  auto* rasterization = location_data.mutable_mask()->mutable_rasterization();
  const auto kForegroundThreshold = static_cast<T>(0);
  for (int y = 0; y < mask.rows; y++) {
    Rasterization::Interval* interval;
    bool traversing = false;
    for (int x = 0; x < mask.cols; x++) {
      const bool is_foreground =
          mask.template at<T>(y, x) > kForegroundThreshold;
      if (is_foreground) {
        if (!traversing) {
          interval = rasterization->add_interval();
          interval->set_y(y);
          interval->set_left_x(x);
          traversing = true;
        }
        interval->set_right_x(x);
      } else {
        traversing = false;
      }
    }
  }
  return Location(location_data);
}

template Location CreateCvMaskLocation(const cv::Mat_<uint8_t>& mask);
template Location CreateCvMaskLocation(const cv::Mat_<float>& mask);

}  // namespace mediapipe
