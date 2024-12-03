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

#include "mediapipe/framework/formats/location.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <memory>

#include "absl/log/absl_check.h"
#include "absl/log/absl_log.h"
#include "absl/memory/memory.h"
#include "absl/strings/substitute.h"
#include "mediapipe/framework/formats/annotation/locus.pb.h"
#include "mediapipe/framework/formats/annotation/rasterization.pb.h"
#include "mediapipe/framework/port/canonical_errors.h"
#include "mediapipe/framework/port/point2.h"
#include "mediapipe/framework/port/ret_check.h"
#include "mediapipe/framework/port/status.h"
#include "mediapipe/framework/port/statusor.h"
#include "mediapipe/framework/tool/status_util.h"
#include "mediapipe/framework/type_map.h"

namespace mediapipe {

namespace {
// Extracts from the BinaryMask, stored as mediapipe.Rasterization in
// the location_data, the tightest bounding box, that contains all pixels
// encoded in the rasterizations.
Rectangle_i MaskToRectangle(const LocationData& location_data) {
  ABSL_CHECK(location_data.mask().has_rasterization());
  const auto& rasterization = location_data.mask().rasterization();
  if (rasterization.interval_size() == 0) {
    return Rectangle_i(0, 0, 0, 0);
  }
  int xmin = std::numeric_limits<int>::max();
  int xmax = std::numeric_limits<int>::min();
  int ymin = std::numeric_limits<int>::max();
  int ymax = std::numeric_limits<int>::min();
  for (const auto& interval : rasterization.interval()) {
    xmin = std::min(xmin, interval.left_x());
    xmax = std::max(xmax, interval.right_x());
    ymin = std::min(ymin, interval.y());
    ymax = std::max(ymax, interval.y());
  }
  return Rectangle_i(xmin, ymin, xmax - xmin + 1, ymax - ymin + 1);
}

}  // namespace

Location::Location() {}

Location::Location(const LocationData& location_data)
    : location_data_(location_data) {
  ABSL_CHECK(IsValidLocationData(location_data_));
}

Location Location::CreateGlobalLocation() {
  LocationData location_data;
  location_data.set_format(LocationData::GLOBAL);
  return Location(location_data);
}

Location Location::CreateBBoxLocation(int xmin, int ymin, int width,
                                      int height) {
  LocationData location_data;
  location_data.set_format(LocationData::BOUNDING_BOX);
  auto* bounding_box = location_data.mutable_bounding_box();
  bounding_box->set_xmin(xmin);
  bounding_box->set_ymin(ymin);
  bounding_box->set_width(width);
  bounding_box->set_height(height);
  return Location(location_data);
}

Location Location::CreateBBoxLocation(const Rectangle_i& rect) {
  return CreateBBoxLocation(rect.xmin(), rect.ymin(), rect.Width(),
                            rect.Height());
}

Location Location::CreateBBoxLocation(const ::mediapipe::BoundingBox& bbox) {
  return CreateBBoxLocation(bbox.left_x(), bbox.upper_y(),
                            bbox.right_x() - bbox.left_x(),
                            bbox.lower_y() - bbox.upper_y());
}

Location Location::CreateRelativeBBoxLocation(float relative_xmin,
                                              float relative_ymin,
                                              float relative_width,
                                              float relative_height) {
  LocationData location_data;
  location_data.set_format(LocationData::RELATIVE_BOUNDING_BOX);
  auto* bounding_box = location_data.mutable_relative_bounding_box();
  bounding_box->set_xmin(relative_xmin);
  bounding_box->set_ymin(relative_ymin);
  bounding_box->set_width(relative_width);
  bounding_box->set_height(relative_height);
  return Location(location_data);
}

Location Location::CreateRelativeBBoxLocation(const Rectangle_f& rect) {
  return CreateRelativeBBoxLocation(rect.xmin(), rect.ymin(), rect.Width(),
                                    rect.Height());
}

LocationData::Format Location::GetFormat() const {
  return location_data_.format();
}

// static
bool Location::IsValidLocationData(const LocationData& location_data) {
  switch (location_data.format()) {
    case LocationData::GLOBAL: {
      // Nothing to check for global location data.
      return true;
    }
    case LocationData::BOUNDING_BOX: {
      return location_data.has_bounding_box() &&
             location_data.bounding_box().has_xmin() &&
             location_data.bounding_box().has_ymin() &&
             location_data.bounding_box().has_width() &&
             location_data.bounding_box().has_height();
    }
    case LocationData::RELATIVE_BOUNDING_BOX: {
      return location_data.has_relative_bounding_box() &&
             location_data.relative_bounding_box().has_xmin() &&
             location_data.relative_bounding_box().has_ymin() &&
             location_data.relative_bounding_box().has_width() &&
             location_data.relative_bounding_box().has_height();
    }
    case LocationData::MASK: {
      return location_data.has_mask() && location_data.mask().has_width() &&
             location_data.mask().has_height() &&
             location_data.mask().has_rasterization();
    }
    default: {
      return false;
    }
  }
}

template <>
Rectangle_i Location::GetBBox<Rectangle_i>() const {
  ABSL_CHECK_EQ(LocationData::BOUNDING_BOX, location_data_.format());
  const auto& box = location_data_.bounding_box();
  return Rectangle_i(box.xmin(), box.ymin(), box.width(), box.height());
}

Location& Location::Scale(const float scale) {
  ABSL_CHECK(!location_data_.has_mask())
      << "Location mask scaling is not implemented.";
  ABSL_CHECK_GT(scale, 0.0f);
  switch (location_data_.format()) {
    case LocationData::GLOBAL: {
      // Do nothing.
      break;
    }
    case LocationData::BOUNDING_BOX: {
      auto* bb = location_data_.mutable_bounding_box();
      bb->set_xmin(scale * bb->xmin());
      bb->set_ymin(scale * bb->ymin());
      bb->set_width(scale * bb->width());
      bb->set_height(scale * bb->height());
      break;
    }
    case LocationData::RELATIVE_BOUNDING_BOX: {
      auto* bb = location_data_.mutable_relative_bounding_box();
      bb->set_xmin(scale * bb->xmin());
      bb->set_ymin(scale * bb->ymin());
      bb->set_width(scale * bb->width());
      bb->set_height(scale * bb->height());
      for (auto& keypoint : *location_data_.mutable_relative_keypoints()) {
        keypoint.set_x(scale * keypoint.x());
        keypoint.set_y(scale * keypoint.y());
      }
      break;
    }
    case LocationData::MASK: {
      ABSL_LOG(FATAL)
          << "Scaling for location data of type MASK is not supported.";
      break;
    }
  }
  return *this;
}

Location& Location::Square(int image_width, int image_height) {
  switch (location_data_.format()) {
    case LocationData::GLOBAL: {
      // Do nothing.
      break;
    }
    case LocationData::BOUNDING_BOX: {
      auto* box = location_data_.mutable_bounding_box();
      const int max_dimension = std::max(box->width(), box->height());
      if (max_dimension > box->width()) {
        box->set_xmin(box->xmin() + box->width() / 2 - max_dimension / 2);
        box->set_width(max_dimension);
      } else if (max_dimension > box->height()) {
        box->set_ymin(box->ymin() + box->height() / 2 - max_dimension / 2);
        box->set_height(max_dimension);
      }
      break;
    }
    case LocationData::RELATIVE_BOUNDING_BOX: {
      auto* box = location_data_.mutable_relative_bounding_box();
      const float absolute_xmin = box->xmin() * image_width;
      const float absolute_ymin = box->ymin() * image_height;
      const float absolute_width = box->width() * image_width;
      const float absolute_height = box->height() * image_height;
      const float max_dimension = std::max(absolute_width, absolute_height);
      if (max_dimension > absolute_width) {
        box->set_xmin((absolute_xmin + absolute_width / 2 - max_dimension / 2) /
                      image_width);
        box->set_width(max_dimension / image_width);
      } else if (max_dimension > absolute_height) {
        box->set_ymin(
            (absolute_ymin + absolute_height / 2 - max_dimension / 2) /
            image_height);
        box->set_height(max_dimension / image_height);
      }
      break;
    }
    case LocationData::MASK: {
      ABSL_LOG(FATAL)
          << "Squaring for location data of type MASK is not supported.";
      break;
    }
  }
  return *this;
}

namespace {

// Finds an optimal shift t such that I = [min_value + t, max_value + t) will be
// included in the interval J = [0, range) if possible. If the above is not
// possible, then interval I will be centered at the center of interval J.
// This function is inteded to shift boundaries of intervals such that they
// best fit within an image.
float BestShift(float min_value, float max_value, float range) {
  ABSL_CHECK_LE(min_value, max_value);
  const float value_range = max_value - min_value;
  if (value_range > range) {
    return 0.5f * (range - min_value - max_value);
  } else {
    if (min_value < 0.0f) {
      return -min_value;
    } else if (max_value > range) {
      return range - max_value;
    }
  }
  return 0.0f;
}

}  // namespace

Location& Location::ShiftToFitBestIntoImage(int image_width, int image_height) {
  switch (location_data_.format()) {
    case LocationData::GLOBAL: {
      // Do nothing.
      break;
    }
    case LocationData::BOUNDING_BOX: {
      auto* box = location_data_.mutable_bounding_box();
      box->set_xmin(static_cast<int>(std::round(
          box->xmin() +
          BestShift(box->xmin(), box->xmin() + box->width(), image_width))));
      box->set_ymin(static_cast<int>(std::round(
          box->ymin() +
          BestShift(box->ymin(), box->ymin() + box->height(), image_height))));
      break;
    }
    case LocationData::RELATIVE_BOUNDING_BOX: {
      auto* box = location_data_.mutable_relative_bounding_box();
      box->set_xmin(box->xmin() +
                    BestShift(box->xmin(), box->xmin() + box->width(), 1.0f));
      box->set_ymin(box->ymin() +
                    BestShift(box->ymin(), box->ymin() + box->height(), 1.0f));
      break;
    }
    case LocationData::MASK: {
      auto mask_bounding_box = MaskToRectangle(location_data_);
      const float x_shift = BestShift(mask_bounding_box.xmin(),
                                      mask_bounding_box.xmax(), image_width);
      const float y_shift = BestShift(mask_bounding_box.xmin(),
                                      mask_bounding_box.xmax(), image_height);
      auto* mask = location_data_.mutable_mask();
      ABSL_CHECK_EQ(image_width, mask->width());
      ABSL_CHECK_EQ(image_height, mask->height());
      for (auto& interval :
           *mask->mutable_rasterization()->mutable_interval()) {
        interval.set_y(interval.y() + y_shift);
        interval.set_left_x(interval.left_x() + x_shift);
        interval.set_right_x(interval.right_x() + x_shift);
      }
      break;
    }
  }
  return *this;
}

Location& Location::Crop(const Rectangle_i& crop_box) {
  switch (location_data_.format()) {
    case LocationData::GLOBAL: {
      // Do nothing.
      break;
    }
    case LocationData::BOUNDING_BOX: {
      auto* box = location_data_.mutable_bounding_box();
      const int xmin = std::max(box->xmin(), crop_box.xmin());
      const int ymin = std::max(box->ymin(), crop_box.ymin());
      const int xmax = std::min(box->width() + box->xmin(), crop_box.xmax());
      const int ymax = std::min(box->height() + box->ymin(), crop_box.ymax());
      box->set_xmin(xmin - crop_box.xmin());
      box->set_ymin(ymin - crop_box.ymin());
      box->set_width(xmax - xmin);
      box->set_height(ymax - ymin);
      break;
    }
    case LocationData::RELATIVE_BOUNDING_BOX:
      ABSL_LOG(FATAL)
          << "Can't crop a relative bounding box using absolute coordinates. "
             "Use the 'Rectangle_f version of Crop() instead";
    case LocationData::MASK: {
      LocationData::BinaryMask new_mask;
      new_mask.set_width(crop_box.Width());
      new_mask.set_height(crop_box.Height());
      auto* rasterization = new_mask.mutable_rasterization();
      for (const auto& interval :
           location_data_.mask().rasterization().interval()) {
        if (interval.y() >= crop_box.ymin() && interval.y() < crop_box.ymax() &&
            interval.left_x() < crop_box.xmax() &&
            interval.right_x() > crop_box.xmin()) {
          auto* new_interval = rasterization->add_interval();
          new_interval->set_y(interval.y() - crop_box.ymin());
          new_interval->set_left_x(
              std::max(interval.left_x() - crop_box.xmin(), 0));
          new_interval->set_right_x(
              std::min(interval.right_x() - crop_box.xmin(), crop_box.Width()));
        }
      }
      location_data_.mutable_mask()->Swap(&new_mask);
      break;
    }
  }
  return *this;
}

Location& Location::Crop(const Rectangle_f& crop_box) {
  switch (location_data_.format()) {
    case LocationData::GLOBAL:
      // Do nothing.
      break;
    case LocationData::BOUNDING_BOX:
      ABSL_LOG(FATAL)
          << "Can't crop an absolute bounding box using relative coordinates. "
             "Use the 'Rectangle_i version of Crop() instead";
    case LocationData::RELATIVE_BOUNDING_BOX: {
      auto* box = location_data_.mutable_relative_bounding_box();
      float right = box->xmin() + box->width();
      float bottom = box->ymin() + box->height();
      box->set_xmin(std::max(crop_box.xmin(), box->xmin()));
      box->set_ymin(std::max(crop_box.ymin(), box->ymin()));
      float newRight = std::min(crop_box.xmax(), right);
      float newBottom = std::min(crop_box.ymax(), bottom);
      box->set_width(newRight - box->xmin());
      box->set_height(newBottom - box->ymin());
      break;
    }
    case LocationData::MASK:
      ABSL_LOG(FATAL)
          << "Can't crop a mask using relative coordinates. Use the "
             "'Rectangle_i' version of Crop() instead";
  }
  return *this;
}

template <>
Rectangle_i Location::ConvertToBBox<Rectangle_i>(int image_width,
                                                 int image_height) const {
  switch (location_data_.format()) {
    case LocationData::GLOBAL: {
      return Rectangle_i(0, 0, image_width, image_height);
    }
    case LocationData::BOUNDING_BOX: {
      const auto& box = location_data_.bounding_box();
      return Rectangle_i(box.xmin(), box.ymin(), box.width(), box.height());
    }
    case LocationData::RELATIVE_BOUNDING_BOX: {
      const auto& box = location_data_.relative_bounding_box();
      // Taking the floor rather than rounding for the width and height ensures
      // that if the original relative bounding box was within the image bounds,
      // the absolute bounding box that it is converted to will also be
      // within the image bounds.
      return Rectangle_i(
          static_cast<int>(std::round(image_width * box.xmin())),
          static_cast<int>(std::round(image_height * box.ymin())),
          static_cast<int>(image_width * box.width()),
          static_cast<int>(image_height * box.height()));
    }
    case LocationData::MASK: {
      return MaskToRectangle(location_data_);
    }
    default: {
      // Note that this shouldn't happen, but a default clause is required by
      // the Android compiler.
      return Rectangle_i();
    }
  }
}

Rectangle_f Location::GetRelativeBBox() const {
  ABSL_CHECK_EQ(LocationData::RELATIVE_BOUNDING_BOX, location_data_.format());
  const auto& box = location_data_.relative_bounding_box();
  return Rectangle_f(box.xmin(), box.ymin(), box.width(), box.height());
}

Rectangle_f Location::ConvertToRelativeBBox(int image_width,
                                            int image_height) const {
  switch (location_data_.format()) {
    case LocationData::GLOBAL: {
      return Rectangle_f(0.0f, 0.0f, 1.0f, 1.0f);
    }
    case LocationData::BOUNDING_BOX: {
      const auto& box = location_data_.bounding_box();
      return Rectangle_f(static_cast<float>(box.xmin()) / image_width,
                         static_cast<float>(box.ymin()) / image_height,
                         static_cast<float>(box.width()) / image_width,
                         static_cast<float>(box.height()) / image_height);
    }
    case LocationData::RELATIVE_BOUNDING_BOX: {
      const auto& box = location_data_.relative_bounding_box();
      return Rectangle_f(box.xmin(), box.ymin(), box.width(), box.height());
    }
    case LocationData::MASK: {
      auto rect = MaskToRectangle(location_data_);
      return Rectangle_f(static_cast<float>(rect.xmin()) / image_width,
                         static_cast<float>(rect.ymin()) / image_height,
                         static_cast<float>(rect.Width()) / image_width,
                         static_cast<float>(rect.Height()) / image_height);
    }
    default: {
      // Note that this shouldn't happen, but a default clause is required by
      // the Android compiler.
      return Rectangle_f();
    }
  }
}

template <>
::mediapipe::BoundingBox Location::GetBBox<::mediapipe::BoundingBox>() const {
  ABSL_CHECK_EQ(LocationData::BOUNDING_BOX, location_data_.format());
  const auto& box = location_data_.bounding_box();
  ::mediapipe::BoundingBox bounding_box;
  bounding_box.set_left_x(box.xmin());
  bounding_box.set_upper_y(box.ymin());
  bounding_box.set_right_x(box.width() + box.xmin());
  bounding_box.set_lower_y(box.height() + box.ymin());
  return bounding_box;
}

template <>
::mediapipe::BoundingBox Location::ConvertToBBox<::mediapipe::BoundingBox>(
    int image_width, int image_height) const {
  const auto& rect = ConvertToBBox<Rectangle_i>(image_width, image_height);
  ::mediapipe::BoundingBox bounding_box;
  bounding_box.set_left_x(rect.xmin());
  bounding_box.set_upper_y(rect.ymin());
  bounding_box.set_right_x(rect.xmax());
  bounding_box.set_lower_y(rect.ymax());
  return bounding_box;
}

std::vector<Point2_f> Location::GetRelativeKeypoints() const {
  ABSL_CHECK_EQ(LocationData::RELATIVE_BOUNDING_BOX, location_data_.format());
  std::vector<Point2_f> keypoints;
  for (const auto& keypoint : location_data_.relative_keypoints()) {
    keypoints.emplace_back(Point2_f(keypoint.x(), keypoint.y()));
  }
  return keypoints;
}

std::vector<Point2_i> Location::ConvertToKeypoints(int image_width,
                                                   int image_height) const {
  std::vector<Point2_i> keypoints;
  for (const auto& keypoint : location_data_.relative_keypoints()) {
    keypoints.emplace_back(
        Point2_i(static_cast<int>(std::round(image_width * keypoint.x())),
                 static_cast<int>(std::round(image_height * keypoint.y()))));
  }
  return keypoints;
}

void Location::SetRelativeKeypoints(const std::vector<Point2_f>& keypoints) {
  location_data_.clear_relative_keypoints();
  for (const auto& keypoint : keypoints) {
    auto* relative_keypoint = location_data_.add_relative_keypoints();
    relative_keypoint->set_x(keypoint.x());
    relative_keypoint->set_y(keypoint.y());
  }
}

void Location::SetFromProto(const LocationData& proto) {
  location_data_ = proto;
}

void Location::ConvertToProto(LocationData* proto) const {
  *proto = location_data_;
}

LocationData Location::ConvertToProto() const {
  LocationData location_data;
  ConvertToProto(&location_data);
  return location_data;
}

}  // namespace mediapipe
