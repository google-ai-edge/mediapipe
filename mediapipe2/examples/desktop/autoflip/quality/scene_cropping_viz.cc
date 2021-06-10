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

#include "mediapipe/examples/desktop/autoflip/quality/scene_cropping_viz.h"

#include <memory>

#include "absl/memory/memory.h"
#include "mediapipe/examples/desktop/autoflip/autoflip_messages.pb.h"
#include "mediapipe/examples/desktop/autoflip/quality/cropping.pb.h"
#include "mediapipe/examples/desktop/autoflip/quality/focus_point.pb.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/formats/image_format.pb.h"
#include "mediapipe/framework/formats/image_frame.h"
#include "mediapipe/framework/formats/image_frame_opencv.h"
#include "mediapipe/framework/port/opencv_core_inc.h"
#include "mediapipe/framework/port/opencv_imgproc_inc.h"
#include "mediapipe/framework/port/ret_check.h"
#include "mediapipe/framework/port/status.h"

namespace mediapipe {
namespace autoflip {

// Colors for focus signal sources.
const cv::Scalar kCyan =
    cv::Scalar(0.0, 255.0, 255.0);  // brain object detector
const cv::Scalar kMagenta = cv::Scalar(255.0, 0.0, 255.0);        // motion
const cv::Scalar kYellow = cv::Scalar(255.0, 255.0, 0.0);         // fg ocr
const cv::Scalar kLightYellow = cv::Scalar(255.0, 250.0, 205.0);  // bg ocr
const cv::Scalar kRed = cv::Scalar(255.0, 0.0, 0.0);              // logo
const cv::Scalar kGreen = cv::Scalar(0.0, 255.0, 0.0);            // face
const cv::Scalar kBlue =
    cv::Scalar(0.0, 0.0, 255.0);  // creatism saliency model
const cv::Scalar kOrange =
    cv::Scalar(255.0, 165.0, 0.0);  // ica object detector
const cv::Scalar kWhite = cv::Scalar(255.0, 255.0, 255.0);  // others

absl::Status DrawDetectionsAndCropRegions(
    const std::vector<cv::Mat>& scene_frames,
    const std::vector<bool>& is_key_frames,
    const std::vector<KeyFrameInfo>& key_frame_infos,
    const std::vector<KeyFrameCropResult>& key_frame_crop_results,
    const ImageFormat::Format image_format,
    std::vector<std::unique_ptr<ImageFrame>>* viz_frames) {
  RET_CHECK(viz_frames) << "Output viz frames is null.";
  viz_frames->clear();
  const int num_frames = scene_frames.size();

  std::pair<cv::Point, cv::Point> crop_corners;
  std::vector<std::pair<cv::Point, cv::Point>> region_corners;
  std::vector<cv::Scalar> region_colors;
  auto RectToCvPoints =
      [](const Rect& rect) -> std::pair<cv::Point, cv::Point> {
    return std::make_pair(
        cv::Point(rect.x(), rect.y()),
        cv::Point(rect.x() + rect.width(), rect.y() + rect.height()));
  };

  int key_frame_idx = 0;
  for (int i = 0; i < num_frames; ++i) {
    const auto& scene_frame = scene_frames[i];
    auto viz_frame = absl::make_unique<ImageFrame>(
        image_format, scene_frame.cols, scene_frame.rows);
    cv::Mat viz_mat = formats::MatView(viz_frame.get());
    scene_frame.copyTo(viz_mat);

    if (is_key_frames[i]) {
      const auto& bbox = key_frame_crop_results[key_frame_idx].region();
      crop_corners = RectToCvPoints(bbox);
      region_corners.clear();
      region_colors.clear();
      const auto& detections = key_frame_infos[key_frame_idx].detections();
      for (int j = 0; j < detections.detections_size(); ++j) {
        const auto& detection = detections.detections(j);
        const auto corners = RectToCvPoints(detection.location());
        region_corners.push_back(corners);
        if (detection.signal_type().has_standard()) {
          switch (detection.signal_type().standard()) {
            case SignalType::FACE_FULL:
            case SignalType::FACE_LANDMARK:
            case SignalType::FACE_ALL_LANDMARKS:
            case SignalType::FACE_CORE_LANDMARKS:
              region_colors.push_back(kGreen);
              break;
            case SignalType::HUMAN:
              region_colors.push_back(kLightYellow);
              break;
            case SignalType::CAR:
              region_colors.push_back(kMagenta);
              break;
            case SignalType::PET:
              region_colors.push_back(kYellow);
              break;
            case SignalType::OBJECT:
              region_colors.push_back(kCyan);
              break;
            case SignalType::MOTION:
            case SignalType::TEXT:
            case SignalType::LOGO:
              region_colors.push_back(kRed);
              break;
            case SignalType::USER_HINT:
            default:
              region_colors.push_back(kWhite);
              break;
          }
        } else {
          // For the case where "custom" signal type is used.
          region_colors.push_back(kWhite);
        }
      }
      key_frame_idx++;
    }

    cv::rectangle(viz_mat, crop_corners.first, crop_corners.second, kGreen, 4);
    for (int j = 0; j < region_corners.size(); ++j) {
      cv::rectangle(viz_mat, region_corners[j].first, region_corners[j].second,
                    region_colors[j], 2);
    }
    viz_frames->push_back(std::move(viz_frame));
  }
  return absl::OkStatus();
}

namespace {
cv::Rect LimitBounds(const cv::Rect& rect, const int max_width,
                     const int max_height) {
  cv::Rect result;
  result.x = fmax(rect.x, 0);
  result.y = fmax(rect.y, 0);
  result.width =
      result.x + rect.width >= max_width ? max_width - result.x : rect.width;
  result.height = result.y + rect.height >= max_height ? max_height - result.y
                                                       : rect.height;
  return result;
}
}  // namespace

absl::Status DrawDetectionAndFramingWindow(
    const std::vector<cv::Mat>& org_scene_frames,
    const std::vector<cv::Rect>& crop_from_locations,
    const ImageFormat::Format image_format, const float overlay_opacity,
    std::vector<std::unique_ptr<ImageFrame>>* viz_frames) {
  for (int i = 0; i < org_scene_frames.size(); i++) {
    const auto& scene_frame = org_scene_frames[i];
    auto viz_frame = absl::make_unique<ImageFrame>(
        image_format, scene_frame.cols, scene_frame.rows);
    cv::Mat darkened = formats::MatView(viz_frame.get());
    scene_frame.copyTo(darkened);
    cv::Mat overlay = cv::Mat::zeros(darkened.size(), darkened.type());
    cv::addWeighted(overlay, overlay_opacity, darkened, 1 - overlay_opacity, 0,
                    darkened);
    const auto& crop_from_bounded =
        LimitBounds(crop_from_locations[i], scene_frame.cols, scene_frame.rows);
    scene_frame(crop_from_bounded).copyTo(darkened(crop_from_bounded));
    viz_frames->push_back(std::move(viz_frame));
  }
  return absl::OkStatus();
}

absl::Status DrawFocusPointAndCropWindow(
    const std::vector<cv::Mat>& scene_frames,
    const std::vector<FocusPointFrame>& focus_point_frames,
    const float overlay_opacity, const int crop_window_width,
    const int crop_window_height, const ImageFormat::Format image_format,
    std::vector<std::unique_ptr<ImageFrame>>* viz_frames) {
  RET_CHECK(viz_frames) << "Output viz frames is null.";
  viz_frames->clear();
  const int num_frames = scene_frames.size();
  RET_CHECK_GT(crop_window_width, 0) << "Crop window width is non-positive.";
  RET_CHECK_GT(crop_window_height, 0) << "Crop window height is non-positive.";
  const int half_width = crop_window_width / 2;
  const int half_height = crop_window_height / 2;

  for (int i = 0; i < num_frames; ++i) {
    const auto& scene_frame = scene_frames[i];
    auto viz_frame = absl::make_unique<ImageFrame>(
        image_format, scene_frame.cols, scene_frame.rows);
    cv::Mat darkened = formats::MatView(viz_frame.get());
    scene_frame.copyTo(darkened);
    cv::Mat viz_mat = darkened.clone();

    // Darken the background.
    cv::Mat overlay = cv::Mat::zeros(darkened.size(), darkened.type());
    cv::addWeighted(overlay, overlay_opacity, darkened, 1 - overlay_opacity, 0,
                    darkened);

    if (focus_point_frames[i].point_size() > 0) {
      float center_x = 0.0f, center_y = 0.0f;
      for (int j = 0; j < focus_point_frames[i].point_size(); ++j) {
        const auto& point = focus_point_frames[i].point(j);
        const int x = point.norm_point_x() * scene_frame.cols;
        const int y = point.norm_point_y() * scene_frame.rows;
        cv::circle(viz_mat, cv::Point(x, y), 3, kRed, cv::FILLED);
        center_x += x;
        center_y += y;
      }
      center_x /= focus_point_frames[i].point_size();
      center_y /= focus_point_frames[i].point_size();
      cv::Point min_corner(center_x - half_width, center_y - half_height);
      cv::Point max_corner(center_x + half_width, center_y + half_height);
      viz_mat(cv::Rect(min_corner, max_corner))
          .copyTo(darkened(cv::Rect(min_corner, max_corner)));
    }
    viz_frames->push_back(std::move(viz_frame));
  }
  return absl::OkStatus();
}

}  // namespace autoflip
}  // namespace mediapipe
