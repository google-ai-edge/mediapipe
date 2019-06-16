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

#include "mediapipe/util/tensor_to_detection.h"

#include "absl/strings/str_format.h"
#include "absl/types/variant.h"
#include "mediapipe/framework/formats/location.h"
#include "mediapipe/framework/port/canonical_errors.h"
#include "mediapipe/framework/port/map_util.h"
#include "mediapipe/framework/port/status.h"
#include "tensorflow/core/framework/tensor_types.h"

namespace mediapipe {

using ::absl::StrFormat;

namespace tf = ::tensorflow;

Detection TensorToDetection(
    float box_ymin, float box_xmin, float box_ymax, float box_xmax,
    const float score, const absl::variant<int, std::string>& class_label) {
  Detection detection;
  detection.add_score(score);

  // According to mediapipe/framework/formats/detection.proto
  // "Either std::string or integer labels must be used but not both at the
  // same time."
  if (absl::holds_alternative<int>(class_label)) {
    detection.add_label_id(absl::get<int>(class_label));
  } else {
    detection.add_label(absl::get<std::string>(class_label));
  }

  LocationData* location_data = detection.mutable_location_data();
  location_data->set_format(LocationData::RELATIVE_BOUNDING_BOX);

  LocationData::RelativeBoundingBox* relative_bbox =
      location_data->mutable_relative_bounding_box();

  relative_bbox->set_xmin(box_xmin);
  relative_bbox->set_ymin(box_ymin);
  relative_bbox->set_width(box_xmax - box_xmin);
  relative_bbox->set_height(box_ymax - box_ymin);
  return detection;
}

Status TensorsToDetections(const ::tensorflow::Tensor& num_detections,
                           const ::tensorflow::Tensor& boxes,
                           const ::tensorflow::Tensor& scores,
                           const ::tensorflow::Tensor& classes,
                           const std::map<int, std::string>& label_map,
                           std::vector<Detection>* detections) {
  const ::tensorflow::Tensor empty_keypoints = ::tensorflow::Tensor(
      ::tensorflow::DT_FLOAT, ::tensorflow::TensorShape({0, 0, 0}));
  const ::tensorflow::Tensor empty_masks = ::tensorflow::Tensor(
      ::tensorflow::DT_FLOAT, ::tensorflow::TensorShape({0, 0, 0}));
  return TensorsToDetections(num_detections, boxes, scores, classes,
                             empty_keypoints, empty_masks,
                             /*mask_threshold=*/0.0f, label_map, detections);
}

Status TensorsToDetections(const ::tensorflow::Tensor& num_detections,
                           const ::tensorflow::Tensor& boxes,
                           const ::tensorflow::Tensor& scores,
                           const ::tensorflow::Tensor& classes,
                           const ::tensorflow::Tensor& keypoints,
                           const ::tensorflow::Tensor& masks,
                           float mask_threshold,
                           const std::map<int, std::string>& label_map,
                           std::vector<Detection>* detections) {
  int num_boxes = -1;
  if (num_detections.dims() > 0 && num_detections.dim_size(0) != 0) {
    if (num_detections.dtype() != tf::DT_INT32) {
      const auto& num_boxes_scalar = num_detections.scalar<float>();
      num_boxes = static_cast<int>(num_boxes_scalar());
    } else {
      num_boxes = num_detections.scalar<int32>()();
    }
    if (boxes.dim_size(0) < num_boxes) {
      return InvalidArgumentError(
          "First dimension of boxes tensor must be at least num_boxes");
    }

    if (classes.dim_size(0) != 0 && classes.dim_size(0) < num_boxes) {
      return InvalidArgumentError(
          "First dimension of classes tensor must be at least num_boxes");
    }

  } else {
    // If num_detections is not present, the number of boxes is determined by
    // the first dimension of the box tensor.
    if (boxes.dim_size(0) <= 0) {
      return InvalidArgumentError("Box tensor is empty");
    }
    num_boxes = boxes.dim_size(0);
  }

  if (scores.dim_size(0) < num_boxes) {
    return InvalidArgumentError(
        "First dimension of scores tensor must be at least num_boxes");
  }
  if (keypoints.dim_size(0) != 0 && keypoints.dim_size(0) < num_boxes) {
    return InvalidArgumentError(
        "First dimension of keypoint tensors must be at least num_boxes");
  }
  int num_keypoints = keypoints.dim_size(1);

  if (masks.dim_size(0) != 0 && masks.dim_size(0) < num_boxes) {
    return InvalidArgumentError(
        "First dimension of the masks tensor should be at least num_boxes");
  }

  const auto& score_vec =
      scores.dims() > 1 ? scores.flat<float>() : scores.vec<float>();
  const auto& classes_vec = classes.vec<float>();
  const auto& boxes_mat = boxes.tensor<float, 2>();
  const auto& keypoints_mat = keypoints.tensor<float, 3>();
  const auto& masks_mat = masks.tensor<float, 3>();

  for (int i = 0; i < num_boxes; ++i) {
    int class_id = -1;
    float score = -std::numeric_limits<float>::max();
    if (classes.dim_size(0) == 0) {
      // If class tensor is missing, we will sort the scores of all classes for
      // each box and keep the top.
      if (scores.dims() != 2) {
        return InvalidArgumentError(
            "Score tensor must have 2 dimensions where the last dimension has "
            "the scores for each class");
      }
      const int num_class = scores.dim_size(1);
      // Find the top score for box i.
      for (int score_idx = 0; score_idx < num_class; ++score_idx) {
        const auto score_for_class = score_vec(i * num_class + score_idx);
        if (score < score_for_class) {
          score = score_for_class;
          class_id = score_idx;
        }
      }
    } else {
      // If class tensor and score tensor are both present, we use the them
      // directly.
      if (scores.dims() != 1) {
        return InvalidArgumentError("Score tensor has more than 1 dimensions");
      }
      score = score_vec(i);
      class_id = static_cast<int>(classes_vec(i));
    }

    // Boxes is a tensor with shape (num_boxes x 4).
    // We extract the (1 x 4) slice corresponding to the i-th box and flatten
    // it into a vector of length 4.
    Detection detection;
    if (label_map.empty()) {
      detection =
          TensorToDetection(boxes_mat(i, 0), boxes_mat(i, 1), boxes_mat(i, 2),
                            boxes_mat(i, 3), score, class_id);
    } else {
      if (!::mediapipe::ContainsKey(label_map, class_id)) {
        return InvalidArgumentError(StrFormat(
            "Input label_map does not contain entry for integer label: %d",
            class_id));
      }
      detection = TensorToDetection(
          boxes_mat(i, 0), boxes_mat(i, 1), boxes_mat(i, 2), boxes_mat(i, 3),
          score, ::mediapipe::FindOrDie(label_map, class_id));
    }
    // Adding keypoints
    LocationData* location_data = detection.mutable_location_data();
    for (int j = 0; j < num_keypoints; ++j) {
      auto* keypoint = location_data->add_relative_keypoints();
      keypoint->set_y(keypoints_mat(i, j, 0));
      keypoint->set_x(keypoints_mat(i, j, 1));
    }
    // Adding masks
    if (masks.dim_size(0) != 0) {
      cv::Mat mask_image(cv::Size(masks.dim_size(2), masks.dim_size(1)),
                         CV_32FC1);
      for (int h = 0; h < masks.dim_size(1); ++h) {
        for (int w = 0; w < masks.dim_size(2); ++w) {
          const float value = masks_mat(i, h, w);
          mask_image.at<float>(h, w) = value > mask_threshold ? value : 0.0f;
        }
      }
      LocationData mask_location_data;
      mediapipe::Location::CreateCvMaskLocation<float>(mask_image)
          .ConvertToProto(&mask_location_data);
      location_data->MergeFrom(mask_location_data);
    }
    detections->emplace_back(detection);
  }
  return ::mediapipe::OkStatus();
}

}  // namespace mediapipe
