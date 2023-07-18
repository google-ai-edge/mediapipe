// Copyright (c) 2023 Intel Corporation
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
//

#include <unordered_map>
#include <vector>

#include "absl/strings/str_format.h"
#include "absl/types/span.h"
#include "mediapipe/calculators/openvino/openvino_tensors_to_detections_calculator.pb.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/deps/file_path.h"
#include "mediapipe/framework/formats/detection.pb.h"
#include "mediapipe/framework/formats/location.h"
#include "mediapipe/framework/formats/object_detection/anchor.pb.h"
#include "mediapipe/framework/port/ret_check.h"

#include <openvino/openvino.hpp>

namespace {
constexpr int kNumInputTensorsWithAnchors = 3;
constexpr int kNumCoordsPerBox = 4;

constexpr char kTensorsTag[] = "TENSORS";
}  // namespace

namespace mediapipe {

namespace {

void ConvertRawValuesToAnchors(const float* raw_anchors, int num_boxes,
                               std::vector<Anchor>* anchors) {
  anchors->clear();
  for (int i = 0; i < num_boxes; ++i) {
    Anchor new_anchor;
    new_anchor.set_y_center(raw_anchors[i * kNumCoordsPerBox + 0]);
    new_anchor.set_x_center(raw_anchors[i * kNumCoordsPerBox + 1]);
    new_anchor.set_h(raw_anchors[i * kNumCoordsPerBox + 2]);
    new_anchor.set_w(raw_anchors[i * kNumCoordsPerBox + 3]);
    anchors->push_back(new_anchor);
  }
}

void ConvertAnchorsToRawValues(const std::vector<Anchor>& anchors,
                               int num_boxes, float* raw_anchors) {
  CHECK_EQ(anchors.size(), num_boxes);
  int box = 0;
  for (const auto& anchor : anchors) {
    raw_anchors[box * kNumCoordsPerBox + 0] = anchor.y_center();
    raw_anchors[box * kNumCoordsPerBox + 1] = anchor.x_center();
    raw_anchors[box * kNumCoordsPerBox + 2] = anchor.h();
    raw_anchors[box * kNumCoordsPerBox + 3] = anchor.w();
    ++box;
  }
}

}  // namespace

// Convert result TFLite tensors from object detection models into MediaPipe
// Detections.
//
// Input:
//  TENSORS - Vector of ov::Tensorof type kOpenVINOFloat32. The vector of
//               tensors can have 2 or 3 tensors. First tensor is the predicted
//               raw boxes/keypoints. The size of the values must be (num_boxes
//               * num_predicted_values). Second tensor is the score tensor. The
//               size of the valuse must be (num_boxes * num_classes). It's
//               optional to pass in a third tensor for anchors (e.g. for SSD
//               models) depend on the outputs of the detection model. The size
//               of anchor tensor must be (num_boxes * 4).
//  TENSORS_GPU - vector of GlBuffer of MTLBuffer.
// Output:
//  DETECTIONS - Result MediaPipe detections.
//
// Usage example:
// node {
//   calculator: "OpenVINOTensorsToDetectionsCalculator"
//   input_stream: "TENSORS:tensors"
//   input_side_packet: "ANCHORS:anchors"
//   output_stream: "DETECTIONS:detections"
//   options: {
//     [mediapipe.OpenVINOTensorsToDetectionsCalculatorOptions.ext] {
//       num_classes: 91
//       num_boxes: 1917
//       num_coords: 4
//       ignore_classes: [0, 1, 2]
//       x_scale: 10.0
//       y_scale: 10.0
//       h_scale: 5.0
//       w_scale: 5.0
//     }
//   }
// }
class OpenVINOTensorsToDetectionsCalculator : public CalculatorBase {
 public:
  static absl::Status GetContract(CalculatorContract* cc);

  absl::Status Open(CalculatorContext* cc) override;
  absl::Status Process(CalculatorContext* cc) override;
  absl::Status Close(CalculatorContext* cc) override;

 private:
  absl::Status ProcessCPU(CalculatorContext* cc,
                          std::vector<Detection>* output_detections);

  absl::Status LoadOptions(CalculatorContext* cc);
  absl::Status DecodeBoxes(const float* raw_boxes,
                           const std::vector<Anchor>& anchors,
                           std::vector<float>* boxes);
  absl::Status ConvertToDetections(const float* detection_boxes,
                                   const float* detection_scores,
                                   const int* detection_classes,
                                   std::vector<Detection>* output_detections);
  Detection ConvertToDetection(float box_ymin, float box_xmin, float box_ymax,
                               float box_xmax, float score, int class_id,
                               bool flip_vertically);

  int num_classes_ = 0;
  int num_boxes_ = 0;
  int num_coords_ = 0;
  std::set<int> ignore_classes_;

  ::mediapipe::OpenVINOTensorsToDetectionsCalculatorOptions options_;
  std::vector<Anchor> anchors_;
  bool side_packet_anchors_{};

  bool anchors_init_ = false;
};
REGISTER_CALCULATOR(OpenVINOTensorsToDetectionsCalculator);

absl::Status OpenVINOTensorsToDetectionsCalculator::GetContract(
    CalculatorContract* cc) {
  RET_CHECK(!cc->Inputs().GetTags().empty());
  RET_CHECK(!cc->Outputs().GetTags().empty());

  if (cc->Inputs().HasTag(kTensorsTag)) {
    cc->Inputs().Tag(kTensorsTag).Set<std::vector<ov::Tensor>>();
  }

  if (cc->Outputs().HasTag("DETECTIONS")) {
    cc->Outputs().Tag("DETECTIONS").Set<std::vector<Detection>>();
  }

  if (cc->InputSidePackets().UsesTags()) {
    if (cc->InputSidePackets().HasTag("ANCHORS")) {
      cc->InputSidePackets().Tag("ANCHORS").Set<std::vector<Anchor>>();
    }
  }

  return absl::OkStatus();
}

absl::Status OpenVINOTensorsToDetectionsCalculator::Open(CalculatorContext* cc) {
  cc->SetOffset(TimestampDiff(0));

  MP_RETURN_IF_ERROR(LoadOptions(cc));
  side_packet_anchors_ = cc->InputSidePackets().HasTag("ANCHORS");

  return absl::OkStatus();
}

absl::Status OpenVINOTensorsToDetectionsCalculator::Process(
    CalculatorContext* cc) {
  if (cc->Inputs().Tag(kTensorsTag).IsEmpty()) {
    return absl::OkStatus();
  }

  auto output_detections = absl::make_unique<std::vector<Detection>>();

  MP_RETURN_IF_ERROR(ProcessCPU(cc, output_detections.get()));

  // Output
  if (cc->Outputs().HasTag("DETECTIONS")) {
    cc->Outputs()
        .Tag("DETECTIONS")
        .Add(output_detections.release(), cc->InputTimestamp());
  }

  return absl::OkStatus();
}

absl::Status OpenVINOTensorsToDetectionsCalculator::ProcessCPU(
    CalculatorContext* cc, std::vector<Detection>* output_detections) {
  const auto& input_tensors =
      cc->Inputs().Tag(kTensorsTag).Get<std::vector<ov::Tensor>>();

  if (input_tensors.size() == 2 ||
      input_tensors.size() == kNumInputTensorsWithAnchors) {
    // Postprocessing on CPU for model without postprocessing op. E.g. output
    // raw score tensor and box tensor. Anchor decoding will be handled below.
    const ov::Tensor* raw_box_tensor = &input_tensors[0];
    const ov::Tensor* raw_score_tensor = &input_tensors[1];

    // TODO: Add flexible input tensor size handling.
    CHECK_EQ(raw_box_tensor->get_shape().size(), 3);
    CHECK_EQ(raw_box_tensor->get_shape()[0], 1);
    CHECK_EQ(raw_box_tensor->get_shape()[1], num_boxes_);
    CHECK_EQ(raw_box_tensor->get_shape()[2], num_coords_);
    CHECK_EQ(raw_score_tensor->get_shape().size(), 3);
    CHECK_EQ(raw_score_tensor->get_shape()[0], 1);
    CHECK_EQ(raw_score_tensor->get_shape()[1], num_boxes_);
    CHECK_EQ(raw_score_tensor->get_shape()[2], num_classes_);
    const float* raw_boxes = raw_box_tensor->data<float>();
    const float* raw_scores = raw_score_tensor->data<float>();

    // TODO: Support other options to load anchors.
    if (!anchors_init_) {
      if (input_tensors.size() == kNumInputTensorsWithAnchors) {
        const ov::Tensor* anchor_tensor = &input_tensors[2];
        CHECK_EQ(anchor_tensor->get_shape().size(), 2);
        CHECK_EQ(anchor_tensor->get_shape()[0], num_boxes_);
        CHECK_EQ(anchor_tensor->get_shape()[1], kNumCoordsPerBox);
        const float* raw_anchors = anchor_tensor->data<float>();
        ConvertRawValuesToAnchors(raw_anchors, num_boxes_, &anchors_);
      } else if (side_packet_anchors_) {
        CHECK(!cc->InputSidePackets().Tag("ANCHORS").IsEmpty());
        anchors_ =
            cc->InputSidePackets().Tag("ANCHORS").Get<std::vector<Anchor>>();
      } else {
        return absl::UnavailableError("No anchor data available.");
      }
      anchors_init_ = true;
    }
    std::vector<float> boxes(num_boxes_ * num_coords_);
    MP_RETURN_IF_ERROR(DecodeBoxes(raw_boxes, anchors_, &boxes));

    std::vector<float> detection_scores(num_boxes_);
    std::vector<int> detection_classes(num_boxes_);

    // Filter classes by scores.
    for (int i = 0; i < num_boxes_; ++i) {
      int class_id = -1;
      float max_score = -std::numeric_limits<float>::max();
      // Find the top score for box i.
      for (int score_idx = 0; score_idx < num_classes_; ++score_idx) {
        if (ignore_classes_.find(score_idx) == ignore_classes_.end()) {
          auto score = raw_scores[i * num_classes_ + score_idx];
          if (options_.sigmoid_score()) {
            if (options_.has_score_clipping_thresh()) {
              score = score < -options_.score_clipping_thresh()
                          ? -options_.score_clipping_thresh()
                          : score;
              score = score > options_.score_clipping_thresh()
                          ? options_.score_clipping_thresh()
                          : score;
            }
            score = 1.0f / (1.0f + std::exp(-score));
          }
          if (max_score < score) {
            max_score = score;
            class_id = score_idx;
          }
        }
      }
      detection_scores[i] = max_score;
      detection_classes[i] = class_id;
    }

    MP_RETURN_IF_ERROR(
        ConvertToDetections(boxes.data(), detection_scores.data(),
                            detection_classes.data(), output_detections));
  } else {
    // Postprocessing on CPU with postprocessing op (e.g. anchor decoding and
    // non-maximum suppression) within the model.
    RET_CHECK_EQ(input_tensors.size(), 4);

    const ov::Tensor* detection_boxes_tensor = &input_tensors[0];
    const ov::Tensor* detection_classes_tensor = &input_tensors[1];
    const ov::Tensor* detection_scores_tensor = &input_tensors[2];
    const ov::Tensor* num_boxes_tensor = &input_tensors[3];
    RET_CHECK_EQ(num_boxes_tensor->get_shape().size(), 1);
    RET_CHECK_EQ(num_boxes_tensor->get_shape()[0], 1);
    const float* num_boxes = num_boxes_tensor->data<float>();
    num_boxes_ = num_boxes[0];
    RET_CHECK_EQ(detection_boxes_tensor->get_shape().size(), 3);
    RET_CHECK_EQ(detection_boxes_tensor->get_shape()[0], 1);
    const int max_detections = detection_boxes_tensor->get_shape()[1];
    RET_CHECK_EQ(detection_boxes_tensor->get_shape()[2], num_coords_);
    RET_CHECK_EQ(detection_classes_tensor->get_shape().size(), 2);
    RET_CHECK_EQ(detection_classes_tensor->get_shape()[0], 1);
    RET_CHECK_EQ(detection_classes_tensor->get_shape()[1], max_detections);
    RET_CHECK_EQ(detection_scores_tensor->get_shape().size(), 2);
    RET_CHECK_EQ(detection_scores_tensor->get_shape()[0], 1);
    RET_CHECK_EQ(detection_scores_tensor->get_shape()[1], max_detections);

    const float* detection_boxes = detection_boxes_tensor->data<float>();
    const float* detection_scores = detection_scores_tensor->data<float>();
    std::vector<int> detection_classes(num_boxes_);
    for (int i = 0; i < num_boxes_; ++i) {
      detection_classes[i] =
          static_cast<int>(detection_classes_tensor->data<float>()[i]);
    }
    MP_RETURN_IF_ERROR(ConvertToDetections(detection_boxes, detection_scores,
                                           detection_classes.data(),
                                           output_detections));
  }
  return absl::OkStatus();
}

absl::Status OpenVINOTensorsToDetectionsCalculator::Close(CalculatorContext* cc) {
  return absl::OkStatus();
}

absl::Status OpenVINOTensorsToDetectionsCalculator::LoadOptions(
    CalculatorContext* cc) {
  // Get calculator options specified in the graph.
  options_ =
      cc->Options<::mediapipe::OpenVINOTensorsToDetectionsCalculatorOptions>();

  num_classes_ = options_.num_classes();
  num_boxes_ = options_.num_boxes();
  num_coords_ = options_.num_coords();

  // Currently only support 2D when num_values_per_keypoint equals to 2.
  CHECK_EQ(options_.num_values_per_keypoint(), 2);

  // Check if the output size is equal to the requested boxes and keypoints.
  CHECK_EQ(options_.num_keypoints() * options_.num_values_per_keypoint() +
               kNumCoordsPerBox,
           num_coords_);

  for (int i = 0; i < options_.ignore_classes_size(); ++i) {
    ignore_classes_.insert(options_.ignore_classes(i));
  }

  return absl::OkStatus();
}

absl::Status OpenVINOTensorsToDetectionsCalculator::DecodeBoxes(
    const float* raw_boxes, const std::vector<Anchor>& anchors,
    std::vector<float>* boxes) {
  for (int i = 0; i < num_boxes_; ++i) {
    const int box_offset = i * num_coords_ + options_.box_coord_offset();

    float y_center = raw_boxes[box_offset];
    float x_center = raw_boxes[box_offset + 1];
    float h = raw_boxes[box_offset + 2];
    float w = raw_boxes[box_offset + 3];
    if (options_.reverse_output_order()) {
      x_center = raw_boxes[box_offset];
      y_center = raw_boxes[box_offset + 1];
      w = raw_boxes[box_offset + 2];
      h = raw_boxes[box_offset + 3];
    }

    x_center =
        x_center / options_.x_scale() * anchors[i].w() + anchors[i].x_center();
    y_center =
        y_center / options_.y_scale() * anchors[i].h() + anchors[i].y_center();

    if (options_.apply_exponential_on_box_size()) {
      h = std::exp(h / options_.h_scale()) * anchors[i].h();
      w = std::exp(w / options_.w_scale()) * anchors[i].w();
    } else {
      h = h / options_.h_scale() * anchors[i].h();
      w = w / options_.w_scale() * anchors[i].w();
    }

    const float ymin = y_center - h / 2.f;
    const float xmin = x_center - w / 2.f;
    const float ymax = y_center + h / 2.f;
    const float xmax = x_center + w / 2.f;

    (*boxes)[i * num_coords_ + 0] = ymin;
    (*boxes)[i * num_coords_ + 1] = xmin;
    (*boxes)[i * num_coords_ + 2] = ymax;
    (*boxes)[i * num_coords_ + 3] = xmax;

    if (options_.num_keypoints()) {
      for (int k = 0; k < options_.num_keypoints(); ++k) {
        const int offset = i * num_coords_ + options_.keypoint_coord_offset() +
                           k * options_.num_values_per_keypoint();

        float keypoint_y = raw_boxes[offset];
        float keypoint_x = raw_boxes[offset + 1];
        if (options_.reverse_output_order()) {
          keypoint_x = raw_boxes[offset];
          keypoint_y = raw_boxes[offset + 1];
        }

        (*boxes)[offset] = keypoint_x / options_.x_scale() * anchors[i].w() +
                           anchors[i].x_center();
        (*boxes)[offset + 1] =
            keypoint_y / options_.y_scale() * anchors[i].h() +
            anchors[i].y_center();
      }
    }
  }

  return absl::OkStatus();
}

absl::Status OpenVINOTensorsToDetectionsCalculator::ConvertToDetections(
    const float* detection_boxes, const float* detection_scores,
    const int* detection_classes, std::vector<Detection>* output_detections) {
  for (int i = 0; i < num_boxes_; ++i) {
    if (options_.has_min_score_thresh() &&
        detection_scores[i] < options_.min_score_thresh()) {
      continue;
    }
    const int box_offset = i * num_coords_;
    Detection detection = ConvertToDetection(
        detection_boxes[box_offset + 0], detection_boxes[box_offset + 1],
        detection_boxes[box_offset + 2], detection_boxes[box_offset + 3],
        detection_scores[i], detection_classes[i], options_.flip_vertically());
    const auto& bbox = detection.location_data().relative_bounding_box();
    if (bbox.width() < 0 || bbox.height() < 0) {
      // Decoded detection boxes could have negative values for width/height due
      // to model prediction. Filter out those boxes since some downstream
      // calculators may assume non-negative values. (b/171391719)
      continue;
    }

    // Add keypoints.
    if (options_.num_keypoints() > 0) {
      auto* location_data = detection.mutable_location_data();
      for (int kp_id = 0; kp_id < options_.num_keypoints() *
                                      options_.num_values_per_keypoint();
           kp_id += options_.num_values_per_keypoint()) {
        auto keypoint = location_data->add_relative_keypoints();
        const int keypoint_index =
            box_offset + options_.keypoint_coord_offset() + kp_id;
        keypoint->set_x(detection_boxes[keypoint_index + 0]);
        keypoint->set_y(options_.flip_vertically()
                            ? 1.f - detection_boxes[keypoint_index + 1]
                            : detection_boxes[keypoint_index + 1]);
      }
    }
    output_detections->emplace_back(detection);
  }
  return absl::OkStatus();
}

Detection OpenVINOTensorsToDetectionsCalculator::ConvertToDetection(
    float box_ymin, float box_xmin, float box_ymax, float box_xmax, float score,
    int class_id, bool flip_vertically) {
  Detection detection;
  detection.add_score(score);
  detection.add_label_id(class_id);

  LocationData* location_data = detection.mutable_location_data();
  location_data->set_format(LocationData::RELATIVE_BOUNDING_BOX);

  LocationData::RelativeBoundingBox* relative_bbox =
      location_data->mutable_relative_bounding_box();

  relative_bbox->set_xmin(box_xmin);
  relative_bbox->set_ymin(flip_vertically ? 1.f - box_ymax : box_ymin);
  relative_bbox->set_width(box_xmax - box_xmin);
  relative_bbox->set_height(box_ymax - box_ymin);
  return detection;
}

}  // namespace mediapipe
