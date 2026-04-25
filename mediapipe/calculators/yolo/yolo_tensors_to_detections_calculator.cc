/* Copyright 2026 The MediaPipe Authors.

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

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <type_traits>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/strings/str_join.h"
#include "absl/strings/str_format.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/formats/detection.pb.h"
#include "mediapipe/framework/formats/tensor.h"
#include "mediapipe/tasks/cc/common.h"
#include "mediapipe/tasks/cc/components/processors/proto/yolo_tensors_to_detections_calculator.pb.h"

namespace mediapipe {
namespace tasks {
namespace components {
namespace processors {

namespace {

using Options = proto::YoloTensorsToDetectionsCalculatorOptions;

constexpr char kDetectionsTag[] = "DETECTIONS";
constexpr char kTensorsTag[] = "TENSORS";
constexpr int kBoxFeatureCount = 4;
constexpr int kEndToEndFeatureCount = 6;

absl::Status ValidateQuantizationOverride(const Options& options) {
  if (options.has_quantization_scale_override() !=
      options.has_quantization_zero_point_override()) {
    return CreateStatusWithPayload(
        absl::StatusCode::kInvalidArgument,
        "YOLO output quantization override requires both "
        "`quantization_scale_override` and "
        "`quantization_zero_point_override`.",
        MediaPipeTasksStatus::kInvalidArgumentError);
  }
  if (options.has_quantization_scale_override() &&
      options.quantization_scale_override() == 0.0f) {
    return CreateStatusWithPayload(
        absl::StatusCode::kInvalidArgument,
        "`quantization_scale_override` must be non-zero.",
        MediaPipeTasksStatus::kInvalidArgumentError);
  }
  return absl::OkStatus();
}

Tensor::QuantizationParameters ResolveQuantizationParameters(
    const Tensor& output_tensor, const Options& options) {
  if (!options.has_quantization_scale_override()) {
    return output_tensor.quantization_parameters();
  }
  return Tensor::QuantizationParameters(options.quantization_scale_override(),
                                        options.quantization_zero_point_override());
}

absl::StatusOr<std::vector<int>> SqueezeLeadingSingletonDims(
    const Tensor::Shape& shape) {
  std::vector<int> dims = shape.dims;
  while (dims.size() > 2 && !dims.empty() && dims.front() == 1) {
    dims.erase(dims.begin());
  }
  if (dims.size() != 2) {
    return CreateStatusWithPayload(
        absl::StatusCode::kInvalidArgument,
        absl::StrFormat(
            "Expected a YOLO output tensor with rank 2 after squeezing "
            "leading singleton dimensions, found shape %s",
            absl::StrJoin(shape.dims, "x")),
        MediaPipeTasksStatus::kInvalidArgumentError);
  }
  return dims;
}

absl::StatusOr<Options::DecodeMode> ResolveDecodeMode(
    const std::vector<int>& dims, Options::DecodeMode configured_mode) {
  if (configured_mode != Options::DECODE_MODE_AUTO) {
    return configured_mode;
  }
  if ((dims[0] == kEndToEndFeatureCount && dims[1] > kEndToEndFeatureCount) ||
      (dims[1] == kEndToEndFeatureCount && dims[0] > kEndToEndFeatureCount)) {
    return Options::END_TO_END;
  }
  return Options::ULTRALYTICS_DETECTION_HEAD;
}

absl::StatusOr<Options::TensorLayout> ResolveTensorLayout(
    const std::vector<int>& dims, Options::DecodeMode decode_mode,
    Options::TensorLayout configured_layout) {
  if (configured_layout != Options::TENSOR_LAYOUT_AUTO) {
    return configured_layout;
  }
  const int first_dim = dims[0];
  const int second_dim = dims[1];
  if (decode_mode == Options::END_TO_END) {
    if (second_dim == kEndToEndFeatureCount && first_dim > 0) {
      return Options::BOXES_FIRST;
    }
    if (first_dim == kEndToEndFeatureCount && second_dim > 0) {
      return Options::FEATURES_FIRST;
    }
    return CreateStatusWithPayload(
        absl::StatusCode::kInvalidArgument,
        absl::StrFormat(
            "Unable to infer end-to-end YOLO tensor layout from shape [%d, "
            "%d]. Specify `tensor_layout` explicitly.",
            first_dim, second_dim),
        MediaPipeTasksStatus::kInvalidArgumentError);
  }

  if (first_dim > second_dim && second_dim > kBoxFeatureCount &&
      second_dim <= 512) {
    return Options::BOXES_FIRST;
  }
  if (second_dim > first_dim && first_dim > kBoxFeatureCount &&
      first_dim <= 512) {
    return Options::FEATURES_FIRST;
  }
  return CreateStatusWithPayload(
      absl::StatusCode::kInvalidArgument,
      absl::StrFormat(
          "Unable to infer YOLO tensor layout from shape [%d, %d]. Specify "
          "`tensor_layout` explicitly.",
          first_dim, second_dim),
      MediaPipeTasksStatus::kInvalidArgumentError);
}

template <typename T>
float ReadTensorValue(const T* data, int index,
                      const Tensor::QuantizationParameters& quantization) {
  if constexpr (std::is_same_v<T, float>) {
    return data[index];
  } else {
    return (static_cast<int>(data[index]) - quantization.zero_point) *
           quantization.scale;
  }
}

Detection BuildDetection(int class_id, float score, float cx, float cy, float w,
                         float h, int input_width, int input_height) {
  // Ultralytics TFLite exports commonly emit normalized cx/cy/w/h in [0, 1],
  // while some other YOLO exports emit pixel-space coordinates. Support both.
  const bool coordinates_are_normalized =
      std::max({std::fabs(cx), std::fabs(cy), std::fabs(w), std::fabs(h)}) <=
      2.0f;

  if (!coordinates_are_normalized) {
    cx /= input_width;
    cy /= input_height;
    w /= input_width;
    h /= input_height;
  }

  const float half_w = w / 2.0f;
  const float half_h = h / 2.0f;

  const float xmin = std::clamp(cx - half_w, 0.0f, 1.0f);
  const float ymin = std::clamp(cy - half_h, 0.0f, 1.0f);
  const float xmax = std::clamp(cx + half_w, 0.0f, 1.0f);
  const float ymax = std::clamp(cy + half_h, 0.0f, 1.0f);

  Detection detection;
  detection.add_score(score);
  detection.add_label_id(class_id);

  auto* location_data = detection.mutable_location_data();
  location_data->set_format(LocationData::RELATIVE_BOUNDING_BOX);
  auto* relative_bbox = location_data->mutable_relative_bounding_box();
  relative_bbox->set_xmin(xmin);
  relative_bbox->set_ymin(ymin);
  relative_bbox->set_width(std::max(0.0f, xmax - xmin));
  relative_bbox->set_height(std::max(0.0f, ymax - ymin));
  return detection;
}

Detection BuildDetectionFromCorners(int class_id, float score, float x1,
                                    float y1, float x2, float y2,
                                    int input_width, int input_height) {
  const bool coordinates_are_normalized =
      std::max({std::fabs(x1), std::fabs(y1), std::fabs(x2), std::fabs(y2)}) <=
      2.0f;

  if (!coordinates_are_normalized) {
    x1 /= input_width;
    x2 /= input_width;
    y1 /= input_height;
    y2 /= input_height;
  }

  const float xmin = std::clamp(std::min(x1, x2), 0.0f, 1.0f);
  const float ymin = std::clamp(std::min(y1, y2), 0.0f, 1.0f);
  const float xmax = std::clamp(std::max(x1, x2), 0.0f, 1.0f);
  const float ymax = std::clamp(std::max(y1, y2), 0.0f, 1.0f);

  Detection detection;
  detection.add_score(score);
  detection.add_label_id(class_id);

  auto* location_data = detection.mutable_location_data();
  location_data->set_format(LocationData::RELATIVE_BOUNDING_BOX);
  auto* relative_bbox = location_data->mutable_relative_bounding_box();
  relative_bbox->set_xmin(xmin);
  relative_bbox->set_ymin(ymin);
  relative_bbox->set_width(std::max(0.0f, xmax - xmin));
  relative_bbox->set_height(std::max(0.0f, ymax - ymin));
  return detection;
}

}  // namespace

class YoloTensorsToDetectionsCalculator : public CalculatorBase {
 public:
  static absl::Status GetContract(CalculatorContract* cc) {
    cc->Inputs().Tag(kTensorsTag).Set<std::vector<Tensor>>();
    cc->Outputs().Tag(kDetectionsTag).Set<std::vector<Detection>>();
    return absl::OkStatus();
  }

  absl::Status Open(CalculatorContext* cc) override {
    options_ = cc->Options<Options>();
    RET_CHECK(options_.has_input_width() && options_.has_input_height())
        << "YOLO calculator requires input_width and input_height.";
    RET_CHECK_GT(options_.input_width(), 0);
    RET_CHECK_GT(options_.input_height(), 0);
    MP_RETURN_IF_ERROR(ValidateQuantizationOverride(options_));
    return absl::OkStatus();
  }

  absl::Status Process(CalculatorContext* cc) override {
    const auto& tensors =
        cc->Inputs().Tag(kTensorsTag).Get<std::vector<Tensor>>();
    RET_CHECK(!tensors.empty())
        << "YOLO detector expects at least one output tensor.";
    const Tensor& output_tensor = tensors[0];
    MP_ASSIGN_OR_RETURN(std::vector<int> squeezed_dims,
                        SqueezeLeadingSingletonDims(output_tensor.shape()));
    MP_ASSIGN_OR_RETURN(const Options::DecodeMode decode_mode,
                        ResolveDecodeMode(squeezed_dims, options_.decode_mode()));
    MP_ASSIGN_OR_RETURN(
        const Options::TensorLayout layout,
        ResolveTensorLayout(squeezed_dims, decode_mode,
                            options_.tensor_layout()));

    const int num_boxes =
        layout == Options::BOXES_FIRST ? squeezed_dims[0] : squeezed_dims[1];
    const int num_features =
        layout == Options::BOXES_FIRST ? squeezed_dims[1] : squeezed_dims[0];
    if (decode_mode == Options::END_TO_END) {
      RET_CHECK_EQ(num_features, kEndToEndFeatureCount)
          << "End-to-end YOLO tensors must expose exactly 6 values per "
             "candidate: x1, y1, x2, y2, score, class_id.";
    } else {
      RET_CHECK_GT(num_features, kBoxFeatureCount)
          << "YOLO tensor must contain 4 box coordinates and at least one "
             "class score.";
    }

    auto detections = absl::make_unique<std::vector<Detection>>();
    detections->reserve(num_boxes);

    const auto quantization =
        ResolveQuantizationParameters(output_tensor, options_);
    switch (output_tensor.element_type()) {
      case Tensor::ElementType::kFloat32:
        if (decode_mode == Options::END_TO_END) {
          DecodeEndToEnd<float>(output_tensor.GetCpuReadView().buffer<float>(),
                                num_boxes, num_features, layout, quantization,
                                detections.get());
        } else {
          DecodeUltralytics<float>(
              output_tensor.GetCpuReadView().buffer<float>(), num_boxes,
              num_features, layout, quantization, detections.get());
        }
        break;
      case Tensor::ElementType::kUInt8:
        if (decode_mode == Options::END_TO_END) {
          DecodeEndToEnd<uint8_t>(
              output_tensor.GetCpuReadView().buffer<uint8_t>(), num_boxes,
              num_features, layout, quantization, detections.get());
        } else {
          DecodeUltralytics<uint8_t>(
              output_tensor.GetCpuReadView().buffer<uint8_t>(), num_boxes,
              num_features, layout, quantization, detections.get());
        }
        break;
      case Tensor::ElementType::kInt8:
        if (decode_mode == Options::END_TO_END) {
          DecodeEndToEnd<int8_t>(
              output_tensor.GetCpuReadView().buffer<int8_t>(), num_boxes,
              num_features, layout, quantization, detections.get());
        } else {
          DecodeUltralytics<int8_t>(
              output_tensor.GetCpuReadView().buffer<int8_t>(), num_boxes,
              num_features, layout, quantization, detections.get());
        }
        break;
      default:
        return CreateStatusWithPayload(
            absl::StatusCode::kInvalidArgument,
            absl::StrFormat(
                "Unsupported YOLO output tensor type: %s",
                Tensor::ElementTypeName(output_tensor.element_type())),
            MediaPipeTasksStatus::kInvalidArgumentError);
    }

    cc->Outputs().Tag(kDetectionsTag).Add(detections.release(),
                                          cc->InputTimestamp());
    return absl::OkStatus();
  }

 private:
  template <typename T>
  void DecodeUltralytics(const T* data, int num_boxes, int num_features,
                         Options::TensorLayout layout,
                         const Tensor::QuantizationParameters& quantization,
                         std::vector<Detection>* detections) {
    const float score_threshold =
        options_.has_min_score_threshold()
            ? options_.min_score_threshold()
            : std::numeric_limits<float>::lowest();

    auto value_at = [&](int box_index, int feature_index) {
      const int flat_index =
          layout == Options::BOXES_FIRST
              ? box_index * num_features + feature_index
              : feature_index * num_boxes + box_index;
      return ReadTensorValue(data, flat_index, quantization);
    };

    for (int box_index = 0; box_index < num_boxes; ++box_index) {
      int best_class = -1;
      float best_score = score_threshold;
      for (int feature_index = kBoxFeatureCount; feature_index < num_features;
           ++feature_index) {
        const float score = value_at(box_index, feature_index);
        if (score > best_score) {
          best_score = score;
          best_class = feature_index - kBoxFeatureCount;
        }
      }
      if (best_class < 0) {
        continue;
      }

      const float cx = value_at(box_index, 0);
      const float cy = value_at(box_index, 1);
      const float w = value_at(box_index, 2);
      const float h = value_at(box_index, 3);
      if (w <= 0.0f || h <= 0.0f) {
        continue;
      }

      Detection detection =
          BuildDetection(best_class, best_score, cx, cy, w, h,
                         options_.input_width(), options_.input_height());
      const auto& bbox = detection.location_data().relative_bounding_box();
      if (bbox.width() <= 0.0f || bbox.height() <= 0.0f) {
        continue;
      }
      detections->push_back(std::move(detection));
    }
  }

  template <typename T>
  void DecodeEndToEnd(const T* data, int num_boxes, int num_features,
                      Options::TensorLayout layout,
                      const Tensor::QuantizationParameters& quantization,
                      std::vector<Detection>* detections) {
    const float score_threshold =
        options_.has_min_score_threshold()
            ? options_.min_score_threshold()
            : std::numeric_limits<float>::lowest();

    auto value_at = [&](int box_index, int feature_index) {
      const int flat_index =
          layout == Options::BOXES_FIRST
              ? box_index * num_features + feature_index
              : feature_index * num_boxes + box_index;
      return ReadTensorValue(data, flat_index, quantization);
    };

    for (int box_index = 0; box_index < num_boxes; ++box_index) {
      const float score = value_at(box_index, 4);
      if (score <= score_threshold) {
        continue;
      }
      const int class_id =
          static_cast<int>(std::lround(value_at(box_index, 5)));
      if (class_id < 0) {
        continue;
      }

      Detection detection = BuildDetectionFromCorners(
          class_id, score, value_at(box_index, 0), value_at(box_index, 1),
          value_at(box_index, 2), value_at(box_index, 3),
          options_.input_width(), options_.input_height());
      const auto& bbox = detection.location_data().relative_bounding_box();
      if (bbox.width() <= 0.0f || bbox.height() <= 0.0f) {
        continue;
      }
      detections->push_back(std::move(detection));
    }
  }

  Options options_;
};

REGISTER_CALCULATOR(YoloTensorsToDetectionsCalculator);

}  // namespace processors
}  // namespace components
}  // namespace tasks
}  // namespace mediapipe
