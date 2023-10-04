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

#include <memory>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "mediapipe/framework/api2/node.h"
#include "mediapipe/framework/api2/packet.h"
#include "mediapipe/framework/api2/port.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/formats/detection.pb.h"
#include "mediapipe/framework/formats/location_data.pb.h"

namespace mediapipe {
namespace api2 {
namespace {

template <typename T>
T BoundedValue(T value, T upper_bound) {
  T output = std::min(value, upper_bound);
  if (output < 0) {
    return 0;
  }
  return output;
}

absl::Status ConvertRelativeBoundingBoxToBoundingBox(
    const std::pair<int, int>& image_size, Detection* detection) {
  const int image_width = image_size.first;
  const int image_height = image_size.second;
  const auto& relative_bbox =
      detection->location_data().relative_bounding_box();
  auto* bbox = detection->mutable_location_data()->mutable_bounding_box();
  bbox->set_xmin(
      BoundedValue<int>(relative_bbox.xmin() * image_width, image_width));
  bbox->set_ymin(
      BoundedValue<int>(relative_bbox.ymin() * image_height, image_height));
  bbox->set_width(
      BoundedValue<int>(relative_bbox.width() * image_width, image_width));
  bbox->set_height(
      BoundedValue<int>(relative_bbox.height() * image_height, image_height));
  detection->mutable_location_data()->set_format(LocationData::BOUNDING_BOX);
  detection->mutable_location_data()->clear_relative_bounding_box();
  return absl::OkStatus();
}

absl::Status ConvertBoundingBoxToRelativeBoundingBox(
    const std::pair<int, int>& image_size, Detection* detection) {
  int image_width = image_size.first;
  int image_height = image_size.second;
  const auto& bbox = detection->location_data().bounding_box();
  auto* relative_bbox =
      detection->mutable_location_data()->mutable_relative_bounding_box();
  relative_bbox->set_xmin(
      BoundedValue<float>((float)bbox.xmin() / image_width, 1.0f));
  relative_bbox->set_ymin(
      BoundedValue<float>((float)bbox.ymin() / image_height, 1.0f));
  relative_bbox->set_width(
      BoundedValue<float>((float)bbox.width() / image_width, 1.0f));
  relative_bbox->set_height(
      BoundedValue<float>((float)bbox.height() / image_height, 1.0f));
  detection->mutable_location_data()->clear_bounding_box();
  detection->mutable_location_data()->set_format(
      LocationData::RELATIVE_BOUNDING_BOX);
  return absl::OkStatus();
}

absl::StatusOr<LocationData::Format> GetLocationDataFormat(
    const Detection& detection) {
  if (!detection.has_location_data()) {
    return absl::InvalidArgumentError("Detection must have location data.");
  }
  LocationData::Format format = detection.location_data().format();
  RET_CHECK(format == LocationData::RELATIVE_BOUNDING_BOX ||
            format == LocationData::BOUNDING_BOX)
      << "Detection's location data format must be either "
         "RELATIVE_BOUNDING_BOX or BOUNDING_BOX";
  return format;
}

absl::StatusOr<LocationData::Format> GetLocationDataFormat(
    std::vector<Detection>& detections) {
  RET_CHECK(!detections.empty());
  LocationData::Format output_format;
  MP_ASSIGN_OR_RETURN(output_format, GetLocationDataFormat(detections[0]));
  for (int i = 1; i < detections.size(); ++i) {
    MP_ASSIGN_OR_RETURN(LocationData::Format format,
                        GetLocationDataFormat(detections[i]));
    if (output_format != format) {
      return absl::InvalidArgumentError(
          "Input detections have different location data formats.");
    }
  }
  return output_format;
}

absl::Status ConvertBoundingBox(const std::pair<int, int>& image_size,
                                Detection* detection) {
  if (!detection->has_location_data()) {
    return absl::InvalidArgumentError("Detection must have location data.");
  }
  switch (detection->location_data().format()) {
    case LocationData::RELATIVE_BOUNDING_BOX:
      return ConvertRelativeBoundingBoxToBoundingBox(image_size, detection);
    case LocationData::BOUNDING_BOX:
      return ConvertBoundingBoxToRelativeBoundingBox(image_size, detection);
    default:
      return absl::InvalidArgumentError(
          "Detection's location data format must be either "
          "RELATIVE_BOUNDING_BOX or BOUNDING_BOX.");
  }
}

}  // namespace

// Transforms relative bounding box(es) to pixel bounding box(es) in a detection
// proto/detection list/detection vector, or vice versa.
//
// Inputs:
// One of the following:
// DETECTION: A Detection proto.
// DETECTIONS: An std::vector<Detection>/ a DetectionList proto.
// IMAGE_SIZE: A std::pair<int, int> represention image width and height.
//
// Outputs:
// At least one of the following:
// PIXEL_DETECTION: A Detection proto with pixel bounding box.
// PIXEL_DETECTIONS: An std::vector<Detection> with pixel bounding boxes.
// PIXEL_DETECTION_LIST: A DetectionList proto with pixel bounding boxes.
// RELATIVE_DETECTION: A Detection proto with relative bounding box.
// RELATIVE_DETECTIONS: An std::vector<Detection> with relative bounding boxes.
// RELATIVE_DETECTION_LIST: A DetectionList proto with relative bounding boxes.
//
// Example config:
// For input detection(s) with relative bounding box(es):
// node {
//   calculator: "DetectionTransformationCalculator"
//   input_stream: "DETECTION:input_detection"
//   input_stream: "IMAGE_SIZE:image_size"
//   output_stream: "PIXEL_DETECTION:output_detection"
//   output_stream: "PIXEL_DETECTIONS:output_detections"
//   output_stream: "PIXEL_DETECTION_LIST:output_detection_list"
// }
//
// For input detection(s) with pixel bounding box(es):
// node {
//   calculator: "DetectionTransformationCalculator"
//   input_stream: "DETECTION:input_detection"
//   input_stream: "IMAGE_SIZE:image_size"
//   output_stream: "RELATIVE_DETECTION:output_detection"
//   output_stream: "RELATIVE_DETECTIONS:output_detections"
//   output_stream: "RELATIVE_DETECTION_LIST:output_detection_list"
// }
class DetectionTransformationCalculator : public Node {
 public:
  static constexpr Input<Detection>::Optional kInDetection{"DETECTION"};
  static constexpr Input<OneOf<DetectionList, std::vector<Detection>>>::Optional
      kInDetections{"DETECTIONS"};
  static constexpr Input<std::pair<int, int>> kInImageSize{"IMAGE_SIZE"};
  static constexpr Output<Detection>::Optional kOutPixelDetection{
      "PIXEL_DETECTION"};
  static constexpr Output<std::vector<Detection>>::Optional kOutPixelDetections{
      "PIXEL_DETECTIONS"};
  static constexpr Output<DetectionList>::Optional kOutPixelDetectionList{
      "PIXEL_DETECTION_LIST"};
  static constexpr Output<Detection>::Optional kOutRelativeDetection{
      "RELATIVE_DETECTION"};
  static constexpr Output<std::vector<Detection>>::Optional
      kOutRelativeDetections{"RELATIVE_DETECTIONS"};
  static constexpr Output<DetectionList>::Optional kOutRelativeDetectionList{
      "RELATIVE_DETECTION_LIST"};

  MEDIAPIPE_NODE_CONTRACT(kInDetection, kInDetections, kInImageSize,
                          kOutPixelDetection, kOutPixelDetections,
                          kOutPixelDetectionList, kOutRelativeDetection,
                          kOutRelativeDetections, kOutRelativeDetectionList);

  static absl::Status UpdateContract(CalculatorContract* cc) {
    RET_CHECK(kInImageSize(cc).IsConnected()) << "Image size must be provided.";
    RET_CHECK(kInDetections(cc).IsConnected() ^ kInDetection(cc).IsConnected());
    if (kInDetections(cc).IsConnected()) {
      RET_CHECK(kOutPixelDetections(cc).IsConnected() ||
                kOutPixelDetectionList(cc).IsConnected() ||
                kOutRelativeDetections(cc).IsConnected() ||
                kOutRelativeDetectionList(cc).IsConnected())
          << "Output must be a container of detections.";
    }
    RET_CHECK(kOutPixelDetections(cc).IsConnected() ||
              kOutPixelDetectionList(cc).IsConnected() ||
              kOutPixelDetection(cc).IsConnected() ||
              kOutRelativeDetections(cc).IsConnected() ||
              kOutRelativeDetectionList(cc).IsConnected() ||
              kOutRelativeDetection(cc).IsConnected())
        << "Must connect at least one output stream.";
    return absl::OkStatus();
  }

  absl::Status Open(CalculatorContext* cc) override {
    output_pixel_bounding_boxes_ = kOutPixelDetections(cc).IsConnected() ||
                                   kOutPixelDetectionList(cc).IsConnected() ||
                                   kOutPixelDetection(cc).IsConnected();
    output_relative_bounding_boxes_ =
        kOutRelativeDetections(cc).IsConnected() ||
        kOutRelativeDetectionList(cc).IsConnected() ||
        kOutRelativeDetection(cc).IsConnected();
    RET_CHECK(output_pixel_bounding_boxes_ ^ output_relative_bounding_boxes_)
        << "All output streams must have the same stream tag prefix, either "
           "\"PIXEL\" or \"RELATIVE_\".";
    return absl::OkStatus();
  }

  absl::Status Process(CalculatorContext* cc) override {
    std::pair<int, int> image_size = kInImageSize(cc).Get();
    std::vector<Detection> transformed_detections;
    LocationData::Format input_location_data_format;
    if (kInDetections(cc).IsEmpty() && kInDetection(cc).IsEmpty()) {
      return absl::OkStatus();
    }
    if (kInDetections(cc).IsConnected()) {
      transformed_detections = kInDetections(cc).Visit(
          [&](const DetectionList& detection_list) {
            return std::vector<Detection>(detection_list.detection().begin(),
                                          detection_list.detection().end());
          },
          [&](const std::vector<Detection>& detection_vector) {
            return detection_vector;
          });
      if (transformed_detections.empty()) {
        OutputEmptyDetections(cc);
        return absl::OkStatus();
      }
      MP_ASSIGN_OR_RETURN(input_location_data_format,
                          GetLocationDataFormat(transformed_detections));
      for (Detection& detection : transformed_detections) {
        MP_RETURN_IF_ERROR(ConvertBoundingBox(image_size, &detection));
      }
    } else {
      Detection transformed_detection(kInDetection(cc).Get());
      if (!transformed_detection.has_location_data()) {
        OutputEmptyDetections(cc);
        return absl::OkStatus();
      }
      MP_ASSIGN_OR_RETURN(input_location_data_format,
                          GetLocationDataFormat(kInDetection(cc).Get()));
      MP_RETURN_IF_ERROR(
          ConvertBoundingBox(image_size, &transformed_detection));
      transformed_detections.push_back(transformed_detection);
    }
    if (input_location_data_format == LocationData::RELATIVE_BOUNDING_BOX) {
      RET_CHECK(!output_relative_bounding_boxes_)
          << "Input detections are with relative bounding box(es), and the "
             "output detections must have pixel bounding box(es).";
      if (kOutPixelDetection(cc).IsConnected()) {
        kOutPixelDetection(cc).Send(transformed_detections[0]);
      }
      if (kOutPixelDetections(cc).IsConnected()) {
        kOutPixelDetections(cc).Send(transformed_detections);
      }
      if (kOutPixelDetectionList(cc).IsConnected()) {
        DetectionList detection_list;
        for (const auto& detection : transformed_detections) {
          detection_list.add_detection()->CopyFrom(detection);
        }
        kOutPixelDetectionList(cc).Send(detection_list);
      }
    } else {
      RET_CHECK(!output_pixel_bounding_boxes_)
          << "Input detections are with pixel bounding box(es), and the "
             "output detections must have relative bounding box(es).";
      if (kOutRelativeDetection(cc).IsConnected()) {
        kOutRelativeDetection(cc).Send(transformed_detections[0]);
      }
      if (kOutRelativeDetections(cc).IsConnected()) {
        kOutRelativeDetections(cc).Send(transformed_detections);
      }
      if (kOutRelativeDetectionList(cc).IsConnected()) {
        DetectionList detection_list;
        for (const auto& detection : transformed_detections) {
          detection_list.add_detection()->CopyFrom(detection);
        }
        kOutRelativeDetectionList(cc).Send(detection_list);
      }
    }
    return absl::OkStatus();
  }

 private:
  void OutputEmptyDetections(CalculatorContext* cc) {
    if (kOutPixelDetection(cc).IsConnected()) {
      kOutPixelDetection(cc).Send(Detection());
    }
    if (kOutPixelDetections(cc).IsConnected()) {
      kOutPixelDetections(cc).Send(std::vector<Detection>());
    }
    if (kOutPixelDetectionList(cc).IsConnected()) {
      kOutPixelDetectionList(cc).Send(DetectionList());
    }
    if (kOutRelativeDetection(cc).IsConnected()) {
      kOutRelativeDetection(cc).Send(Detection());
    }
    if (kOutRelativeDetections(cc).IsConnected()) {
      kOutRelativeDetections(cc).Send(std::vector<Detection>());
    }
    if (kOutRelativeDetectionList(cc).IsConnected()) {
      kOutRelativeDetectionList(cc).Send(DetectionList());
    }
  }

  bool output_relative_bounding_boxes_;
  bool output_pixel_bounding_boxes_;
};

MEDIAPIPE_REGISTER_NODE(DetectionTransformationCalculator);

}  // namespace api2
}  // namespace mediapipe
