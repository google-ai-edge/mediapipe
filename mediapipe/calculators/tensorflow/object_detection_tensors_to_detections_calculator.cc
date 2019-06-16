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

#include <map>
#include <memory>
#include <set>
#include <string>
#include <vector>

#include "mediapipe/calculators/tensorflow/object_detection_tensors_to_detections_calculator.pb.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/formats/detection.pb.h"
#include "mediapipe/framework/formats/location.h"
#include "mediapipe/framework/port/canonical_errors.h"
#include "mediapipe/framework/port/ret_check.h"
#include "mediapipe/framework/port/source_location.h"
#include "mediapipe/framework/port/status.h"
#include "mediapipe/framework/port/status_builder.h"
#include "mediapipe/util/tensor_to_detection.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"

namespace mediapipe {
class CalculatorOptions;
}  // namespace mediapipe

namespace mediapipe {

namespace tf = ::tensorflow;

namespace {
const char kNumDetections[] = "NUM_DETECTIONS";
const char kBoxes[] = "BOXES";
const char kScores[] = "SCORES";
const char kClasses[] = "CLASSES";
const char kDetections[] = "DETECTIONS";
const char kKeypoints[] = "KEYPOINTS";
const char kMasks[] = "MASKS";
const char kLabelMap[] = "LABELMAP";
const int kNumCoordsPerBox = 4;
}  // namespace

// Takes object detection results and converts them into MediaPipe Detections.
//
// Inputs are assumed to be tensors of the form:
// `num_detections`     : float32 scalar tensor indicating the number of valid
//                        detections.
// `detection_boxes`    : float32 tensor of the form [num_boxes, 4]. Format for
//                        coordinates is {ymin, xmin, ymax, xmax}.
// `detection_scores`   : float32 tensor of the form [num_boxes].
// `detection_classes`  : float32 tensor of the form [num_boxes].
// `detection_keypoints`: float32 tensor of the form
//                        [num_boxes, num_keypoints, 2].
// `detection_masks`    : float32 tensor of the form
//                        [num_boxes, height, width].
//
// These are generated according to the Vale object detector model exporter,
// which may be found in
//   image/understanding/object_detection/export_inference_graph.py
//
// By default, the output Detections store label ids (integers) for each
// detection.  Optionally, a label map (of the form std::map<int, std::string>
// mapping label ids to label names as strings) can be made available as an
// input side packet, in which case the output Detections store
// labels as their associated std::string provided by the label map.
//
// Usage example:
// node {
//   calculator: "ObjectDetectionTensorsToDetectionsCalculator"
//   input_stream: "BOXES:detection_boxes_tensor"
//   input_stream: "SCORES:detection_scores_tensor"
//   input_stream: "CLASSES:detection_classes_tensor"
//   input_stream: "NUM_DETECTIONS:num_detections_tensor"
//   output_stream: "DETECTIONS:detections"
//   options: {
//     [mediapipe.ObjectDetectionsTensorToDetectionsCalculatorOptions.ext]: {
//         tensor_dim_to_squeeze: 0
//     }
//   }
// }
class ObjectDetectionTensorsToDetectionsCalculator : public CalculatorBase {
 public:
  ObjectDetectionTensorsToDetectionsCalculator() = default;

  static ::mediapipe::Status GetContract(CalculatorContract* cc) {
    cc->Inputs().Tag(kBoxes).Set<tf::Tensor>();
    cc->Inputs().Tag(kScores).Set<tf::Tensor>();

    if (cc->Inputs().HasTag(kNumDetections)) {
      cc->Inputs().Tag(kNumDetections).Set<tf::Tensor>();
    }
    if (cc->Inputs().HasTag(kClasses)) {
      cc->Inputs().Tag(kClasses).Set<tf::Tensor>();
    }
    if (cc->Inputs().HasTag(kKeypoints)) {
      cc->Inputs().Tag(kKeypoints).Set<tf::Tensor>();
    }

    if (cc->Inputs().HasTag(kMasks)) {
      cc->Inputs().Tag(kMasks).Set<tf::Tensor>();

      const auto& calculator_options =
          cc->Options<ObjectDetectionsTensorToDetectionsCalculatorOptions>();
      float mask_threshold = calculator_options.mask_threshold();
      if (!(mask_threshold >= 0.0 && mask_threshold <= 1.0)) {
        return ::mediapipe::InvalidArgumentErrorBuilder(MEDIAPIPE_LOC)
               << "mask_threshold must be in range [0.0, 1.0]";
      }
    }

    cc->Outputs().Tag(kDetections).Set<std::vector<Detection>>();

    if (cc->InputSidePackets().HasTag(kLabelMap)) {
      cc->InputSidePackets()
          .Tag(kLabelMap)
          .Set<std::unique_ptr<std::map<int, std::string>>>();
    }
    return ::mediapipe::OkStatus();
  }

  ::mediapipe::Status Open(CalculatorContext* cc) override {
    if (cc->InputSidePackets().HasTag(kLabelMap)) {
      label_map_ = GetFromUniquePtr<std::map<int, std::string>>(
          cc->InputSidePackets().Tag(kLabelMap));
    }
    const auto& tensor_dim_to_squeeze_field =
        cc->Options<ObjectDetectionsTensorToDetectionsCalculatorOptions>()
            .tensor_dim_to_squeeze();
    tensor_dims_to_squeeze_ = std::vector<int32>(
        tensor_dim_to_squeeze_field.begin(), tensor_dim_to_squeeze_field.end());
    std::sort(tensor_dims_to_squeeze_.rbegin(), tensor_dims_to_squeeze_.rend());
    cc->SetOffset(0);
    return ::mediapipe::OkStatus();
  }

  ::mediapipe::Status Process(CalculatorContext* cc) override {
    const auto& options =
        cc->Options<ObjectDetectionsTensorToDetectionsCalculatorOptions>();

    tf::Tensor input_num_detections_tensor =
        tf::Tensor(tf::DT_FLOAT, tf::TensorShape({0}));
    if (cc->Inputs().HasTag(kClasses)) {
      ASSIGN_OR_RETURN(
          input_num_detections_tensor,
          MaybeSqueezeDims(kNumDetections,
                           cc->Inputs().Tag(kNumDetections).Get<tf::Tensor>()));
    }
    if (input_num_detections_tensor.dtype() != tf::DT_INT32) {
      RET_CHECK_EQ(input_num_detections_tensor.dtype(), tf::DT_FLOAT);
    }

    ASSIGN_OR_RETURN(
        auto input_boxes_tensor,
        MaybeSqueezeDims(kBoxes, cc->Inputs().Tag(kBoxes).Get<tf::Tensor>()));
    RET_CHECK_EQ(input_boxes_tensor.dtype(), tf::DT_FLOAT);

    ASSIGN_OR_RETURN(
        auto input_scores_tensor,
        MaybeSqueezeDims(kScores, cc->Inputs().Tag(kScores).Get<tf::Tensor>()));
    RET_CHECK_EQ(input_scores_tensor.dtype(), tf::DT_FLOAT);

    tf::Tensor input_classes_tensor =
        tf::Tensor(tf::DT_FLOAT, tf::TensorShape({0}));
    if (cc->Inputs().HasTag(kClasses)) {
      ASSIGN_OR_RETURN(
          input_classes_tensor,
          MaybeSqueezeDims(kClasses,
                           cc->Inputs().Tag(kClasses).Get<tf::Tensor>()));
    }
    RET_CHECK_EQ(input_classes_tensor.dtype(), tf::DT_FLOAT);

    auto output_detections = absl::make_unique<std::vector<Detection>>();

    const tf::Tensor& input_keypoints_tensor =
        cc->Inputs().HasTag(kKeypoints)
            ? cc->Inputs().Tag(kKeypoints).Get<tf::Tensor>()
            : tf::Tensor(tf::DT_FLOAT, tf::TensorShape({0, 0, 0}));

    const tf::Tensor& input_masks_tensor =
        cc->Inputs().HasTag(kMasks)
            ? cc->Inputs().Tag(kMasks).Get<tf::Tensor>()
            : tf::Tensor(tf::DT_FLOAT, tf::TensorShape({0, 0, 0}));
    RET_CHECK_EQ(input_masks_tensor.dtype(), tf::DT_FLOAT);

    const std::map<int, std::string> label_map =
        (label_map_ == nullptr) ? std::map<int, std::string>{} : *label_map_;

    RET_CHECK_OK(TensorsToDetections(
        input_num_detections_tensor, input_boxes_tensor, input_scores_tensor,
        input_classes_tensor, input_keypoints_tensor, input_masks_tensor,
        options.mask_threshold(), label_map, output_detections.get()));

    cc->Outputs()
        .Tag(kDetections)
        .Add(output_detections.release(), cc->InputTimestamp());

    return ::mediapipe::OkStatus();
  }

 private:
  std::map<int, std::string>* label_map_;
  std::vector<int32> tensor_dims_to_squeeze_;

  ::mediapipe::StatusOr<tf::Tensor> MaybeSqueezeDims(
      const std::string& tensor_tag, const tf::Tensor& input_tensor) {
    if (tensor_dims_to_squeeze_.empty()) {
      return input_tensor;
    }
    tf::TensorShape tensor_shape = input_tensor.shape();
    for (const int dim : tensor_dims_to_squeeze_) {
      RET_CHECK_GT(tensor_shape.dims(), dim)
          << "Dimension " << dim
          << " does not exist in input tensor with num dimensions "
          << input_tensor.dims() << " dims";
      RET_CHECK_EQ(tensor_shape.dim_size(dim), 1)
          << "Cannot remove dimension " << dim << " with size "
          << tensor_shape.dim_size(dim);
      tensor_shape.RemoveDim(dim);
    }
    tf::Tensor output_tensor;
    RET_CHECK(output_tensor.CopyFrom(input_tensor, tensor_shape));
    return std::move(output_tensor);
  }
};

REGISTER_CALCULATOR(ObjectDetectionTensorsToDetectionsCalculator);

}  // namespace mediapipe
