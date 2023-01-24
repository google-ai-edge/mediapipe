/* Copyright 2023 The MediaPipe Authors. All Rights Reserved.

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

#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "mediapipe/calculators/core/clip_vector_size_calculator.pb.h"
#include "mediapipe/calculators/tensor/image_to_tensor_calculator.pb.h"
#include "mediapipe/calculators/tensor/tensors_to_detections_calculator.pb.h"
#include "mediapipe/calculators/tflite/ssd_anchors_calculator.pb.h"
#include "mediapipe/calculators/util/detection_label_id_to_text_calculator.pb.h"
#include "mediapipe/calculators/util/detections_to_rects_calculator.pb.h"
#include "mediapipe/calculators/util/non_max_suppression_calculator.pb.h"
#include "mediapipe/calculators/util/rect_transformation_calculator.pb.h"
#include "mediapipe/framework/api2/builder.h"
#include "mediapipe/framework/api2/port.h"
#include "mediapipe/framework/formats/detection.pb.h"
#include "mediapipe/framework/formats/image.h"
#include "mediapipe/framework/formats/rect.pb.h"
#include "mediapipe/framework/formats/tensor.h"
#include "mediapipe/tasks/cc/common.h"
#include "mediapipe/tasks/cc/components/processors/image_preprocessing_graph.h"
#include "mediapipe/tasks/cc/core/model_resources.h"
#include "mediapipe/tasks/cc/core/model_task_graph.h"
#include "mediapipe/tasks/cc/core/proto/inference_subgraph.pb.h"
#include "mediapipe/tasks/cc/core/utils.h"
#include "mediapipe/tasks/cc/vision/face_detector/proto/face_detector_graph_options.pb.h"
#include "mediapipe/tasks/cc/vision/utils/image_tensor_specs.h"

namespace mediapipe {
namespace tasks {
namespace vision {
namespace face_detector {

using ::mediapipe::NormalizedRect;
using ::mediapipe::Tensor;
using ::mediapipe::api2::Input;
using ::mediapipe::api2::Output;
using ::mediapipe::api2::builder::Graph;
using ::mediapipe::api2::builder::Source;
using ::mediapipe::tasks::vision::face_detector::proto::
    FaceDetectorGraphOptions;

namespace {
constexpr char kImageTag[] = "IMAGE";
constexpr char kNormRectTag[] = "NORM_RECT";
constexpr char kDetectionsTag[] = "DETECTIONS";

void ConfigureSsdAnchorsCalculator(
    mediapipe::SsdAnchorsCalculatorOptions* options) {
  // TODO config SSD anchors parameters from metadata.
  options->set_num_layers(1);
  options->set_min_scale(0.1484375);
  options->set_max_scale(0.75);
  options->set_input_size_height(192);
  options->set_input_size_width(192);
  options->set_anchor_offset_x(0.5);
  options->set_anchor_offset_y(0.5);
  options->add_strides(4);
  options->add_aspect_ratios(1.0);
  options->set_fixed_anchor_size(true);
  options->set_interpolated_scale_aspect_ratio(0.0);
}

void ConfigureTensorsToDetectionsCalculator(
    const FaceDetectorGraphOptions& tasks_options,
    mediapipe::TensorsToDetectionsCalculatorOptions* options) {
  // TODO use metadata to configure these fields.
  options->set_num_classes(1);
  options->set_num_boxes(2304);
  options->set_num_coords(16);
  options->set_box_coord_offset(0);
  options->set_keypoint_coord_offset(4);
  options->set_num_keypoints(6);
  options->set_num_values_per_keypoint(2);
  options->set_sigmoid_score(true);
  options->set_score_clipping_thresh(100.0);
  options->set_reverse_output_order(true);
  options->set_min_score_thresh(tasks_options.min_detection_confidence());
  options->set_x_scale(192.0);
  options->set_y_scale(192.0);
  options->set_w_scale(192.0);
  options->set_h_scale(192.0);
}

void ConfigureNonMaxSuppressionCalculator(
    const FaceDetectorGraphOptions& tasks_options,
    mediapipe::NonMaxSuppressionCalculatorOptions* options) {
  options->set_min_suppression_threshold(
      tasks_options.min_suppression_threshold());
  options->set_overlap_type(
      mediapipe::NonMaxSuppressionCalculatorOptions::INTERSECTION_OVER_UNION);
  options->set_algorithm(
      mediapipe::NonMaxSuppressionCalculatorOptions::WEIGHTED);
}

}  // namespace

class FaceDetectorGraph : public core::ModelTaskGraph {
 public:
  absl::StatusOr<CalculatorGraphConfig> GetConfig(
      SubgraphContext* sc) override {
    ASSIGN_OR_RETURN(const auto* model_resources,
                     CreateModelResources<FaceDetectorGraphOptions>(sc));
    Graph graph;
    ASSIGN_OR_RETURN(auto face_detections,
                     BuildFaceDetectionSubgraph(
                         sc->Options<FaceDetectorGraphOptions>(),
                         *model_resources, graph[Input<Image>(kImageTag)],
                         graph[Input<NormalizedRect>(kNormRectTag)], graph));
    face_detections >> graph[Output<std::vector<Detection>>(kDetectionsTag)];
    return graph.GetConfig();
  }

 private:
  absl::StatusOr<Source<std::vector<Detection>>> BuildFaceDetectionSubgraph(
      const FaceDetectorGraphOptions& subgraph_options,
      const core::ModelResources& model_resources, Source<Image> image_in,
      Source<NormalizedRect> norm_rect_in, Graph& graph) {
    // Image preprocessing subgraph to convert image to tensor for the tflite
    // model.
    auto& preprocessing = graph.AddNode(
        "mediapipe.tasks.components.processors.ImagePreprocessingGraph");
    bool use_gpu =
        components::processors::DetermineImagePreprocessingGpuBackend(
            subgraph_options.base_options().acceleration());
    MP_RETURN_IF_ERROR(components::processors::ConfigureImagePreprocessingGraph(
        model_resources, use_gpu,
        &preprocessing.GetOptions<
            components::processors::proto::ImagePreprocessingGraphOptions>()));
    auto& image_to_tensor_options =
        *preprocessing
             .GetOptions<components::processors::proto::
                             ImagePreprocessingGraphOptions>()
             .mutable_image_to_tensor_options();
    image_to_tensor_options.set_keep_aspect_ratio(true);
    image_to_tensor_options.set_border_mode(
        mediapipe::ImageToTensorCalculatorOptions::BORDER_ZERO);
    image_in >> preprocessing.In("IMAGE");
    norm_rect_in >> preprocessing.In("NORM_RECT");
    auto preprocessed_tensors = preprocessing.Out("TENSORS");
    auto matrix = preprocessing.Out("MATRIX");

    // Face detection model inferece.
    auto& inference = AddInference(
        model_resources, subgraph_options.base_options().acceleration(), graph);
    preprocessed_tensors >> inference.In("TENSORS");
    auto model_output_tensors =
        inference.Out("TENSORS").Cast<std::vector<Tensor>>();

    // Generates a single side packet containing a vector of SSD anchors.
    auto& ssd_anchor = graph.AddNode("SsdAnchorsCalculator");
    ConfigureSsdAnchorsCalculator(
        &ssd_anchor.GetOptions<mediapipe::SsdAnchorsCalculatorOptions>());
    auto anchors = ssd_anchor.SideOut("");

    // Converts output tensors to Detections.
    auto& tensors_to_detections =
        graph.AddNode("TensorsToDetectionsCalculator");
    ConfigureTensorsToDetectionsCalculator(
        subgraph_options,
        &tensors_to_detections
             .GetOptions<mediapipe::TensorsToDetectionsCalculatorOptions>());
    model_output_tensors >> tensors_to_detections.In("TENSORS");
    anchors >> tensors_to_detections.SideIn("ANCHORS");
    auto detections = tensors_to_detections.Out("DETECTIONS");

    // Non maximum suppression removes redundant face detections.
    auto& non_maximum_suppression =
        graph.AddNode("NonMaxSuppressionCalculator");
    ConfigureNonMaxSuppressionCalculator(
        subgraph_options,
        &non_maximum_suppression
             .GetOptions<mediapipe::NonMaxSuppressionCalculatorOptions>());
    detections >> non_maximum_suppression.In("");
    auto nms_detections = non_maximum_suppression.Out("");

    // Projects detections back into the input image coordinates system.
    auto& detection_projection = graph.AddNode("DetectionProjectionCalculator");
    nms_detections >> detection_projection.In("DETECTIONS");
    matrix >> detection_projection.In("PROJECTION_MATRIX");
    auto face_detections =
        detection_projection[Output<std::vector<Detection>>("DETECTIONS")];

    return {face_detections};
  }
};

REGISTER_MEDIAPIPE_GRAPH(
    ::mediapipe::tasks::vision::face_detector::FaceDetectorGraph);

}  // namespace face_detector
}  // namespace vision
}  // namespace tasks
}  // namespace mediapipe
