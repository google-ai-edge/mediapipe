/* Copyright 2022 The MediaPipe Authors.

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

#include <memory>
#include <optional>
#include <type_traits>
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
#include "mediapipe/tasks/cc/vision/hand_detector/proto/hand_detector_graph_options.pb.h"
#include "mediapipe/tasks/cc/vision/utils/image_tensor_specs.h"

namespace mediapipe {
namespace tasks {
namespace vision {
namespace hand_detector {

namespace {

using ::mediapipe::NormalizedRect;
using ::mediapipe::api2::Input;
using ::mediapipe::api2::Output;
using ::mediapipe::api2::builder::Graph;
using ::mediapipe::api2::builder::Source;
using ::mediapipe::tasks::vision::hand_detector::proto::
    HandDetectorGraphOptions;

constexpr char kImageTag[] = "IMAGE";
constexpr char kNormRectTag[] = "NORM_RECT";
constexpr char kPalmDetectionsTag[] = "PALM_DETECTIONS";
constexpr char kHandRectsTag[] = "HAND_RECTS";
constexpr char kPalmRectsTag[] = "PALM_RECTS";

struct HandDetectionOuts {
  Source<std::vector<Detection>> palm_detections;
  Source<std::vector<NormalizedRect>> hand_rects;
  Source<std::vector<NormalizedRect>> palm_rects;
  Source<Image> image;
};

void ConfigureTensorsToDetectionsCalculator(
    const HandDetectorGraphOptions& tasks_options,
    mediapipe::TensorsToDetectionsCalculatorOptions* options) {
  // TODO use metadata to configure these fields.
  options->set_num_classes(1);
  options->set_num_boxes(2016);
  options->set_num_coords(18);
  options->set_box_coord_offset(0);
  options->set_keypoint_coord_offset(4);
  options->set_num_keypoints(7);
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
    mediapipe::NonMaxSuppressionCalculatorOptions* options) {
  options->set_min_suppression_threshold(0.3);
  options->set_overlap_type(
      mediapipe::NonMaxSuppressionCalculatorOptions::INTERSECTION_OVER_UNION);
  options->set_algorithm(
      mediapipe::NonMaxSuppressionCalculatorOptions::WEIGHTED);
  // TODO "return_empty_detections" was removed from 1P graph,
  // consider setting it from metadata accordingly.
  options->set_return_empty_detections(true);
}

void ConfigureSsdAnchorsCalculator(
    mediapipe::SsdAnchorsCalculatorOptions* options) {
  // TODO config SSD anchors parameters from metadata.
  options->set_num_layers(4);
  options->set_min_scale(0.1484375);
  options->set_max_scale(0.75);
  options->set_input_size_height(192);
  options->set_input_size_width(192);
  options->set_anchor_offset_x(0.5);
  options->set_anchor_offset_y(0.5);
  options->add_strides(8);
  options->add_strides(16);
  options->add_strides(16);
  options->add_strides(16);
  options->add_aspect_ratios(1.0);
  options->set_fixed_anchor_size(true);
}

void ConfigureDetectionsToRectsCalculator(
    mediapipe::DetectionsToRectsCalculatorOptions* options) {
  // Center of wrist.
  options->set_rotation_vector_start_keypoint_index(0);
  // MCP of middle finger.
  options->set_rotation_vector_end_keypoint_index(2);
  options->set_rotation_vector_target_angle(90);
  options->set_output_zero_rect_for_empty_detections(true);
}

void ConfigureRectTransformationCalculator(
    mediapipe::RectTransformationCalculatorOptions* options) {
  options->set_scale_x(2.6);
  options->set_scale_y(2.6);
  options->set_shift_y(-0.5);
  options->set_square_long(true);
}

}  // namespace

// A "mediapipe.tasks.vision.hand_detector.HandDetectorGraph" performs hand
// detection. The Hand Detection Graph is based on palm detection model, and
// scale the detected palm bounding box to enclose the detected whole hand.
// Accepts CPU input images and outputs Landmark on CPU.
//
// Inputs:
//   IMAGE - Image
//     Image to perform detection on.
//   NORM_RECT - NormalizedRect @Optional
//     Describes image rotation and region of image to perform detection on. If
//     not provided, whole image is used for hand detection.
//
// Outputs:
//   PALM_DETECTIONS - std::vector<Detection>
//     Detected palms with maximum `num_hands` specified in options.
//   HAND_RECTS - std::vector<NormalizedRect>
//     Detected hand bounding boxes in normalized coordinates.
//   PLAM_RECTS - std::vector<NormalizedRect>
//     Detected palm bounding boxes in normalized coordinates.
//   IMAGE - Image
//     The input image that the hand detector runs on and has the pixel data
//     stored on the target storage (CPU vs GPU).
// All returned coordinates are in the unrotated and uncropped input image
// coordinates system.
//
// Example:
// node {
//   calculator: "mediapipe.tasks.vision.hand_detector.HandDetectorGraph"
//   input_stream: "IMAGE:image"
//   input_stream: "NORM_RECT:norm_rect"
//   output_stream: "PALM_DETECTIONS:palm_detections"
//   output_stream: "HAND_RECTS:hand_rects_from_palm_detections"
//   output_stream: "PALM_RECTS:palm_rects"
//   output_stream: "IMAGE:image_out"
//   options {
//     [mediapipe.tasks.vision.hand_detector.proto.HandDetectorGraphOptions.ext]
//     {
//       base_options {
//          model_asset {
//            file_name: "palm_detection.tflite"
//          }
//       }
//       min_detection_confidence: 0.5
//       num_hands: 2
//     }
//   }
// }
// TODO Decouple detection part and rects part.
class HandDetectorGraph : public core::ModelTaskGraph {
 public:
  absl::StatusOr<CalculatorGraphConfig> GetConfig(
      SubgraphContext* sc) override {
    MP_ASSIGN_OR_RETURN(const auto* model_resources,
                        CreateModelResources<HandDetectorGraphOptions>(sc));
    Graph graph;
    MP_ASSIGN_OR_RETURN(
        auto hand_detection_outs,
        BuildHandDetectionSubgraph(
            sc->Options<HandDetectorGraphOptions>(), *model_resources,
            graph[Input<Image>(kImageTag)],
            graph[Input<NormalizedRect>::Optional(kNormRectTag)], graph));
    hand_detection_outs.palm_detections >>
        graph[Output<std::vector<Detection>>(kPalmDetectionsTag)];
    hand_detection_outs.hand_rects >>
        graph[Output<std::vector<NormalizedRect>>(kHandRectsTag)];
    hand_detection_outs.palm_rects >>
        graph[Output<std::vector<NormalizedRect>>(kPalmRectsTag)];
    hand_detection_outs.image >> graph[Output<Image>(kImageTag)];
    return graph.GetConfig();
  }

 private:
  // Updates graph to perform hand detection. Returns palm detections and
  // corresponding hand RoI rects.
  //
  // subgraph_options: the mediapipe tasks module HandDetectionOptions.
  // model_resources: the ModelSources object initialized from an hand detection
  // model file with model metadata.
  // image_in: image stream to run hand detection on.
  // graph: the mediapipe builder::Graph instance to be updated.
  absl::StatusOr<HandDetectionOuts> BuildHandDetectionSubgraph(
      const HandDetectorGraphOptions& subgraph_options,
      const core::ModelResources& model_resources, Source<Image> image_in,
      Source<NormalizedRect> norm_rect_in, Graph& graph) {
    // Add image preprocessing subgraph. The model expects aspect ratio
    // unchanged.
    auto& preprocessing = graph.AddNode(
        "mediapipe.tasks.components.processors.ImagePreprocessingGraph");
    auto& image_to_tensor_options =
        *preprocessing
             .GetOptions<components::processors::proto::
                             ImagePreprocessingGraphOptions>()
             .mutable_image_to_tensor_options();
    image_to_tensor_options.set_keep_aspect_ratio(true);
    image_to_tensor_options.set_border_mode(
        mediapipe::ImageToTensorCalculatorOptions::BORDER_ZERO);
    bool use_gpu =
        components::processors::DetermineImagePreprocessingGpuBackend(
            subgraph_options.base_options().acceleration());
    MP_RETURN_IF_ERROR(components::processors::ConfigureImagePreprocessingGraph(
        model_resources, use_gpu, subgraph_options.base_options().gpu_origin(),
        &preprocessing.GetOptions<
            components::processors::proto::ImagePreprocessingGraphOptions>()));
    image_in >> preprocessing.In("IMAGE");
    norm_rect_in >> preprocessing.In("NORM_RECT");
    auto preprocessed_tensors = preprocessing.Out("TENSORS");
    auto matrix = preprocessing.Out("MATRIX");
    auto image_size = preprocessing.Out("IMAGE_SIZE");

    // Adds SSD palm detection model.
    auto& inference = AddInference(
        model_resources, subgraph_options.base_options().acceleration(), graph);
    preprocessed_tensors >> inference.In("TENSORS");
    auto model_output_tensors = inference.Out("TENSORS");

    // TODO: support hand detection metadata.
    bool has_metadata = false;

    // Generates a single side packet containing a vector of SSD anchors.
    auto& ssd_anchor = graph.AddNode("SsdAnchorsCalculator");
    auto& ssd_anchor_options =
        ssd_anchor.GetOptions<mediapipe::SsdAnchorsCalculatorOptions>();
    if (!has_metadata) {
      ConfigureSsdAnchorsCalculator(&ssd_anchor_options);
    }
    auto anchors = ssd_anchor.SideOut("");

    // Converts output tensors to Detections.
    auto& tensors_to_detections =
        graph.AddNode("TensorsToDetectionsCalculator");
    if (!has_metadata) {
      ConfigureTensorsToDetectionsCalculator(
          subgraph_options,
          &tensors_to_detections
               .GetOptions<mediapipe::TensorsToDetectionsCalculatorOptions>());
    }

    model_output_tensors >> tensors_to_detections.In("TENSORS");
    anchors >> tensors_to_detections.SideIn("ANCHORS");
    auto detections = tensors_to_detections.Out("DETECTIONS");

    // Non maximum suppression removes redundant palm detections.
    auto& non_maximum_suppression =
        graph.AddNode("NonMaxSuppressionCalculator");
    ConfigureNonMaxSuppressionCalculator(
        &non_maximum_suppression
             .GetOptions<mediapipe::NonMaxSuppressionCalculatorOptions>());
    detections >> non_maximum_suppression.In("");
    auto nms_detections = non_maximum_suppression.Out("");

    // Maps detection label IDs to the corresponding label text "Palm".
    auto& detection_label_id_to_text =
        graph.AddNode("DetectionLabelIdToTextCalculator");
    detection_label_id_to_text
        .GetOptions<mediapipe::DetectionLabelIdToTextCalculatorOptions>()
        .add_label("Palm");
    nms_detections >> detection_label_id_to_text.In("");
    auto detections_with_text = detection_label_id_to_text.Out("");

    // Projects detections back into the input image coordinates system.
    auto& detection_projection = graph.AddNode("DetectionProjectionCalculator");
    detections_with_text >> detection_projection.In("DETECTIONS");
    matrix >> detection_projection.In("PROJECTION_MATRIX");
    auto palm_detections =
        detection_projection[Output<std::vector<Detection>>("DETECTIONS")];

    // Converts each palm detection into a rectangle (normalized by image size)
    // that encloses the palm and is rotated such that the line connecting
    // center of the wrist and MCP of the middle finger is aligned with the
    // Y-axis of the rectangle.
    auto& detections_to_rects = graph.AddNode("DetectionsToRectsCalculator");
    ConfigureDetectionsToRectsCalculator(
        &detections_to_rects
             .GetOptions<mediapipe::DetectionsToRectsCalculatorOptions>());
    palm_detections >> detections_to_rects.In("DETECTIONS");
    image_size >> detections_to_rects.In("IMAGE_SIZE");
    auto palm_rects =
        detections_to_rects[Output<std::vector<NormalizedRect>>("NORM_RECTS")];

    // Expands and shifts the rectangle that contains the palm so that it's
    //  likely to cover the entire hand.
    auto& rect_transformation = graph.AddNode("RectTransformationCalculator");
    ConfigureRectTransformationCalculator(
        &rect_transformation
             .GetOptions<mediapipe::RectTransformationCalculatorOptions>());
    palm_rects >> rect_transformation.In("NORM_RECTS");
    image_size >> rect_transformation.In("IMAGE_SIZE");
    auto hand_rects = rect_transformation.Out("");

    // Clips the size of the input vector to the provided max_vec_size. This
    // determines the maximum number of hand instances this graph outputs.
    // Note that the performance gain of clipping detections earlier in this
    // graph is minimal because NMS will minimize overlapping detections and the
    // number of detections isn't expected to exceed 5-10.
    auto& clip_normalized_rect_vector_size =
        graph.AddNode("ClipNormalizedRectVectorSizeCalculator");
    clip_normalized_rect_vector_size
        .GetOptions<mediapipe::ClipVectorSizeCalculatorOptions>()
        .set_max_vec_size(subgraph_options.num_hands());
    hand_rects >> clip_normalized_rect_vector_size.In("");
    auto clipped_hand_rects =
        clip_normalized_rect_vector_size[Output<std::vector<NormalizedRect>>(
            "")];

    return HandDetectionOuts{
        /* palm_detections= */ palm_detections,
        /* hand_rects= */ clipped_hand_rects,
        /* palm_rects= */ palm_rects,
        /* image= */ preprocessing[Output<Image>(kImageTag)]};
  }
};

REGISTER_MEDIAPIPE_GRAPH(
    ::mediapipe::tasks::vision::hand_detector::HandDetectorGraph);

}  // namespace hand_detector
}  // namespace vision
}  // namespace tasks
}  // namespace mediapipe
