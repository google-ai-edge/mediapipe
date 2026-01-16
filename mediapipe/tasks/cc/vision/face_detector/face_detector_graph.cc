/* Copyright 2025 The MediaPipe Authors.

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

#include <string>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "mediapipe/calculators/core/clip_vector_size_calculator.pb.h"
#include "mediapipe/calculators/tensor/image_to_tensor_calculator.pb.h"
#include "mediapipe/calculators/tensor/tensors_to_detections_calculator.pb.h"
#include "mediapipe/calculators/tflite/ssd_anchors_calculator.pb.h"
#include "mediapipe/calculators/util/detections_to_rects_calculator.pb.h"
#include "mediapipe/calculators/util/non_max_suppression_calculator.pb.h"
#include "mediapipe/calculators/util/rect_transformation_calculator.pb.h"
#include "mediapipe/framework/api2/builder.h"
#include "mediapipe/framework/api2/port.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/formats/detection.pb.h"
#include "mediapipe/framework/formats/image.h"
#include "mediapipe/framework/formats/rect.pb.h"
#include "mediapipe/framework/formats/tensor.h"
#include "mediapipe/framework/port/ret_check.h"
#include "mediapipe/framework/port/status_macros.h"
#include "mediapipe/tasks/cc/components/processors/image_preprocessing_graph.h"
#include "mediapipe/tasks/cc/core/model_resources.h"
#include "mediapipe/tasks/cc/core/model_task_graph.h"
#include "mediapipe/tasks/cc/core/proto/inference_subgraph.pb.h"
#include "mediapipe/tasks/cc/metadata/metadata_extractor.h"
#include "mediapipe/tasks/cc/vision/face_detector/proto/face_detector_graph_options.pb.h"
#include "mediapipe/tasks/cc/vision/utils/image_tensor_specs.h"
#include "mediapipe/tasks/metadata/face_detector_metadata_schema_generated.h"

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
using ::mediapipe::tasks::vision::BuildInputImageTensorSpecs;
using ::mediapipe::tasks::vision::face_detector::proto::
    FaceDetectorGraphOptions;

namespace {
constexpr char kImageTag[] = "IMAGE";
constexpr char kNormRectTag[] = "NORM_RECT";
constexpr char kDetectionsTag[] = "DETECTIONS";
constexpr char kAnchorsTag[] = "ANCHORS";
constexpr char kImageSizeTag[] = "IMAGE_SIZE";
constexpr char kNormRectsTag[] = "NORM_RECTS";
constexpr char kProjectionMatrixTag[] = "PROJECTION_MATRIX";
constexpr char kTensorsTag[] = "TENSORS";
constexpr char kMatrixTag[] = "MATRIX";
constexpr char kFaceRectsTag[] = "FACE_RECTS";
constexpr char kExpandedFaceRectsTag[] = "EXPANDED_FACE_RECTS";
constexpr char kPixelDetectionsTag[] = "PIXEL_DETECTIONS";
constexpr char kDetectorMetadataName[] = "FACE_DETECTOR_METADATA";

struct FaceDetectionOuts {
  Source<std::vector<Detection>> face_detections;
  Source<std::vector<NormalizedRect>> face_rects;
  Source<std::vector<NormalizedRect>> expanded_face_rects;
  Source<Image> image;
};

// Get FaceDetectorOptions from the model metadata extractor, if not found,
// return nullptr.
const FaceDetectorOptions* GetFaceDetectorOptionsFromMetadata(
    const core::ModelResources& model_resources) {
  const auto* metadata_extractor = model_resources.GetMetadataExtractor();
  if (metadata_extractor->GetCustomMetadataList() != nullptr &&
      metadata_extractor->GetCustomMetadataList()->size() > 0) {
    for (const auto* custom_metadata :
         *metadata_extractor->GetCustomMetadataList()) {
      if (custom_metadata->name()->str() == kDetectorMetadataName) {
        return GetFaceDetectorOptions(custom_metadata->data()->data());
      }
    }
  }
  return nullptr;
}

absl::Status ConfigureSsdAnchorsCalculator(
    const FaceDetectorOptions* face_detector_options, const int image_width,
    const int image_height, mediapipe::SsdAnchorsCalculatorOptions* options) {
  if (face_detector_options != nullptr) {
    // For models with metadata.
    const auto* ssd_anchors_options = face_detector_options->anchor_config();
    RET_CHECK(ssd_anchors_options);
    options->set_num_layers(ssd_anchors_options->num_layers());
    options->set_min_scale(ssd_anchors_options->min_scale());
    options->set_max_scale(ssd_anchors_options->max_scale());
    options->set_input_size_height(image_height);
    options->set_input_size_width(image_width);
    options->set_anchor_offset_x(ssd_anchors_options->anchor_offset_x());
    options->set_anchor_offset_y(ssd_anchors_options->anchor_offset_y());
    for (int i = 0; i < ssd_anchors_options->strides()->size(); ++i) {
      options->add_strides(ssd_anchors_options->strides()->Get(i));
    }
    for (int i = 0; i < ssd_anchors_options->aspect_ratios()->size(); ++i) {
      options->add_aspect_ratios(ssd_anchors_options->aspect_ratios()->Get(i));
    }
    options->set_fixed_anchor_size(ssd_anchors_options->fixed_anchor_size());
    options->set_interpolated_scale_aspect_ratio(
        ssd_anchors_options->interpolated_scale_aspect_ratio());
  } else {
    // Default settings for legacy model without metadata.
    options->set_num_layers(4);
    options->set_min_scale(0.1484375);
    options->set_max_scale(0.75);
    options->set_input_size_height(128);
    options->set_input_size_width(128);
    options->set_anchor_offset_x(0.5);
    options->set_anchor_offset_y(0.5);
    options->add_strides(8);
    options->add_strides(16);
    options->add_strides(16);
    options->add_strides(16);
    options->add_aspect_ratios(1.0);
    options->set_fixed_anchor_size(true);
    options->set_interpolated_scale_aspect_ratio(1.0);
  }
  return absl::OkStatus();
}

absl::Status ConfigureTensorsToDetectionsCalculator(
    const FaceDetectorOptions* face_detector_options,
    const FaceDetectorGraphOptions& tasks_options,
    mediapipe::TensorsToDetectionsCalculatorOptions* options) {
  if (face_detector_options != nullptr) {
    // For models with metadata.
    const auto* tensors_decoding_options =
        face_detector_options->tensors_decoding_config();
    RET_CHECK(tensors_decoding_options);
    options->set_num_classes(tensors_decoding_options->num_classes());
    options->set_num_boxes(tensors_decoding_options->num_boxes());
    options->set_num_coords(tensors_decoding_options->num_coords());
    options->set_box_coord_offset(tensors_decoding_options->box_coord_offset());
    options->set_keypoint_coord_offset(
        tensors_decoding_options->keypoint_coord_offset());
    options->set_num_keypoints(tensors_decoding_options->num_keypoints());
    options->set_num_values_per_keypoint(
        tensors_decoding_options->num_values_per_keypoint());
    options->set_x_scale(tensors_decoding_options->x_scale());
    options->set_y_scale(tensors_decoding_options->y_scale());
    options->set_w_scale(tensors_decoding_options->w_scale());
    options->set_h_scale(tensors_decoding_options->h_scale());
    options->set_min_score_thresh(tasks_options.min_detection_confidence());
    options->set_sigmoid_score(tensors_decoding_options->sigmoid_score());
    options->set_score_clipping_thresh(
        tensors_decoding_options->score_clipping_thresh());
    options->set_reverse_output_order(
        tensors_decoding_options->reverse_output_order());
  } else {
    // Default settings for legacy model without metadata.
    options->set_num_classes(1);
    options->set_num_boxes(896);
    options->set_num_coords(16);
    options->set_box_coord_offset(0);
    options->set_keypoint_coord_offset(4);
    options->set_num_keypoints(6);
    options->set_num_values_per_keypoint(2);
    options->set_sigmoid_score(true);
    options->set_score_clipping_thresh(100.0);
    options->set_reverse_output_order(true);
    options->set_min_score_thresh(tasks_options.min_detection_confidence());
    options->set_x_scale(128.0);
    options->set_y_scale(128.0);
    options->set_w_scale(128.0);
    options->set_h_scale(128.0);
  }
  return absl::OkStatus();
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

void ConfigureDetectionsToRectsCalculator(
    mediapipe::DetectionsToRectsCalculatorOptions* options) {
  // Left eye from the observer’s point of view.
  options->set_rotation_vector_start_keypoint_index(0);
  // Right eye from the observer’s point of view.
  options->set_rotation_vector_end_keypoint_index(1);
  options->set_rotation_vector_target_angle_degrees(0);
}

void ConfigureRectTransformationCalculator(
    mediapipe::RectTransformationCalculatorOptions* options) {
  options->set_scale_x(1.5);
  options->set_scale_y(1.5);
}

}  // namespace

// A "mediapipe.tasks.vision.face_detector.FaceDetectorGraph" performs face
// detection.
//
// Inputs:
//   IMAGE - Image
//     Image to perform detection on.
//   NORM_RECT - NormalizedRect @Optional
//     Describes image rotation and region of image to perform detection on. If
//     not provided, whole image is used for face detection.
//
// Outputs:
//   DETECTIONS - std::vector<Detection>
//     Detected face with maximum `num_faces` specified in options.
//   FACE_RECTS - std::vector<NormalizedRect>
//     Detected face bounding boxes in normalized coordinates.
//   EXPANDED_FACE_RECTS - std::vector<NormalizedRect>
//     Expanded face bounding boxes in normalized coordinates so that bounding
//     boxes likely contain the whole face. This is usually used as RoI for face
//     landmarks detection to run on.
//   IMAGE - Image
//     The input image that the face detector runs on and has the pixel data
//     stored on the target storage (CPU vs GPU).
// All returned coordinates are in the unrotated and uncropped input image
// coordinates system.
//
// Example:
// node {
//   calculator: "mediapipe.tasks.vision.face_detector.FaceDetectorGraph"
//   input_stream: "IMAGE:image"
//   input_stream: "NORM_RECT:norm_rect"
//   output_stream: "DETECTIONS:palm_detections"
//   output_stream: "FACE_RECTS:face_rects"
//   output_stream: "EXPANDED_FACE_RECTS:expanded_face_rects"
//   output_stream: "IMAGE:image_out"
//   options {
//     [mediapipe.tasks.vision.face_detector.proto.FaceDetectorGraphOptions.ext]
//     {
//       base_options {
//          model_asset {
//            file_name: "face_detection.tflite"
//          }
//       }
//       min_detection_confidence: 0.5
//       num_faces: 2
//     }
//   }
// }
class FaceDetectorGraph : public core::ModelTaskGraph {
 public:
  absl::StatusOr<CalculatorGraphConfig> GetConfig(
      SubgraphContext* sc) override {
    MP_ASSIGN_OR_RETURN(const auto* model_resources,
                        CreateModelResources<FaceDetectorGraphOptions>(sc));
    Graph graph;
    MP_ASSIGN_OR_RETURN(FaceDetectionOuts outs,
                        BuildFaceDetectionSubgraph(
                            sc->Options<FaceDetectorGraphOptions>(),
                            *model_resources, graph[Input<Image>(kImageTag)],
                            graph[Input<NormalizedRect>(kNormRectTag)], graph));
    outs.face_detections >>
        graph.Out(kDetectionsTag).Cast<std::vector<Detection>>();
    outs.face_rects >>
        graph.Out(kFaceRectsTag).Cast<std::vector<NormalizedRect>>();
    outs.expanded_face_rects >>
        graph.Out(kExpandedFaceRectsTag).Cast<std::vector<NormalizedRect>>();
    outs.image >> graph.Out(kImageTag).Cast<Image>();

    return graph.GetConfig();
  }

 private:
  std::string GetImagePreprocessingGraphName() {
    return "mediapipe.tasks.components.processors.ImagePreprocessingGraph";
  }

  absl::StatusOr<FaceDetectionOuts> BuildFaceDetectionSubgraph(
      const FaceDetectorGraphOptions& subgraph_options,
      const core::ModelResources& model_resources, Source<Image> image_in,
      Source<NormalizedRect> norm_rect_in, Graph& graph) {
    // Prepare face detector options and image tensor specs from model.
    const FaceDetectorOptions* face_detector_options =
        GetFaceDetectorOptionsFromMetadata(model_resources);
    MP_ASSIGN_OR_RETURN(ImageTensorSpecs input_specs,
                        BuildInputImageTensorSpecs(model_resources));

    // Image preprocessing subgraph to convert image to tensor for the tflite
    // model.
    auto& preprocessing = graph.AddNode(GetImagePreprocessingGraphName());
    bool use_gpu =
        components::processors::DetermineImagePreprocessingGpuBackend(
            subgraph_options.base_options().acceleration());
    MP_RETURN_IF_ERROR(components::processors::ConfigureImagePreprocessingGraph(
        model_resources, use_gpu, subgraph_options.base_options().gpu_origin(),
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
    image_in >> preprocessing.In(kImageTag);
    norm_rect_in >> preprocessing.In(kNormRectTag);
    auto preprocessed_tensors = preprocessing.Out(kTensorsTag);
    auto matrix = preprocessing.Out(kMatrixTag);
    auto image_size = preprocessing.Out(kImageSizeTag);

    // Face detection model inference.
    auto& inference = AddInference(
        model_resources, subgraph_options.base_options().acceleration(), graph);
    preprocessed_tensors >> inference.In(kTensorsTag);
    auto model_output_tensors =
        inference.Out(kTensorsTag).Cast<std::vector<Tensor>>();

    // Generates a single side packet containing a vector of SSD anchors.
    auto& ssd_anchor = graph.AddNode("SsdAnchorsCalculator");
    MP_RETURN_IF_ERROR(ConfigureSsdAnchorsCalculator(
        face_detector_options, input_specs.image_width,
        input_specs.image_height,
        &ssd_anchor.GetOptions<mediapipe::SsdAnchorsCalculatorOptions>()));
    auto anchors = ssd_anchor.SideOut("");

    // Converts output tensors to Detections.
    auto& tensors_to_detections =
        graph.AddNode("TensorsToDetectionsCalculator");
    MP_RETURN_IF_ERROR(ConfigureTensorsToDetectionsCalculator(
        face_detector_options, subgraph_options,
        &tensors_to_detections
             .GetOptions<mediapipe::TensorsToDetectionsCalculatorOptions>()));
    model_output_tensors >> tensors_to_detections.In(kTensorsTag);
    anchors >> tensors_to_detections.SideIn(kAnchorsTag);
    auto detections = tensors_to_detections.Out(kDetectionsTag);

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
    nms_detections >> detection_projection.In(kDetectionsTag);
    matrix >> detection_projection.In(kProjectionMatrixTag);
    Source<std::vector<Detection>> face_detections =
        detection_projection.Out(kDetectionsTag).Cast<std::vector<Detection>>();

    if (subgraph_options.has_num_faces()) {
      // Clip face detections to maximum number of faces;
      auto& clip_detection_vector_size =
          graph.AddNode("ClipDetectionVectorSizeCalculator");
      clip_detection_vector_size
          .GetOptions<mediapipe::ClipVectorSizeCalculatorOptions>()
          .set_max_vec_size(subgraph_options.num_faces());
      face_detections >> clip_detection_vector_size.In("");
      face_detections =
          clip_detection_vector_size.Out("").Cast<std::vector<Detection>>();
    }

    // Converts results of face detection into a rectangle (normalized by image
    // size) that encloses the face and is rotated such that the line connecting
    // left eye and right eye is aligned with the X-axis of the rectangle.
    auto& detections_to_rects = graph.AddNode("DetectionsToRectsCalculator");
    ConfigureDetectionsToRectsCalculator(
        &detections_to_rects
             .GetOptions<mediapipe::DetectionsToRectsCalculatorOptions>());
    image_size >> detections_to_rects.In(kImageSizeTag);
    face_detections >> detections_to_rects.In(kDetectionsTag);
    auto face_rects = detections_to_rects.Out(kNormRectsTag)
                          .Cast<std::vector<NormalizedRect>>();

    // Expands and shifts the rectangle that contains the face so that it's
    // likely to cover the entire face.
    auto& rect_transformation = graph.AddNode("RectTransformationCalculator");
    ConfigureRectTransformationCalculator(
        &rect_transformation
             .GetOptions<mediapipe::RectTransformationCalculatorOptions>());
    face_rects >> rect_transformation.In(kNormRectsTag);
    image_size >> rect_transformation.In(kImageSizeTag);
    auto expanded_face_rects =
        rect_transformation.Out("").Cast<std::vector<NormalizedRect>>();

    // Calculator to convert relative detection bounding boxes to pixel
    // detection bounding boxes.
    auto& detection_transformation =
        graph.AddNode("DetectionTransformationCalculator");
    face_detections >> detection_transformation.In(kDetectionsTag);
    image_size >> detection_transformation.In(kImageSizeTag);
    auto face_pixel_detections =
        detection_transformation.Out(kPixelDetectionsTag)
            .Cast<std::vector<Detection>>();

    return FaceDetectionOuts{
        /* face_detections= */ face_pixel_detections,
        /* face_rects= */ face_rects,
        /* expanded_face_rects= */ expanded_face_rects,
        /* image= */ preprocessing.Out("IMAGE").Cast<Image>()};
  }
};

REGISTER_MEDIAPIPE_GRAPH(
    ::mediapipe::tasks::vision::face_detector::FaceDetectorGraph)

}  // namespace face_detector
}  // namespace vision
}  // namespace tasks
}  // namespace mediapipe
