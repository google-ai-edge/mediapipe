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

#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "mediapipe/framework/api2/builder.h"
#include "mediapipe/framework/api2/port.h"
#include "mediapipe/framework/calculator.pb.h"
#include "mediapipe/framework/formats/detection.pb.h"
#include "mediapipe/framework/formats/image.h"
#include "mediapipe/framework/formats/rect.pb.h"
#include "mediapipe/framework/formats/tensor.h"
#include "mediapipe/tasks/cc/common.h"
#include "mediapipe/tasks/cc/components/processors/detection_postprocessing_graph.h"
#include "mediapipe/tasks/cc/components/processors/image_preprocessing_graph.h"
#include "mediapipe/tasks/cc/components/processors/proto/detection_postprocessing_graph_options.pb.h"
#include "mediapipe/tasks/cc/components/processors/proto/detector_options.pb.h"
#include "mediapipe/tasks/cc/core/model_resources.h"
#include "mediapipe/tasks/cc/core/model_task_graph.h"
#include "mediapipe/tasks/cc/core/proto/inference_subgraph.pb.h"
#include "mediapipe/tasks/cc/vision/object_detector/proto/object_detector_options.pb.h"
#include "mediapipe/tasks/metadata/metadata_schema_generated.h"

namespace mediapipe {
namespace tasks {
namespace vision {

namespace {

using ::mediapipe::NormalizedRect;
using ::mediapipe::api2::Input;
using ::mediapipe::api2::Output;
using ::mediapipe::api2::builder::Graph;
using ::mediapipe::api2::builder::Source;
using ObjectDetectorOptionsProto =
    object_detector::proto::ObjectDetectorOptions;
using TensorsSource =
    mediapipe::api2::builder::Source<std::vector<mediapipe::Tensor>>;

constexpr char kDetectionsTag[] = "DETECTIONS";
constexpr char kImageSizeTag[] = "IMAGE_SIZE";
constexpr char kImageTag[] = "IMAGE";
constexpr char kMatrixTag[] = "MATRIX";
constexpr char kNormRectTag[] = "NORM_RECT";
constexpr char kPixelDetectionsTag[] = "PIXEL_DETECTIONS";
constexpr char kProjectionMatrixTag[] = "PROJECTION_MATRIX";
constexpr char kTensorTag[] = "TENSORS";

// Struct holding the different output streams produced by the object detection
// subgraph.
struct ObjectDetectionOutputStreams {
  Source<std::vector<Detection>> detections;
  Source<Image> image;
};

absl::Status SanityCheckOptions(const ObjectDetectorOptionsProto& options) {
  if (options.max_results() == 0) {
    return CreateStatusWithPayload(
        absl::StatusCode::kInvalidArgument,
        "Invalid `max_results` option: value must be != 0",
        MediaPipeTasksStatus::kInvalidArgumentError);
  }
  if (options.category_allowlist_size() > 0 &&
      options.category_denylist_size() > 0) {
    return CreateStatusWithPayload(
        absl::StatusCode::kInvalidArgument,
        "`category_allowlist` and `category_denylist` are mutually "
        "exclusive options.",
        MediaPipeTasksStatus::kInvalidArgumentError);
  }
  return absl::OkStatus();
}

}  // namespace

// A "mediapipe.tasks.vision.ObjectDetectorGraph" performs object detection.
// - Accepts CPU input images and outputs detections on CPU.
//
// Inputs:
//   IMAGE - Image
//     Image to perform detection on.
//   NORM_RECT - NormalizedRect @Optional
//     Describes image rotation and region of image to perform detection
//     on.
//     @Optional: rect covering the whole image is used if not specified.
//
// Outputs:
//   DETECTIONS - std::vector<Detection>
//     Detected objects with bounding box in pixel units.
//   IMAGE - mediapipe::Image
//     The image that object detection runs on.
// All returned coordinates are in the unrotated and uncropped input image
// coordinates system.
//
// Example:
// node {
//   calculator: "mediapipe.tasks.vision.ObjectDetectorGraph"
//   input_stream: "IMAGE:image_in"
//   output_stream: "DETECTIONS:detections_out"
//   output_stream: "IMAGE:image_out"
//   options {
//     [mediapipe.tasks.vision.object_detector.proto.ObjectDetectorOptions.ext]
//     {
//       base_options {
//         model_asset {
//           file_name: "/path/to/model.tflite"
//         }
//       }
//       max_results: 4
//       score_threshold: 0.5
//       category_allowlist: "foo"
//       category_allowlist: "bar"
//     }
//   }
// }
class ObjectDetectorGraph : public core::ModelTaskGraph {
 public:
  absl::StatusOr<CalculatorGraphConfig> GetConfig(
      SubgraphContext* sc) override {
    MP_ASSIGN_OR_RETURN(const auto* model_resources,
                        CreateModelResources<ObjectDetectorOptionsProto>(sc));
    Graph graph;
    MP_ASSIGN_OR_RETURN(
        auto output_streams,
        BuildObjectDetectionTask(
            sc->Options<ObjectDetectorOptionsProto>(), *model_resources,
            graph[Input<Image>(kImageTag)],
            graph[Input<NormalizedRect>::Optional(kNormRectTag)], graph));
    output_streams.detections >>
        graph[Output<std::vector<Detection>>(kDetectionsTag)];
    output_streams.image >> graph[Output<Image>(kImageTag)];
    return graph.GetConfig();
  }

 private:
  // Adds a mediapipe object detection task graph into the provided
  // builder::Graph instance. The object detection task takes images
  // (mediapipe::Image) as the input and returns two output streams:
  //   - the detection results (std::vector<Detection>),
  //   - the processed image that has pixel data stored on the target storage
  //     (mediapipe::Image).
  //
  // task_options: the mediapipe tasks ObjectDetectorOptions proto.
  // model_resources: the ModelSources object initialized from an object
  // detection model file with model metadata.
  // image_in: (mediapipe::Image) stream to run object detection on.
  // graph: the mediapipe builder::Graph instance to be updated.
  absl::StatusOr<ObjectDetectionOutputStreams> BuildObjectDetectionTask(
      const ObjectDetectorOptionsProto& task_options,
      const core::ModelResources& model_resources, Source<Image> image_in,
      Source<NormalizedRect> norm_rect_in, Graph& graph) {
    MP_RETURN_IF_ERROR(SanityCheckOptions(task_options));
    auto& model = *model_resources.GetTfLiteModel();
    if (model.subgraphs()->size() != 1) {
      return CreateStatusWithPayload(
          absl::StatusCode::kInvalidArgument,
          absl::StrFormat("Expected a model with a single subgraph, found %d.",
                          model.subgraphs()->size()),
          MediaPipeTasksStatus::kInvalidArgumentError);
    }
    // Checks that metadata is available.
    auto* metadata_extractor = model_resources.GetMetadataExtractor();
    if (metadata_extractor->GetModelMetadata() == nullptr ||
        metadata_extractor->GetModelMetadata()->subgraph_metadata() ==
            nullptr) {
      return CreateStatusWithPayload(
          absl::StatusCode::kInvalidArgument,
          "Object detection models require TFLite Model Metadata but none was "
          "found",
          MediaPipeTasksStatus::kMetadataNotFoundError);
    }

    // Adds preprocessing calculators and connects them to the graph input image
    // stream.
    auto& preprocessing = graph.AddNode(
        "mediapipe.tasks.components.processors.ImagePreprocessingGraph");
    bool use_gpu =
        components::processors::DetermineImagePreprocessingGpuBackend(
            task_options.base_options().acceleration());
    MP_RETURN_IF_ERROR(components::processors::ConfigureImagePreprocessingGraph(
        model_resources, use_gpu, task_options.base_options().gpu_origin(),
        &preprocessing.GetOptions<tasks::components::processors::proto::
                                      ImagePreprocessingGraphOptions>()));
    image_in >> preprocessing.In(kImageTag);
    norm_rect_in >> preprocessing.In(kNormRectTag);

    // Adds inference subgraph and connects its input stream to the output
    // tensors produced by the ImageToTensorCalculator.
    auto& inference = AddInference(
        model_resources, task_options.base_options().acceleration(), graph);
    preprocessing.Out(kTensorTag) >> inference.In(kTensorTag);
    TensorsSource model_output_tensors =
        inference.Out(kTensorTag).Cast<std::vector<Tensor>>();

    // Add Detection postprocessing graph to convert tensors to detections.
    auto& postprocessing = graph.AddNode(
        "mediapipe.tasks.components.processors.DetectionPostprocessingGraph");
    components::processors::proto::DetectorOptions detector_options;
    detector_options.set_max_results(task_options.max_results());
    detector_options.set_score_threshold(task_options.score_threshold());
    detector_options.set_display_names_locale(
        task_options.display_names_locale());
    detector_options.mutable_category_allowlist()->CopyFrom(
        task_options.category_allowlist());
    detector_options.mutable_category_denylist()->CopyFrom(
        task_options.category_denylist());
    detector_options.set_multiclass_nms(task_options.multiclass_nms());
    detector_options.set_min_suppression_threshold(
        task_options.min_suppression_threshold());
    MP_RETURN_IF_ERROR(
        components::processors::ConfigureDetectionPostprocessingGraph(
            model_resources, detector_options,
            postprocessing
                .GetOptions<components::processors::proto::
                                DetectionPostprocessingGraphOptions>()));
    model_output_tensors >> postprocessing.In(kTensorTag);
    auto detections = postprocessing.Out(kDetectionsTag);

    // Calculator to projects detections back to the original coordinate system.
    auto& detection_projection = graph.AddNode("DetectionProjectionCalculator");
    detections >> detection_projection.In(kDetectionsTag);
    preprocessing.Out(kMatrixTag) >>
        detection_projection.In(kProjectionMatrixTag);

    // Calculator to convert relative detection bounding boxes to pixel
    // detection bounding boxes.
    auto& detection_transformation =
        graph.AddNode("DetectionTransformationCalculator");
    detection_projection.Out(kDetectionsTag) >>
        detection_transformation.In(kDetectionsTag);
    preprocessing.Out(kImageSizeTag) >>
        detection_transformation.In(kImageSizeTag);
    auto detections_in_pixel =
        detection_transformation.Out(kPixelDetectionsTag);

    // Deduplicate Detections with same bounding box coordinates.
    auto& detections_deduplicate =
        graph.AddNode("DetectionsDeduplicateCalculator");
    detections_in_pixel >> detections_deduplicate.In("");

    // Outputs the labeled detections and the processed image as the subgraph
    // output streams.
    return {{
        /* detections= */
        detections_deduplicate[Output<std::vector<Detection>>("")],
        /* image= */ preprocessing[Output<Image>(kImageTag)],
    }};
  }
};

REGISTER_MEDIAPIPE_GRAPH(::mediapipe::tasks::vision::ObjectDetectorGraph);

}  // namespace vision
}  // namespace tasks
}  // namespace mediapipe
