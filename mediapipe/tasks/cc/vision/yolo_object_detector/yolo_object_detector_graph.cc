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

#include <string>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "mediapipe/calculators/yolo/auto_nms_calculator.pb.h"
#include "mediapipe/framework/api2/builder.h"
#include "mediapipe/framework/api2/port.h"
#include "mediapipe/framework/formats/detection.pb.h"
#include "mediapipe/framework/formats/image.h"
#include "mediapipe/framework/formats/rect.pb.h"
#include "mediapipe/framework/formats/tensor.h"
#include "mediapipe/tasks/cc/common.h"
#include "mediapipe/tasks/cc/components/processors/image_preprocessing_graph.h"
#include "mediapipe/tasks/cc/components/processors/proto/image_preprocessing_graph_options.pb.h"
#include "mediapipe/tasks/cc/components/processors/proto/yolo_tensors_to_detections_calculator.pb.h"
#include "mediapipe/tasks/cc/core/model_resources.h"
#include "mediapipe/tasks/cc/core/model_task_graph.h"
#include "mediapipe/tasks/cc/vision/yolo_object_detector/proto/yolo_object_detector_options.pb.h"
#include "mediapipe/tasks/cc/vision/yolo_object_detector/yolo_object_detector_graph_utils.h"

namespace mediapipe {
namespace tasks {
namespace vision {

namespace {

using ::mediapipe::NormalizedRect;
using ::mediapipe::api2::Input;
using ::mediapipe::api2::Output;
using ::mediapipe::api2::builder::Graph;
using ::mediapipe::api2::builder::Source;
using YoloObjectDetectorOptionsProto =
    yolo_object_detector::proto::YoloObjectDetectorOptions;
using YoloTensorsOptionsProto =
    mediapipe::tasks::components::processors::proto::
        YoloTensorsToDetectionsCalculatorOptions;

constexpr char kDetectionsTag[]          = "DETECTIONS";
constexpr char kImageTag[]               = "IMAGE";
constexpr char kImageSizeTag[]           = "IMAGE_SIZE";
constexpr char kMatrixTag[]              = "MATRIX";
constexpr char kNormRectTag[]            = "NORM_RECT";
constexpr char kProjectionMatrixTag[]    = "PROJECTION_MATRIX";
constexpr char kPixelDetectionsTag[]     = "PIXEL_DETECTIONS";
constexpr char kTensorsTag[]             = "TENSORS";

struct ObjectDetectionOutputStreams {
  Source<std::vector<Detection>> detections;
  Source<Image> image;
};

YoloTensorsOptionsProto::TensorLayout ConvertTensorLayout(
    YoloObjectDetectorOptionsProto::TensorLayout layout) {
  switch (layout) {
    case YoloObjectDetectorOptionsProto::BOXES_FIRST:
      return YoloTensorsOptionsProto::BOXES_FIRST;
    case YoloObjectDetectorOptionsProto::FEATURES_FIRST:
      return YoloTensorsOptionsProto::FEATURES_FIRST;
    default:
      return YoloTensorsOptionsProto::TENSOR_LAYOUT_AUTO;
  }
}

YoloTensorsOptionsProto::DecodeMode ConvertDecodeMode(
    YoloObjectDetectorOptionsProto::DecodeMode mode) {
  switch (mode) {
    case YoloObjectDetectorOptionsProto::ULTRALYTICS_DETECTION_HEAD:
      return YoloTensorsOptionsProto::ULTRALYTICS_DETECTION_HEAD;
    case YoloObjectDetectorOptionsProto::END_TO_END:
      return YoloTensorsOptionsProto::END_TO_END;
    default:
      return YoloTensorsOptionsProto::DECODE_MODE_AUTO;
  }
}

// Resolve postprocess_mode for AutoNmsCalculator at graph build time.
// If task options specify explicit mode, use it.
// Otherwise read the model output shape to infer END_TO_END vs ULTRALYTICS.
AutoNmsCalculatorOptions::PostprocessMode ResolvePostprocessMode(
    const YoloObjectDetectorOptionsProto& task_options,
    const std::string& model_path) {
  if (task_options.postprocess_mode() ==
      YoloObjectDetectorOptionsProto::SKIP_NMS)
    return AutoNmsCalculatorOptions::SKIP_NMS;
  if (task_options.postprocess_mode() ==
      YoloObjectDetectorOptionsProto::APPLY_NMS)
    return AutoNmsCalculatorOptions::APPLY_NMS;
  if (task_options.decode_mode() ==
      YoloObjectDetectorOptionsProto::END_TO_END)
    return AutoNmsCalculatorOptions::SKIP_NMS;
  if (task_options.decode_mode() ==
      YoloObjectDetectorOptionsProto::ULTRALYTICS_DETECTION_HEAD)
    return AutoNmsCalculatorOptions::APPLY_NMS;

  // AUTO: try to infer from model flatbuffer.
  auto dims_or = yolo_object_detector::ExtractModelOutputDims(model_path);
  if (dims_or.ok()) {
    using yolo_object_detector::InferDecodeMode;
    using yolo_object_detector::YoloDecodeMode;
    if (InferDecodeMode(*dims_or) == YoloDecodeMode::kEndToEnd)
      return AutoNmsCalculatorOptions::SKIP_NMS;
  }
  // Conservative fallback: always run NMS.
  return AutoNmsCalculatorOptions::APPLY_NMS;
}

}  // namespace

class YoloObjectDetectorGraph : public core::ModelTaskGraph {
 public:
  absl::StatusOr<CalculatorGraphConfig> GetConfig(
      SubgraphContext* sc) override {
    MP_ASSIGN_OR_RETURN(
        const auto* model_resources,
        CreateModelResources<YoloObjectDetectorOptionsProto>(sc));
    Graph graph;
    MP_ASSIGN_OR_RETURN(
        auto output_streams,
        BuildObjectDetectionTask(
            sc->Options<YoloObjectDetectorOptionsProto>(), *model_resources,
            graph[Input<Image>(kImageTag)],
            graph[Input<NormalizedRect>::Optional(kNormRectTag)], graph));
    output_streams.detections >>
        graph[Output<std::vector<Detection>>(kDetectionsTag)];
    output_streams.image >> graph[Output<Image>(kImageTag)];
    return graph.GetConfig();
  }

 private:
  absl::StatusOr<ObjectDetectionOutputStreams> BuildObjectDetectionTask(
      const YoloObjectDetectorOptionsProto& task_options,
      const core::ModelResources& model_resources,
      Source<Image> image_in,
      Source<NormalizedRect> norm_rect_in,
      Graph& graph) {
    // ── Image preprocessing ──────────────────────────────────────────────
    auto& preprocessing = graph.AddNode(
        "mediapipe.tasks.components.processors.ImagePreprocessingGraph");
    const bool use_gpu =
        components::processors::DetermineImagePreprocessingGpuBackend(
            task_options.base_options().acceleration());
    auto& preproc_opts = preprocessing.GetOptions<
        tasks::components::processors::proto::
            ImagePreprocessingGraphOptions>();
    MP_RETURN_IF_ERROR(
        components::processors::ConfigureImagePreprocessingGraph(
            model_resources, use_gpu,
            task_options.base_options().gpu_origin(), &preproc_opts));
    image_in >> preprocessing.In(kImageTag);
    norm_rect_in >> preprocessing.In(kNormRectTag);

    // ── TFLite inference ─────────────────────────────────────────────────
    // AddInference uses InferenceCalculator (new Tensor format).
    // MODEL_METADATA side packet from TfLiteInferenceCalculator (Task 2)
    // is available when that calculator is used directly; here we resolve
    // NMS behavior at graph build time instead (see ResolvePostprocessMode).
    auto& inference =
        AddInference(model_resources,
                     task_options.base_options().acceleration(), graph);
    preprocessing.Out(kTensorsTag) >> inference.In(kTensorsTag);
    auto model_output_tensors =
        inference.Out(kTensorsTag).Cast<std::vector<Tensor>>();

    // ── YOLO tensor decode ────────────────────────────────────────────────
    // Get model input H×W for coordinate normalization.
    const std::string model_path =
        task_options.base_options().model_asset().file_name();
    MP_ASSIGN_OR_RETURN(auto [input_w, input_h],
                        yolo_object_detector::ExtractModelInputShape(model_path));

    auto& yolo_decode = graph.AddNode("YoloTensorsToDetectionsCalculator");
    auto& yolo_opts =
        yolo_decode.GetOptions<YoloTensorsOptionsProto>();
    yolo_opts.set_input_width(input_w);
    yolo_opts.set_input_height(input_h);
    yolo_opts.set_tensor_layout(
        ConvertTensorLayout(task_options.tensor_layout()));
    yolo_opts.set_decode_mode(
        ConvertDecodeMode(task_options.decode_mode()));
    if (task_options.has_score_threshold())
      yolo_opts.set_min_score_threshold(task_options.score_threshold());

    model_output_tensors >> yolo_decode.In(kTensorsTag);
    auto detections =
        yolo_decode.Out(kDetectionsTag).Cast<std::vector<Detection>>();

    // ── AutoNmsCalculator ─────────────────────────────────────────────────
    auto& auto_nms = graph.AddNode("AutoNmsCalculator");
    auto& nms_opts = auto_nms.GetOptions<AutoNmsCalculatorOptions>();
    nms_opts.set_iou_threshold(task_options.min_suppression_threshold());
    nms_opts.set_postprocess_mode(
        ResolvePostprocessMode(task_options, model_path));
    detections >> auto_nms.In(kDetectionsTag);
    detections = auto_nms.Out(kDetectionsTag).Cast<std::vector<Detection>>();

    // ── Coordinate projection back to original image space ────────────────
    auto& detection_projection =
        graph.AddNode("DetectionProjectionCalculator");
    detections >> detection_projection.In(kDetectionsTag);
    preprocessing.Out(kMatrixTag) >>
        detection_projection.In(kProjectionMatrixTag);

    auto& detection_transformation =
        graph.AddNode("DetectionTransformationCalculator");
    detection_projection.Out(kDetectionsTag) >>
        detection_transformation.In(kDetectionsTag);
    preprocessing.Out(kImageSizeTag) >>
        detection_transformation.In(kImageSizeTag);
    auto detections_in_pixel =
        detection_transformation.Out(kPixelDetectionsTag)
            .Cast<std::vector<Detection>>();

    return {{
        /* detections= */ detections_in_pixel,
        /* image= */ preprocessing[Output<Image>(kImageTag)],
    }};
  }
};

REGISTER_MEDIAPIPE_GRAPH(
    ::mediapipe::tasks::vision::YoloObjectDetectorGraph);

}  // namespace vision
}  // namespace tasks
}  // namespace mediapipe
