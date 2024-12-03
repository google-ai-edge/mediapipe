/* Copyright 2023 The MediaPipe Authors.

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

#include <array>
#include <optional>
#include <utility>
#include <vector>

#include "absl/status/statusor.h"
#include "mediapipe/calculators/tensor/image_to_tensor_calculator.pb.h"
#include "mediapipe/calculators/tensor/tensors_to_landmarks_calculator.pb.h"
#include "mediapipe/framework/api2/builder.h"
#include "mediapipe/framework/api2/stream/detections_to_rects.h"
#include "mediapipe/framework/api2/stream/landmarks_projection.h"
#include "mediapipe/framework/api2/stream/landmarks_to_detection.h"
#include "mediapipe/framework/api2/stream/rect_transformation.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/formats/image.h"
#include "mediapipe/framework/formats/rect.pb.h"
#include "mediapipe/framework/formats/tensor.h"
#include "mediapipe/framework/port/status_macros.h"
#include "mediapipe/tasks/cc/components/processors/image_preprocessing_graph.h"
#include "mediapipe/tasks/cc/components/processors/proto/image_preprocessing_graph_options.pb.h"
#include "mediapipe/tasks/cc/core/model_task_graph.h"
#include "mediapipe/tasks/cc/vision/hand_landmarker/proto/hand_roi_refinement_graph_options.pb.h"
#include "mediapipe/tasks/cc/vision/utils/image_tensor_specs.h"

namespace mediapipe {
namespace tasks {
namespace vision {
namespace hand_landmarker {

using ::mediapipe::api2::builder::ConvertAlignmentPointsDetectionToRect;
using ::mediapipe::api2::builder::ConvertLandmarksToDetection;
using ::mediapipe::api2::builder::Graph;
using ::mediapipe::api2::builder::ProjectLandmarks;
using ::mediapipe::api2::builder::ScaleAndShiftAndMakeSquareLong;
using ::mediapipe::api2::builder::Stream;

// Refine the input hand RoI with hand_roi_refinement model.
//
// Inputs:
//   IMAGE - Image
//     The image to preprocess.
//   NORM_RECT - NormalizedRect
//     Coarse RoI of hand.
// Outputs:
//   NORM_RECT - NormalizedRect
//     Refined RoI of hand.
class HandRoiRefinementGraph : public core::ModelTaskGraph {
 public:
  absl::StatusOr<CalculatorGraphConfig> GetConfig(
      mediapipe::SubgraphContext* context) override {
    Graph graph;
    Stream<Image> image_in = graph.In("IMAGE").Cast<Image>();
    Stream<NormalizedRect> roi_in =
        graph.In("NORM_RECT").Cast<NormalizedRect>();

    auto& graph_options =
        *context->MutableOptions<proto::HandRoiRefinementGraphOptions>();

    MP_ASSIGN_OR_RETURN(
        const auto* model_resources,
        GetOrCreateModelResources<proto::HandRoiRefinementGraphOptions>(
            context));

    auto& preprocessing = graph.AddNode(
        "mediapipe.tasks.components.processors.ImagePreprocessingGraph");
    bool use_gpu =
        components::processors::DetermineImagePreprocessingGpuBackend(
            graph_options.base_options().acceleration());
    auto& image_to_tensor_options =
        *preprocessing
             .GetOptions<tasks::components::processors::proto::
                             ImagePreprocessingGraphOptions>()
             .mutable_image_to_tensor_options();
    image_to_tensor_options.set_keep_aspect_ratio(true);
    image_to_tensor_options.set_border_mode(
        mediapipe::ImageToTensorCalculatorOptions::BORDER_REPLICATE);
    MP_RETURN_IF_ERROR(components::processors::ConfigureImagePreprocessingGraph(
        *model_resources, use_gpu, graph_options.base_options().gpu_origin(),
        &preprocessing.GetOptions<tasks::components::processors::proto::
                                      ImagePreprocessingGraphOptions>()));
    image_in >> preprocessing.In("IMAGE");
    roi_in >> preprocessing.In("NORM_RECT");
    auto tensors_in = preprocessing.Out("TENSORS");
    auto matrix = preprocessing.Out("MATRIX").Cast<std::array<float, 16>>();
    auto image_size =
        preprocessing.Out("IMAGE_SIZE").Cast<std::pair<int, int>>();

    auto& inference = AddInference(
        *model_resources, graph_options.base_options().acceleration(), graph);
    tensors_in >> inference.In("TENSORS");
    auto tensors_out = inference.Out("TENSORS").Cast<std::vector<Tensor>>();

    MP_ASSIGN_OR_RETURN(auto image_tensor_specs,
                        BuildInputImageTensorSpecs(*model_resources));

    // Convert tensors to landmarks. Recrop model outputs two points,
    // center point and guide point.
    auto& to_landmarks = graph.AddNode("TensorsToLandmarksCalculator");
    auto& to_landmarks_opts =
        to_landmarks
            .GetOptions<mediapipe::TensorsToLandmarksCalculatorOptions>();
    to_landmarks_opts.set_num_landmarks(/*num_landmarks=*/2);
    to_landmarks_opts.set_input_image_width(image_tensor_specs.image_width);
    to_landmarks_opts.set_input_image_height(image_tensor_specs.image_height);
    to_landmarks_opts.set_normalize_z(/*z_norm_factor=*/1.0f);
    tensors_out.ConnectTo(to_landmarks.In("TENSORS"));
    auto recrop_landmarks = to_landmarks.Out("NORM_LANDMARKS")
                                .Cast<mediapipe::NormalizedLandmarkList>();

    // Project landmarks.
    auto projected_recrop_landmarks =
        ProjectLandmarks(recrop_landmarks, matrix, graph);

    // Convert re-crop landmarks to detection.
    auto recrop_detection =
        ConvertLandmarksToDetection(projected_recrop_landmarks, graph);

    // Convert re-crop detection to rect.
    auto recrop_rect = ConvertAlignmentPointsDetectionToRect(
        recrop_detection, image_size, /*start_keypoint_index=*/0,
        /*end_keypoint_index=*/1, /*target_angle=*/-90, graph);

    auto refined_roi =
        ScaleAndShiftAndMakeSquareLong(recrop_rect, image_size,
                                       /*scale_x_factor=*/1.0,
                                       /*scale_y_factor=*/1.0, /*shift_x=*/0,
                                       /*shift_y=*/-0.1, graph);
    refined_roi >> graph.Out("NORM_RECT").Cast<NormalizedRect>();
    return graph.GetConfig();
  }
};

REGISTER_MEDIAPIPE_GRAPH(
    ::mediapipe::tasks::vision::hand_landmarker::HandRoiRefinementGraph);

}  // namespace hand_landmarker
}  // namespace vision
}  // namespace tasks
}  // namespace mediapipe
