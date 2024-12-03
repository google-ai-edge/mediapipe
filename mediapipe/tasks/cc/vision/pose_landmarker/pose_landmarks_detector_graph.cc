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

#include <optional>
#include <utility>
#include <vector>

#include "absl/status/statusor.h"
#include "mediapipe/calculators/core/constant_side_packet_calculator.pb.h"
#include "mediapipe/calculators/core/split_vector_calculator.pb.h"
#include "mediapipe/calculators/image/warp_affine_calculator.pb.h"
#include "mediapipe/calculators/tensor/image_to_tensor_calculator.pb.h"
#include "mediapipe/calculators/tensor/tensors_to_landmarks_calculator.pb.h"
#include "mediapipe/calculators/tensor/tensors_to_segmentation_calculator.pb.h"
#include "mediapipe/calculators/util/detections_to_rects_calculator.pb.h"
#include "mediapipe/calculators/util/rect_transformation_calculator.pb.h"
#include "mediapipe/calculators/util/refine_landmarks_from_heatmap_calculator.pb.h"
#include "mediapipe/calculators/util/thresholding_calculator.pb.h"
#include "mediapipe/calculators/util/visibility_copy_calculator.pb.h"
#include "mediapipe/framework/api2/builder.h"
#include "mediapipe/framework/api2/port.h"
#include "mediapipe/framework/api2/stream/get_vector_item.h"
#include "mediapipe/framework/api2/stream/image_size.h"
#include "mediapipe/framework/api2/stream/smoothing.h"
#include "mediapipe/framework/formats/image.h"
#include "mediapipe/framework/formats/landmark.pb.h"
#include "mediapipe/framework/formats/rect.pb.h"
#include "mediapipe/framework/subgraph.h"
#include "mediapipe/gpu/gpu_origin.pb.h"
#include "mediapipe/tasks/cc/common.h"
#include "mediapipe/tasks/cc/components/processors/image_preprocessing_graph.h"
#include "mediapipe/tasks/cc/core/model_resources.h"
#include "mediapipe/tasks/cc/core/model_task_graph.h"
#include "mediapipe/tasks/cc/vision/pose_landmarker/proto/pose_landmarks_detector_graph_options.pb.h"
#include "mediapipe/tasks/cc/vision/utils/image_tensor_specs.h"
#include "mediapipe/util/graph_builder_utils.h"

namespace mediapipe {
namespace tasks {
namespace vision {
namespace pose_landmarker {

using ::mediapipe::NormalizedRect;
using ::mediapipe::api2::Input;
using ::mediapipe::api2::Output;
using ::mediapipe::api2::builder::GetImageSize;
using ::mediapipe::api2::builder::Graph;
using ::mediapipe::api2::builder::SmoothLandmarks;
using ::mediapipe::api2::builder::SmoothLandmarksVisibility;
using ::mediapipe::api2::builder::Source;
using ::mediapipe::api2::builder::Stream;
using ::mediapipe::tasks::core::ModelResources;
using ::mediapipe::tasks::vision::pose_landmarker::proto::
    PoseLandmarksDetectorGraphOptions;

constexpr char kImageTag[] = "IMAGE";
constexpr char kNormRectTag[] = "NORM_RECT";
constexpr char kLandmarksTag[] = "LANDMARKS";
constexpr char kNormLandmarksTag[] = "NORM_LANDMARKS";
constexpr char kWorldLandmarksTag[] = "WORLD_LANDMARKS";
constexpr char kAuxLandmarksTag[] = "AUXILIARY_LANDMARKS";
constexpr char kPoseRectNextFrameTag[] = "POSE_RECT_NEXT_FRAME";
constexpr char kPoseRectsNextFrameTag[] = "POSE_RECTS_NEXT_FRAME";
constexpr char kPresenceTag[] = "PRESENCE";
constexpr char kPresenceScoreTag[] = "PRESENCE_SCORE";
constexpr char kSegmentationMaskTag[] = "SEGMENTATION_MASK";
constexpr char kImageSizeTag[] = "IMAGE_SIZE";
constexpr char kLandmarksToTag[] = "LANDMARKS_TO";
constexpr char kTensorsTag[] = "TENSORS";
constexpr char kFloatTag[] = "FLOAT";
constexpr char kFlagTag[] = "FLAG";
constexpr char kMaskTag[] = "MASK";
constexpr char kDetectionTag[] = "DETECTION";
constexpr char kNormLandmarksFromTag[] = "NORM_LANDMARKS_FROM";
constexpr char kBatchEndTag[] = "BATCH_END";
constexpr char kItemTag[] = "ITEM";
constexpr char kIterableTag[] = "ITERABLE";
constexpr char kLetterboxPaddingTag[] = "LETTERBOX_PADDING";
constexpr char kMatrixTag[] = "MATRIX";
constexpr char kOutputSizeTag[] = "OUTPUT_SIZE";

constexpr int kModelOutputTensorSplitNum = 5;
constexpr int kLandmarksNum = 39;
constexpr float kLandmarksNormalizeZ = 0.4;

struct SinglePoseLandmarkerOutputs {
  Source<NormalizedLandmarkList> pose_landmarks;
  Source<LandmarkList> world_pose_landmarks;
  Source<NormalizedLandmarkList> auxiliary_pose_landmarks;
  Source<NormalizedRect> pose_rect_next_frame;
  Source<bool> pose_presence;
  Source<float> pose_presence_score;
  std::optional<Source<Image>> segmentation_mask;
};

struct PoseLandmarkerOutputs {
  Source<std::vector<NormalizedLandmarkList>> landmark_lists;
  Source<std::vector<LandmarkList>> world_landmark_lists;
  Source<std::vector<NormalizedLandmarkList>> auxiliary_landmark_lists;
  Source<std::vector<NormalizedRect>> pose_rects_next_frame;
  Source<std::vector<bool>> presences;
  Source<std::vector<float>> presence_scores;
  std::optional<Source<std::vector<Image>>> segmentation_masks;
};

absl::Status SanityCheckOptions(
    const PoseLandmarksDetectorGraphOptions& options) {
  if (options.min_detection_confidence() < 0 ||
      options.min_detection_confidence() > 1) {
    return CreateStatusWithPayload(absl::StatusCode::kInvalidArgument,
                                   "Invalid `min_detection_confidence` option: "
                                   "value must be in the range [0.0, 1.0]",
                                   MediaPipeTasksStatus::kInvalidArgumentError);
  }
  return absl::OkStatus();
}

// Split pose landmark detection model output tensor into five parts,
// representing landmarks, presence scores, segmentation, heatmap, and world
// landmarks respectively.
void ConfigureSplitTensorVectorCalculator(
    mediapipe::SplitVectorCalculatorOptions* options) {
  for (int i = 0; i < kModelOutputTensorSplitNum; ++i) {
    auto* range = options->add_ranges();
    range->set_begin(i);
    range->set_end(i + 1);
  }
}

void ConfigureTensorsToLandmarksCalculator(
    const ImageTensorSpecs& input_image_tensor_spec, bool normalize,
    bool sigmoid_activation,
    mediapipe::TensorsToLandmarksCalculatorOptions* options) {
  options->set_num_landmarks(kLandmarksNum);
  options->set_input_image_height(input_image_tensor_spec.image_height);
  options->set_input_image_width(input_image_tensor_spec.image_width);

  if (normalize) {
    options->set_normalize_z(kLandmarksNormalizeZ);
  }

  if (sigmoid_activation) {
    options->set_visibility_activation(
        mediapipe::TensorsToLandmarksCalculatorOptions_Activation_SIGMOID);
    options->set_presence_activation(
        mediapipe::TensorsToLandmarksCalculatorOptions_Activation_SIGMOID);
  }
}

void ConfigureTensorsToSegmentationCalculator(
    mediapipe::TensorsToSegmentationCalculatorOptions* options) {
  options->set_activation(
      mediapipe::TensorsToSegmentationCalculatorOptions_Activation_SIGMOID);
  options->set_gpu_origin(mediapipe::GpuOrigin::TOP_LEFT);
}

void ConfigureRefineLandmarksFromHeatmapCalculator(
    mediapipe::RefineLandmarksFromHeatmapCalculatorOptions* options) {
  // Derived from
  // mediapipe/modules/pose_landmark/tensors_to_pose_landmarks_and_segmentation.pbtxt.
  options->set_kernel_size(7);
}

void ConfigureSplitNormalizedLandmarkListCalculator(
    mediapipe::SplitVectorCalculatorOptions* options) {
  // Derived from
  // mediapipe/modules/pose_landmark/tensors_to_pose_landmarks_and_segmentation.pbtxt
  auto* range = options->add_ranges();
  range->set_begin(0);
  range->set_end(33);
  auto* range_2 = options->add_ranges();
  range_2->set_begin(33);
  range_2->set_end(35);
}

void ConfigureSplitLandmarkListCalculator(
    mediapipe::SplitVectorCalculatorOptions* options) {
  // Derived from
  // mediapipe/modules/pose_landmark/tensors_to_pose_landmarks_and_segmentation.pbtxt
  auto* range = options->add_ranges();
  range->set_begin(0);
  range->set_end(33);
}

void ConfigureVisibilityCopyCalculator(
    mediapipe::VisibilityCopyCalculatorOptions* options) {
  // Derived from
  // mediapipe/modules/pose_landmark/tensors_to_pose_landmarks_and_segmentation.pbtxt
  options->set_copy_visibility(true);
  options->set_copy_presence(true);
}

void ConfigureRectTransformationCalculator(
    mediapipe::RectTransformationCalculatorOptions* options) {
  options->set_scale_x(1.25);
  options->set_scale_y(1.25);
  options->set_square_long(true);
}

void ConfigureAlignmentPointsRectsCalculator(
    mediapipe::DetectionsToRectsCalculatorOptions* options) {
  // Derived from
  // mediapipe/modules/pose_landmark/pose_landmarks_to_roi.pbtxt
  options->set_rotation_vector_start_keypoint_index(0);
  options->set_rotation_vector_end_keypoint_index(1);
  options->set_rotation_vector_target_angle_degrees(90);
}

void ConfigureWarpAffineCalculator(
    mediapipe::WarpAffineCalculatorOptions* options) {
  options->set_border_mode(mediapipe::WarpAffineCalculatorOptions::BORDER_ZERO);
  options->set_gpu_origin(mediapipe::GpuOrigin::TOP_LEFT);
}

template <typename TickT>
Stream<int> CreateIntConstantStream(Stream<TickT> tick_stream, int constant_int,
                                    Graph& graph) {
  auto& constant_side_packet_node =
      graph.AddNode("ConstantSidePacketCalculator");
  constant_side_packet_node
      .GetOptions<mediapipe::ConstantSidePacketCalculatorOptions>()
      .add_packet()
      ->set_int_value(constant_int);
  auto side_packet = constant_side_packet_node.SideOut("PACKET");

  auto& side_packet_to_stream = graph.AddNode("SidePacketToStreamCalculator");
  tick_stream.ConnectTo(side_packet_to_stream.In("TICK"));
  side_packet.ConnectTo(side_packet_to_stream.SideIn(""));
  return side_packet_to_stream.Out("AT_TICK").Cast<int>();
}

// A "mediapipe.tasks.vision.pose_landmarker.SinglePoseLandmarksDetectorGraph"
// performs pose landmarks detection.
// - Accepts CPU input images and outputs Landmark on CPU.
//
// Inputs:
//   IMAGE - Image
//     Image to perform detection on.
//   NORM_RECT - NormalizedRect @Optional
//     Rect enclosing the RoI to perform detection on. If not set, the detection
//     RoI is the whole image.
//
//
// Outputs:
//   LANDMARKS: - NormalizedLandmarkList
//     Detected pose landmarks.
//   WORLD_LANDMARKS - LandmarkList
//     Detected pose landmarks in world coordinates.
//   AUXILIARY_LANDMARKS - NormalizedLandmarkList
//     Detected pose auxiliary landmarks.
//   POSE_RECT_NEXT_FRAME - NormalizedRect
//     The predicted Rect enclosing the pose RoI for landmark detection on the
//     next frame.
//   PRESENCE - bool
//     Boolean value indicates whether the pose is present.
//   PRESENCE_SCORE - float
//     Float value indicates the probability that the pose is present.
//   SEGMENTATION_MASK - Image
//     Segmentation mask for pose.
//
// Example:
// node {
//   calculator:
//   "mediapipe.tasks.vision.pose_landmarker.SingleposeLandmarksDetectorGraph"
//   input_stream: "IMAGE:input_image"
//   input_stream: "POSE_RECT:pose_rect"
//   output_stream: "LANDMARKS:pose_landmarks"
//   output_stream: "WORLD_LANDMARKS:world_pose_landmarks"
//   output_stream: "AUXILIARY_LANDMARKS:auxiliary_landmarks"
//   output_stream: "POSE_RECT_NEXT_FRAME:pose_rect_next_frame"
//   output_stream: "PRESENCE:pose_presence"
//   output_stream: "PRESENCE_SCORE:pose_presence_score"
//   output_stream: "SEGMENTATION_MASK:segmentation_mask"
//   options {
//     [mediapipe.tasks.vision.pose_landmarker.proto.poseLandmarksDetectorGraphOptions.ext]
//     {
//       base_options {
//          model_asset {
//            file_name: "pose_landmark_lite.tflite"
//          }
//       }
//       min_detection_confidence: 0.5
//     }
//   }
// }
class SinglePoseLandmarksDetectorGraph : public core::ModelTaskGraph {
 public:
  absl::StatusOr<CalculatorGraphConfig> GetConfig(
      SubgraphContext* sc) override {
    bool output_segmentation_mask =
        HasOutput(sc->OriginalNode(), kSegmentationMaskTag);
    MP_ASSIGN_OR_RETURN(
        const auto* model_resources,
        CreateModelResources<PoseLandmarksDetectorGraphOptions>(sc));
    Graph graph;
    MP_ASSIGN_OR_RETURN(
        auto pose_landmark_detection_outs,
        BuildSinglePoseLandmarksDetectorGraph(
            sc->Options<PoseLandmarksDetectorGraphOptions>(), *model_resources,
            graph[Input<Image>(kImageTag)],
            graph[Input<NormalizedRect>::Optional(kNormRectTag)], graph,
            output_segmentation_mask));
    pose_landmark_detection_outs.pose_landmarks >>
        graph[Output<NormalizedLandmarkList>(kLandmarksTag)];
    pose_landmark_detection_outs.world_pose_landmarks >>
        graph[Output<LandmarkList>(kWorldLandmarksTag)];
    pose_landmark_detection_outs.auxiliary_pose_landmarks >>
        graph[Output<NormalizedLandmarkList>(kAuxLandmarksTag)];
    pose_landmark_detection_outs.pose_rect_next_frame >>
        graph[Output<NormalizedRect>(kPoseRectNextFrameTag)];
    pose_landmark_detection_outs.pose_presence >>
        graph[Output<bool>(kPresenceTag)];
    pose_landmark_detection_outs.pose_presence_score >>
        graph[Output<float>(kPresenceScoreTag)];
    if (pose_landmark_detection_outs.segmentation_mask) {
      *pose_landmark_detection_outs.segmentation_mask >>
          graph[Output<Image>(kSegmentationMaskTag)];
    }

    return graph.GetConfig();
  }

 private:
  absl::StatusOr<SinglePoseLandmarkerOutputs>
  BuildSinglePoseLandmarksDetectorGraph(
      const PoseLandmarksDetectorGraphOptions& subgraph_options,
      const ModelResources& model_resources, Source<Image> image_in,
      Source<NormalizedRect> pose_rect, Graph& graph,
      bool output_segmentation_mask) {
    MP_RETURN_IF_ERROR(SanityCheckOptions(subgraph_options));

    auto& preprocessing = graph.AddNode(
        "mediapipe.tasks.components.processors.ImagePreprocessingGraph");
    bool use_gpu =
        components::processors::DetermineImagePreprocessingGpuBackend(
            subgraph_options.base_options().acceleration());
    MP_RETURN_IF_ERROR(components::processors::ConfigureImagePreprocessingGraph(
        model_resources, use_gpu, subgraph_options.base_options().gpu_origin(),
        &preprocessing.GetOptions<tasks::components::processors::proto::
                                      ImagePreprocessingGraphOptions>()));
    image_in >> preprocessing.In(kImageTag);
    pose_rect >> preprocessing.In(kNormRectTag);
    auto image_size = preprocessing[Output<std::pair<int, int>>(kImageSizeTag)];
    auto matrix = preprocessing[Output<std::vector<float>>(kMatrixTag)];
    auto letterbox_padding = preprocessing.Out(kLetterboxPaddingTag);

    MP_ASSIGN_OR_RETURN(auto image_tensor_specs,
                        BuildInputImageTensorSpecs(model_resources));

    auto& inference = AddInference(
        model_resources, subgraph_options.base_options().acceleration(), graph);
    preprocessing.Out(kTensorsTag) >> inference.In(kTensorsTag);

    // Split model output tensors to multiple streams.
    auto& split_tensors_vector = graph.AddNode("SplitTensorVectorCalculator");
    ConfigureSplitTensorVectorCalculator(
        &split_tensors_vector
             .GetOptions<mediapipe::SplitVectorCalculatorOptions>());
    inference.Out(kTensorsTag) >> split_tensors_vector.In("");
    auto landmark_tensors = split_tensors_vector.Out(0);
    auto pose_flag_tensors = split_tensors_vector.Out(1);
    auto segmentation_tensors = split_tensors_vector.Out(2);
    auto heatmap_tensors = split_tensors_vector.Out(3);
    auto world_landmark_tensors = split_tensors_vector.Out(4);

    // Converts the pose-flag tensor into a float that represents the confidence
    // score of pose presence.
    auto& tensors_to_pose_presence = graph.AddNode("TensorsToFloatsCalculator");
    pose_flag_tensors >> tensors_to_pose_presence.In(kTensorsTag);
    auto pose_presence_score =
        tensors_to_pose_presence[Output<float>(kFloatTag)];

    // Applies a threshold to the confidence score to determine whether a
    // pose is present.
    auto& pose_presence_thresholding = graph.AddNode("ThresholdingCalculator");
    pose_presence_thresholding
        .GetOptions<mediapipe::ThresholdingCalculatorOptions>()
        .set_threshold(subgraph_options.min_detection_confidence());
    pose_presence_score >> pose_presence_thresholding.In(kFloatTag);
    auto pose_presence = pose_presence_thresholding[Output<bool>(kFlagTag)];

    // GateCalculator for tensors.
    auto& tensors_gate = graph.AddNode("GateCalculator");
    landmark_tensors >> tensors_gate.In("")[0];
    segmentation_tensors >> tensors_gate.In("")[1];
    heatmap_tensors >> tensors_gate.In("")[2];
    world_landmark_tensors >> tensors_gate.In("")[3];
    pose_presence >> tensors_gate.In("ALLOW");
    auto ensured_landmarks_tensors = tensors_gate.Out(0);
    auto ensured_segmentation_tensors = tensors_gate.Out(1);
    auto ensured_heatmap_tensors = tensors_gate.Out(2);
    auto ensured_world_landmark_tensors = tensors_gate.Out(3);

    // Decodes the landmark tensors into a list of landmarks, where the landmark
    // coordinates are normalized by the size of the input image to the model.
    auto& tensors_to_landmarks = graph.AddNode("TensorsToLandmarksCalculator");
    ConfigureTensorsToLandmarksCalculator(
        image_tensor_specs, /* normalize = */ false,
        /*sigmoid_activation= */ true,
        &tensors_to_landmarks
             .GetOptions<mediapipe::TensorsToLandmarksCalculatorOptions>());
    ensured_landmarks_tensors >> tensors_to_landmarks.In(kTensorsTag);

    auto raw_landmarks =
        tensors_to_landmarks[Output<NormalizedLandmarkList>(kNormLandmarksTag)];

    // Refines landmarks with the heatmap tensor.
    auto& refine_landmarks_from_heatmap =
        graph.AddNode("RefineLandmarksFromHeatmapCalculator");
    ConfigureRefineLandmarksFromHeatmapCalculator(
        &refine_landmarks_from_heatmap.GetOptions<
            mediapipe::RefineLandmarksFromHeatmapCalculatorOptions>());
    ensured_heatmap_tensors >> refine_landmarks_from_heatmap.In(kTensorsTag);
    raw_landmarks >> refine_landmarks_from_heatmap.In(kNormLandmarksTag);
    auto landmarks_from_heatmap =
        refine_landmarks_from_heatmap[Output<NormalizedLandmarkList>(
            kNormLandmarksTag)];

    // Splits the landmarks into two sets: the actual pose landmarks and the
    // auxiliary landmarks.
    auto& split_normalized_landmark_list =
        graph.AddNode("SplitNormalizedLandmarkListCalculator");
    ConfigureSplitNormalizedLandmarkListCalculator(
        &split_normalized_landmark_list
             .GetOptions<mediapipe::SplitVectorCalculatorOptions>());
    landmarks_from_heatmap >> split_normalized_landmark_list.In("");
    auto landmarks = split_normalized_landmark_list.Out("")[0]
                         .Cast<NormalizedLandmarkList>();
    auto auxiliary_landmarks = split_normalized_landmark_list.Out("")[1]
                                   .Cast<NormalizedLandmarkList>();

    // Decodes the world-landmark tensors into a vector of world landmarks.
    auto& tensors_to_world_landmarks =
        graph.AddNode("TensorsToLandmarksCalculator");
    ConfigureTensorsToLandmarksCalculator(
        image_tensor_specs, /* normalize = */ false,
        /* sigmoid_activation= */ false,
        &tensors_to_world_landmarks
             .GetOptions<mediapipe::TensorsToLandmarksCalculatorOptions>());
    ensured_world_landmark_tensors >>
        tensors_to_world_landmarks.In(kTensorsTag);
    auto raw_world_landmarks =
        tensors_to_world_landmarks[Output<LandmarkList>(kLandmarksTag)];

    // Keeps only the actual world landmarks.
    auto& split_landmark_list = graph.AddNode("SplitLandmarkListCalculator");
    ConfigureSplitLandmarkListCalculator(
        &split_landmark_list
             .GetOptions<mediapipe::SplitVectorCalculatorOptions>());
    raw_world_landmarks >> split_landmark_list.In("");
    auto split_landmarks = split_landmark_list.Out(0);

    // Reuses the visibility and presence field in pose landmarks for the world
    // landmarks.
    auto& visibility_copy = graph.AddNode("VisibilityCopyCalculator");
    ConfigureVisibilityCopyCalculator(
        &visibility_copy
             .GetOptions<mediapipe::VisibilityCopyCalculatorOptions>());
    split_landmarks >> visibility_copy.In(kLandmarksToTag);
    landmarks >> visibility_copy.In(kNormLandmarksFromTag);
    auto world_landmarks =
        visibility_copy[Output<LandmarkList>(kLandmarksToTag)];

    // Each raw landmark needs to pass through LandmarkLetterboxRemoval +
    // LandmarkProjection.

    // Landmark letterbox removal for landmarks.
    auto& landmark_letterbox_removal =
        graph.AddNode("LandmarkLetterboxRemovalCalculator");
    letterbox_padding >> landmark_letterbox_removal.In(kLetterboxPaddingTag);
    landmarks >> landmark_letterbox_removal.In(kLandmarksTag);
    auto adjusted_landmarks = landmark_letterbox_removal.Out(kLandmarksTag);

    // Projects the landmarks.
    auto& landmarks_projection = graph.AddNode("LandmarkProjectionCalculator");
    adjusted_landmarks >> landmarks_projection.In(kNormLandmarksTag);
    pose_rect >> landmarks_projection.In(kNormRectTag);
    auto projected_landmarks = landmarks_projection.Out(kNormLandmarksTag)
                                   .Cast<NormalizedLandmarkList>();

    // Landmark letterbox removal for auxiliary landmarks.
    auto& auxiliary_landmark_letterbox_removal =
        graph.AddNode("LandmarkLetterboxRemovalCalculator");
    letterbox_padding >>
        auxiliary_landmark_letterbox_removal.In(kLetterboxPaddingTag);
    auxiliary_landmarks >>
        auxiliary_landmark_letterbox_removal.In(kLandmarksTag);
    auto auxiliary_adjusted_landmarks =
        auxiliary_landmark_letterbox_removal.Out(kLandmarksTag);

    // Projects the auxiliary landmarks.
    auto& auxiliary_landmarks_projection =
        graph.AddNode("LandmarkProjectionCalculator");
    auxiliary_adjusted_landmarks >>
        auxiliary_landmarks_projection.In(kNormLandmarksTag);
    pose_rect >> auxiliary_landmarks_projection.In(kNormRectTag);
    auto auxiliary_projected_landmarks =
        auxiliary_landmarks_projection.Out(kNormLandmarksTag)
            .Cast<NormalizedLandmarkList>();

    // Project world landmarks.
    auto& world_landmarks_projection =
        graph.AddNode("WorldLandmarkProjectionCalculator");
    world_landmarks >> world_landmarks_projection.In(kLandmarksTag);
    pose_rect >> world_landmarks_projection.In(kNormRectTag);
    auto world_projected_landmarks =
        world_landmarks_projection.Out(kLandmarksTag).Cast<LandmarkList>();

    std::optional<Stream<Image>> segmentation_mask;
    if (output_segmentation_mask) {
      //  Decodes the segmentation tensor into a mask image with pixel values in
      //  [0, 1] (1 for person and 0 for background).
      auto& tensors_to_segmentation =
          graph.AddNode("TensorsToSegmentationCalculator");
      ConfigureTensorsToSegmentationCalculator(
          &tensors_to_segmentation.GetOptions<
              mediapipe::TensorsToSegmentationCalculatorOptions>());
      ensured_segmentation_tensors >> tensors_to_segmentation.In(kTensorsTag);
      auto raw_segmentation_mask =
          tensors_to_segmentation[Output<Image>(kMaskTag)];

      // Calculates the inverse transformation matrix.
      auto& inverse_matrix = graph.AddNode("InverseMatrixCalculator");
      matrix >> inverse_matrix.In(kMatrixTag);
      auto inverted_matrix = inverse_matrix.Out(kMatrixTag);

      // Projects the segmentation mask from the letterboxed ROI back to the
      // full image.
      auto& warp_affine = graph.AddNode("WarpAffineCalculator");
      ConfigureWarpAffineCalculator(
          &warp_affine.GetOptions<mediapipe::WarpAffineCalculatorOptions>());
      image_size >> warp_affine.In(kOutputSizeTag);
      inverted_matrix >> warp_affine.In(kMatrixTag);
      raw_segmentation_mask >> warp_affine.In(kImageTag);
      segmentation_mask = warp_affine.Out(kImageTag).Cast<Image>();
    }

    // Calculate region of interest based on auxiliary landmarks, to be used
    // in the next frame. Consists of LandmarksToDetection +
    // AlignmentPointsRects + RectTransformation.

    auto& auxiliary_landmarks_to_detection =
        graph.AddNode("LandmarksToDetectionCalculator");
    auxiliary_projected_landmarks >>
        auxiliary_landmarks_to_detection.In(kNormLandmarksTag);
    auto detection = auxiliary_landmarks_to_detection.Out(kDetectionTag);

    auto& detection_to_rect = graph.AddNode("AlignmentPointsRectsCalculator");
    ConfigureAlignmentPointsRectsCalculator(
        &detection_to_rect
             .GetOptions<mediapipe::DetectionsToRectsCalculatorOptions>());
    detection >> detection_to_rect.In(kDetectionTag);
    image_size >> detection_to_rect.In(kImageSizeTag);
    auto raw_pose_rects = detection_to_rect.Out(kNormRectTag);

    auto& rect_transformation = graph.AddNode("RectTransformationCalculator");
    ConfigureRectTransformationCalculator(
        &rect_transformation
             .GetOptions<mediapipe::RectTransformationCalculatorOptions>());
    image_size >> rect_transformation.In(kImageSizeTag);
    raw_pose_rects >> rect_transformation.In("NORM_RECT");
    auto pose_rect_next_frame = rect_transformation[Output<NormalizedRect>("")];

    return {{
        /* pose_landmarks= */ projected_landmarks,
        /* world_pose_landmarks= */ world_projected_landmarks,
        /* auxiliary_pose_landmarks= */ auxiliary_projected_landmarks,
        /* pose_rect_next_frame= */ pose_rect_next_frame,
        /* pose_presence= */ pose_presence,
        /* pose_presence_score= */ pose_presence_score,
        /* segmentation_mask= */ segmentation_mask,
    }};
  }
};

// clang-format off
REGISTER_MEDIAPIPE_GRAPH(
  ::mediapipe::tasks::vision::pose_landmarker::SinglePoseLandmarksDetectorGraph); // NOLINT
// clang-format on

// A "mediapipe.tasks.vision.pose_landmarker.MultiplePoseLandmarksDetectorGraph"
// performs multi pose landmark detection.
// - Accepts CPU input image and a vector of pose rect RoIs to detect the
//   multiple poses landmarks enclosed by the RoIs. Output vectors of
//   pose landmarks related results, where each element in the vectors
//   corresponds to the result of the same pose.
//
// Inputs:
//   IMAGE - Image
//     Image to perform detection on.
//   NORM_RECT - std::vector<NormalizedRect>
//     A vector of multiple pose rects enclosing the pose RoI to perform
//     landmarks detection on.
//
//
// Outputs:
//   LANDMARKS: - std::vector<NormalizedLandmarkList>
//     Vector of detected pose landmarks.
//   WORLD_LANDMARKS - std::vector<LandmarkList>
//     Vector of detected pose landmarks in world coordinates.
//   AUXILIARY_LANDMARKS - std::vector<NormalizedLandmarkList>
//     Vector of detected pose auxiliary landmarks.
//   POSE_RECT_NEXT_FRAME - std::vector<NormalizedRect>
//     Vector of the predicted rects enclosing the same pose RoI for landmark
//     detection on the next frame.
//   PRESENCE - std::vector<bool>
//     Vector of boolean value indicates whether the pose is present.
//   PRESENCE_SCORE - std::vector<float>
//     Vector of float value indicates the probability that the pose is present.
//   SEGMENTATION_MASK - std::vector<Image>
//     Vector of segmentation masks.
//
// Example:
// node {
//   calculator:
//   "mediapipe.tasks.vision.pose_landmarker.MultiplePoseLandmarksDetectorGraph"
//   input_stream: "IMAGE:input_image"
//   input_stream: "POSE_RECT:pose_rect"
//   output_stream: "LANDMARKS:pose_landmarks"
//   output_stream: "WORLD_LANDMARKS:world_pose_landmarks"
//   output_stream: "AUXILIARY_LANDMARKS:auxiliary_landmarks"
//   output_stream: "POSE_RECT_NEXT_FRAME:pose_rect_next_frame"
//   output_stream: "PRESENCE:pose_presence"
//   output_stream: "PRESENCE_SCORE:pose_presence_score"
//   output_stream: "SEGMENTATION_MASK:segmentation_mask"
//   options {
//     [mediapipe.tasks.vision.pose_landmarker.proto.PoseLandmarksDetectorGraphOptions.ext]
//     {
//       base_options {
//          model_asset {
//            file_name: "pose_landmark_lite.tflite"
//          }
//       }
//       min_detection_confidence: 0.5
//     }
//   }
// }
class MultiplePoseLandmarksDetectorGraph : public core::ModelTaskGraph {
 public:
  absl::StatusOr<CalculatorGraphConfig> GetConfig(
      SubgraphContext* sc) override {
    Graph graph;
    bool output_segmentation_masks =
        HasOutput(sc->OriginalNode(), kSegmentationMaskTag);
    MP_ASSIGN_OR_RETURN(
        auto pose_landmark_detection_outputs,
        BuildPoseLandmarksDetectorGraph(
            sc->Options<PoseLandmarksDetectorGraphOptions>(),
            graph[Input<Image>(kImageTag)],
            graph[Input<std::vector<NormalizedRect>>(kNormRectTag)], graph,
            output_segmentation_masks));
    pose_landmark_detection_outputs.landmark_lists >>
        graph[Output<std::vector<NormalizedLandmarkList>>(kLandmarksTag)];
    pose_landmark_detection_outputs.world_landmark_lists >>
        graph[Output<std::vector<LandmarkList>>(kWorldLandmarksTag)];
    pose_landmark_detection_outputs.auxiliary_landmark_lists >>
        graph[Output<std::vector<NormalizedLandmarkList>>(kAuxLandmarksTag)];
    pose_landmark_detection_outputs.pose_rects_next_frame >>
        graph[Output<std::vector<NormalizedRect>>(kPoseRectsNextFrameTag)];
    pose_landmark_detection_outputs.presences >>
        graph[Output<std::vector<bool>>(kPresenceTag)];
    pose_landmark_detection_outputs.presence_scores >>
        graph[Output<std::vector<float>>(kPresenceScoreTag)];
    if (pose_landmark_detection_outputs.segmentation_masks) {
      *pose_landmark_detection_outputs.segmentation_masks >>
          graph[Output<std::vector<Image>>(kSegmentationMaskTag)];
    }

    return graph.GetConfig();
  }

 private:
  absl::StatusOr<PoseLandmarkerOutputs> BuildPoseLandmarksDetectorGraph(
      const PoseLandmarksDetectorGraphOptions& subgraph_options,
      Source<Image> image_in,
      Source<std::vector<NormalizedRect>> multi_pose_rects, Graph& graph,
      bool output_segmentation_masks) {
    auto& begin_loop_multi_pose_rects =
        graph.AddNode("BeginLoopNormalizedRectCalculator");
    image_in >> begin_loop_multi_pose_rects.In("CLONE");
    multi_pose_rects >> begin_loop_multi_pose_rects.In("ITERABLE");
    auto batch_end = begin_loop_multi_pose_rects.Out("BATCH_END");
    auto image = begin_loop_multi_pose_rects.Out("CLONE");
    auto pose_rect = begin_loop_multi_pose_rects.Out("ITEM");

    auto& pose_landmark_subgraph = graph.AddNode(
        "mediapipe.tasks.vision.pose_landmarker."
        "SinglePoseLandmarksDetectorGraph");
    pose_landmark_subgraph.GetOptions<PoseLandmarksDetectorGraphOptions>() =
        subgraph_options;
    image >> pose_landmark_subgraph.In(kImageTag);
    pose_rect >> pose_landmark_subgraph.In(kNormRectTag);
    auto landmarks = pose_landmark_subgraph.Out(kLandmarksTag);
    auto world_landmarks = pose_landmark_subgraph.Out(kWorldLandmarksTag);
    auto auxiliary_landmarks = pose_landmark_subgraph.Out(kAuxLandmarksTag);
    auto pose_rect_next_frame =
        pose_landmark_subgraph.Out(kPoseRectNextFrameTag);
    auto presence = pose_landmark_subgraph.Out(kPresenceTag);
    auto presence_score = pose_landmark_subgraph.Out(kPresenceScoreTag);

    auto& end_loop_landmarks =
        graph.AddNode("EndLoopNormalizedLandmarkListVectorCalculator");
    batch_end >> end_loop_landmarks.In(kBatchEndTag);
    landmarks >> end_loop_landmarks.In(kItemTag);
    auto landmark_lists =
        end_loop_landmarks[Output<std::vector<NormalizedLandmarkList>>(
            kIterableTag)];

    auto& end_loop_world_landmarks =
        graph.AddNode("EndLoopLandmarkListVectorCalculator");
    batch_end >> end_loop_world_landmarks.In(kBatchEndTag);
    world_landmarks >> end_loop_world_landmarks.In(kItemTag);
    auto world_landmark_lists =
        end_loop_world_landmarks[Output<std::vector<LandmarkList>>(
            kIterableTag)];

    auto& end_loop_auxiliary_landmarks =
        graph.AddNode("EndLoopNormalizedLandmarkListVectorCalculator");
    batch_end >> end_loop_auxiliary_landmarks.In(kBatchEndTag);
    auxiliary_landmarks >> end_loop_auxiliary_landmarks.In(kItemTag);
    auto auxiliary_landmark_lists = end_loop_auxiliary_landmarks
        [Output<std::vector<NormalizedLandmarkList>>(kIterableTag)];

    auto& end_loop_rects_next_frame =
        graph.AddNode("EndLoopNormalizedRectCalculator");
    batch_end >> end_loop_rects_next_frame.In(kBatchEndTag);
    pose_rect_next_frame >> end_loop_rects_next_frame.In(kItemTag);
    auto pose_rects_next_frame =
        end_loop_rects_next_frame[Output<std::vector<NormalizedRect>>(
            kIterableTag)];

    auto& end_loop_presence = graph.AddNode("EndLoopBooleanCalculator");
    batch_end >> end_loop_presence.In(kBatchEndTag);
    presence >> end_loop_presence.In(kItemTag);
    auto presences = end_loop_presence[Output<std::vector<bool>>(kIterableTag)];

    auto& end_loop_presence_score = graph.AddNode("EndLoopFloatCalculator");
    batch_end >> end_loop_presence_score.In(kBatchEndTag);
    presence_score >> end_loop_presence_score.In(kItemTag);
    auto presence_scores =
        end_loop_presence_score[Output<std::vector<float>>(kIterableTag)];

    std::optional<Stream<std::vector<Image>>> segmentation_masks_vector;
    if (output_segmentation_masks) {
      auto segmentation_mask = pose_landmark_subgraph.Out(kSegmentationMaskTag);
      auto& end_loop_segmentation_mask =
          graph.AddNode("EndLoopImageCalculator");
      batch_end >> end_loop_segmentation_mask.In(kBatchEndTag);
      segmentation_mask >> end_loop_segmentation_mask.In(kItemTag);
      segmentation_masks_vector =
          end_loop_segmentation_mask[Output<std::vector<Image>>(kIterableTag)];
    }

    // Apply smoothing filter only on the single pose landmarks, because
    // landmarks smoothing calculator doesn't support multiple landmarks yet.
    // Notice the landmarks smoothing calculator cannot be put inside the for
    // loop calculator, because the smoothing calculator utilize the timestamp
    // to smoote landmarks across frames but the for loop calculator makes fake
    // timestamps for the streams.
    if (subgraph_options.smooth_landmarks()) {
      Stream<std::pair<int, int>> image_size = GetImageSize(image_in, graph);
      Stream<int> zero_index =
          CreateIntConstantStream(landmark_lists, 0, graph);
      Stream<NormalizedLandmarkList> landmarks =
          GetItem(landmark_lists, zero_index, graph);
      Stream<LandmarkList> world_landmarks =
          GetItem(world_landmark_lists, zero_index, graph);
      Stream<NormalizedRect> roi =
          GetItem(pose_rects_next_frame, zero_index, graph);

      // Apply smoothing filter on pose landmarks.
      landmarks = SmoothLandmarksVisibility(
          landmarks, /*low_pass_filter_alpha=*/0.1f, graph);
      landmarks = SmoothLandmarks(
          landmarks, image_size, roi,
          {// Min cutoff 0.05 results into ~0.01 alpha in landmark EMA filter
           // when landmark is static.
           /*min_cutoff=*/0.05f,
           // Beta 80.0 in combination with min_cutoff 0.05 results into ~0.94
           // alpha in landmark EMA filter when landmark is moving fast.
           /*beta=*/80.0f,
           // Derivative cutoff 1.0 results into ~0.17 alpha in landmark
           // velocity EMA filter.
           /*derivate_cutoff=*/1.0f},
          graph);

      // Apply smoothing filter on pose world landmarks.
      world_landmarks = SmoothLandmarksVisibility(
          world_landmarks, /*low_pass_filter_alpha=*/0.1f, graph);
      world_landmarks = SmoothLandmarks(
          world_landmarks,
          /*scale_roi=*/std::nullopt,
          {// Min cutoff 0.1 results into ~ 0.02 alpha in landmark EMA filter
           // when landmark is static.
           /*min_cutoff=*/0.1f,
           // Beta 40.0 in combination with min_cutoff 0.1 results into ~0.8
           // alpha in landmark EMA filter when landmark is moving fast.
           /*beta=*/40.0f,
           // Derivative cutoff 1.0 results into ~0.17 alpha in landmark
           // velocity EMA filter.
           /*derivate_cutoff=*/1.0f},
          graph);

      // Wrap the single pose landmarks into a vector of landmarks.
      auto& concat_landmarks =
          graph.AddNode("ConcatenateNormalizedLandmarkListVectorCalculator");
      landmarks >> concat_landmarks.In("");
      landmark_lists =
          concat_landmarks.Out("").Cast<std::vector<NormalizedLandmarkList>>();

      auto& concat_world_landmarks =
          graph.AddNode("ConcatenateLandmarkListVectorCalculator");
      world_landmarks >> concat_world_landmarks.In("");
      world_landmark_lists =
          concat_world_landmarks.Out("").Cast<std::vector<LandmarkList>>();
    }

    return {{
        /* landmark_lists= */ landmark_lists,
        /* world_landmark_lists= */ world_landmark_lists,
        /* auxiliary_landmark_lists= */ auxiliary_landmark_lists,
        /* pose_rects_next_frame= */ pose_rects_next_frame,
        /* presences= */ presences,
        /* presence_scores= */ presence_scores,
        /* segmentation_masks= */ segmentation_masks_vector,
    }};
  }
};

// clang-format off
REGISTER_MEDIAPIPE_GRAPH(
  ::mediapipe::tasks::vision::pose_landmarker::MultiplePoseLandmarksDetectorGraph); // NOLINT
// clang-format on

}  // namespace pose_landmarker
}  // namespace vision
}  // namespace tasks
}  // namespace mediapipe
