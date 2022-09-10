/* Copyright 2022 The MediaPipe Authors. All Rights Reserved.

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
#include <type_traits>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "mediapipe/calculators/core/split_vector_calculator.pb.h"
#include "mediapipe/calculators/tensor/tensors_to_classification_calculator.pb.h"
#include "mediapipe/calculators/tensor/tensors_to_landmarks_calculator.pb.h"
#include "mediapipe/calculators/util/rect_transformation_calculator.pb.h"
#include "mediapipe/calculators/util/thresholding_calculator.pb.h"
#include "mediapipe/framework/api2/builder.h"
#include "mediapipe/framework/api2/port.h"
#include "mediapipe/framework/formats/classification.pb.h"
#include "mediapipe/framework/formats/image.h"
#include "mediapipe/framework/formats/landmark.pb.h"
#include "mediapipe/framework/formats/rect.pb.h"
#include "mediapipe/framework/formats/tensor.h"
#include "mediapipe/tasks/cc/common.h"
#include "mediapipe/tasks/cc/components/image_preprocessing.h"
#include "mediapipe/tasks/cc/core/model_resources.h"
#include "mediapipe/tasks/cc/core/model_task_graph.h"
#include "mediapipe/tasks/cc/core/proto/inference_subgraph.pb.h"
#include "mediapipe/tasks/cc/core/utils.h"
#include "mediapipe/tasks/cc/metadata/metadata_extractor.h"
#include "mediapipe/tasks/cc/vision/hand_landmark/hand_landmark_detector_options.pb.h"
#include "mediapipe/tasks/cc/vision/utils/image_tensor_specs.h"
#include "mediapipe/tasks/metadata/metadata_schema_generated.h"
#include "mediapipe/util/label_map.pb.h"
#include "mediapipe/util/label_map_util.h"

namespace mediapipe {
namespace tasks {
namespace vision {

namespace {

using ::mediapipe::api2::Input;
using ::mediapipe::api2::Output;
using ::mediapipe::api2::builder::Graph;
using ::mediapipe::api2::builder::Source;
using ::mediapipe::tasks::core::ModelResources;
using LabelItems = mediapipe::proto_ns::Map<int64, ::mediapipe::LabelMapItem>;

constexpr char kImageTag[] = "IMAGE";
constexpr char kHandRectTag[] = "HAND_RECT";

constexpr char kLandmarksTag[] = "LANDMARKS";
constexpr char kWorldLandmarksTag[] = "WORLD_LANDMARKS";
constexpr char kHandRectNextFrameTag[] = "HAND_RECT_NEXT_FRAME";
constexpr char kPresenceTag[] = "PRESENCE";
constexpr char kPresenceScoreTag[] = "PRESENCE_SCORE";
constexpr char kHandednessTag[] = "HANDEDNESS";
constexpr char kImageSizeTag[] = "IMAGE_SIZE";

constexpr int kLandmarksNum = 21;
constexpr float kLandmarksNormalizeZ = 0.4;
constexpr int kModelOutputTensorSplitNum = 4;

struct HandLandmarkDetectionOuts {
  Source<NormalizedLandmarkList> hand_landmarks;
  Source<LandmarkList> world_hand_landmarks;
  Source<NormalizedRect> hand_rect_next_frame;
  Source<bool> hand_presence;
  Source<float> hand_presence_score;
  Source<ClassificationList> handedness;
  Source<std::pair<int, int>> image_size;
};

absl::Status SanityCheckOptions(const HandLandmarkDetectorOptions& options) {
  if (options.min_detection_confidence() < 0 ||
      options.min_detection_confidence() > 1) {
    return CreateStatusWithPayload(absl::StatusCode::kInvalidArgument,
                                   "Invalid `min_detection_confidence` option: "
                                   "value must be in the range [0.0, 1.0]",
                                   MediaPipeTasksStatus::kInvalidArgumentError);
  }
  return absl::OkStatus();
}

// Builds an ImageTensorSpecs for configuring the image preprocessing subgraph.
absl::StatusOr<ImageTensorSpecs> BuildImageTensorSpecs(
    const ModelResources& model_resources) {
  const tflite::Model& model = *model_resources.GetTfLiteModel();
  if (model.subgraphs()->size() != 1) {
    return CreateStatusWithPayload(
        absl::StatusCode::kInvalidArgument,
        "Hand landmark model is assumed to have a single subgraph.",
        MediaPipeTasksStatus::kInvalidArgumentError);
  }
  const auto* primary_subgraph = (*model.subgraphs())[0];
  if (primary_subgraph->inputs()->size() != 1) {
    return CreateStatusWithPayload(
        absl::StatusCode::kInvalidArgument,
        "Hand landmark model is assumed to have a single input.",
        MediaPipeTasksStatus::kInvalidArgumentError);
  }
  const auto* input_tensor =
      (*primary_subgraph->tensors())[(*primary_subgraph->inputs())[0]];
  ASSIGN_OR_RETURN(const auto* image_tensor_metadata,
                   vision::GetImageTensorMetadataIfAny(
                       *model_resources.GetMetadataExtractor(), 0));
  return vision::BuildInputImageTensorSpecs(*input_tensor,
                                            image_tensor_metadata);
}

// Split hand landmark detection model output tensor into four parts,
// representing landmarks, presence scores, handedness, and world landmarks,
// respectively.
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
    mediapipe::TensorsToLandmarksCalculatorOptions* options) {
  options->set_num_landmarks(kLandmarksNum);
  if (normalize) {
    options->set_input_image_height(input_image_tensor_spec.image_height);
    options->set_input_image_width(input_image_tensor_spec.image_width);
    options->set_normalize_z(kLandmarksNormalizeZ);
  }
}

void ConfigureTensorsToHandednessCalculator(
    mediapipe::TensorsToClassificationCalculatorOptions* options) {
  options->set_top_k(1);
  options->set_binary_classification(true);
  // TODO: use model Metadata to set label_items.
  LabelMapItem left_hand = LabelMapItem();
  left_hand.set_name("Left");
  left_hand.set_display_name("Left");
  LabelMapItem right_hand = LabelMapItem();
  right_hand.set_name("Right");
  right_hand.set_display_name("Right");
  (*options->mutable_label_items())[0] = std::move(left_hand);
  (*options->mutable_label_items())[1] = std::move(right_hand);
}

void ConfigureHandRectTransformationCalculator(
    mediapipe::RectTransformationCalculatorOptions* options) {
  // TODO: make rect transformation configurable, e.g. from
  // Metadata or configuration options.
  options->set_scale_x(2.0f);
  options->set_scale_y(2.0f);
  options->set_shift_y(-0.1f);
  options->set_square_long(true);
}

}  // namespace

// A "mediapipe.tasks.vision.HandLandmarkDetectorGraph" performs hand landmark
// detection.
// - Accepts CPU input images and outputs Landmark on CPU.
//
// Inputs:
//   IMAGE - Image
//     Image to perform detection on.
//   HAND_RECT - NormalizedRect @Optional
//     Rect enclosing the RoI to perform detection on. If not set, the detection
//     RoI is the whole image.
//
//
// Outputs:
//   LANDMARKS: - NormalizedLandmarkList
//     Detected hand landmarks.
//   WORLD_LANDMARKS - LandmarkList
//     Detected hand landmarks in world coordinates.
//   HAND_RECT_NEXT_FRAME - NormalizedRect
//     The predicted Rect enclosing the hand RoI for landmark detection on the
//     next frame.
//   PRESENCE - bool
//     Boolean value indicates whether the hand is present.
//   PRESENCE_SCORE - float
//     Float value indicates the probability that the hand is present.
//   HANDEDNESS - ClassificationList
//     Classification of handedness.
//
// Example:
// node {
//   calculator: "mediapipe.tasks.vision.HandLandmarkDetectorGraph"
//   input_stream: "IMAGE:input_video"
//   input_stream: "HAND_RECT:hand_rect"
//   output_stream: "LANDMARKS:hand_landmarks"
//   output_stream: "WORLD_LANDMARKS:world_hand_landmarks"
//   output_stream: "HAND_RECT_NEXT_FRAME:hand_rect_next_frame"
//   output_stream: "PRESENCE:hand_presence"
//   output_stream: "PRESENCE_SCORE:hand_presence_score"
//   output_stream: "HANDEDNESS:handedness"
//   options {
//     [mediapipe.tasks.HandLandmarkDetectorGraph.ext] {
//       base_options {
//          model_file {
//            file_name: "hand_landmark_lite.tflite"
//          }
//       }
//       min_detection_confidence: 0.5
//     }
//   }
// }
class HandLandmarkDetectorGraph : public core::ModelTaskGraph {
 public:
  absl::StatusOr<CalculatorGraphConfig> GetConfig(
      SubgraphContext* sc) override {
    ASSIGN_OR_RETURN(const auto* model_resources,
                     CreateModelResources<HandLandmarkDetectorOptions>(sc));
    Graph graph;
    ASSIGN_OR_RETURN(auto hand_landmark_detection_outs,
                     BuildHandLandmarkDetectionSubgraph(
                         sc->Options<HandLandmarkDetectorOptions>(),
                         *model_resources, graph[Input<Image>(kImageTag)],
                         graph[Input<NormalizedRect>(kHandRectTag)], graph));
    hand_landmark_detection_outs.hand_landmarks >>
        graph[Output<NormalizedLandmarkList>(kLandmarksTag)];
    hand_landmark_detection_outs.world_hand_landmarks >>
        graph[Output<LandmarkList>(kWorldLandmarksTag)];
    hand_landmark_detection_outs.hand_rect_next_frame >>
        graph[Output<NormalizedRect>(kHandRectNextFrameTag)];
    hand_landmark_detection_outs.hand_presence >>
        graph[Output<bool>(kPresenceTag)];
    hand_landmark_detection_outs.hand_presence_score >>
        graph[Output<float>(kPresenceScoreTag)];
    hand_landmark_detection_outs.handedness >>
        graph[Output<ClassificationList>(kHandednessTag)];
    hand_landmark_detection_outs.image_size >>
        graph[Output<std::pair<int, int>>(kImageSizeTag)];

    return graph.GetConfig();
  }

 private:
  // Adds a mediapipe hand landmark detection graph into the provided
  // builder::Graph instance.
  //
  // subgraph_options: the mediapipe tasks module HandLandmarkDetectorOptions.
  // model_resources: the ModelSources object initialized from an hand landmark
  // detection model file with model metadata.
  // image_in: (mediapipe::Image) stream to run hand landmark detection on.
  // rect: (NormalizedRect) stream to run on the RoI of image.
  // graph: the mediapipe builder::Graph instance to be updated.
  absl::StatusOr<HandLandmarkDetectionOuts> BuildHandLandmarkDetectionSubgraph(
      const HandLandmarkDetectorOptions& subgraph_options,
      const core::ModelResources& model_resources, Source<Image> image_in,
      Source<NormalizedRect> hand_rect, Graph& graph) {
    MP_RETURN_IF_ERROR(SanityCheckOptions(subgraph_options));

    auto& preprocessing =
        graph.AddNode("mediapipe.tasks.ImagePreprocessingSubgraph");
    MP_RETURN_IF_ERROR(ConfigureImagePreprocessing(
        model_resources,
        &preprocessing.GetOptions<ImagePreprocessingOptions>()));
    image_in >> preprocessing.In("IMAGE");
    hand_rect >> preprocessing.In("NORM_RECT");
    auto image_size = preprocessing[Output<std::pair<int, int>>("IMAGE_SIZE")];

    ASSIGN_OR_RETURN(auto image_tensor_specs,
                     BuildImageTensorSpecs(model_resources));

    auto& inference = AddInference(model_resources, graph);
    preprocessing.Out("TENSORS") >> inference.In("TENSORS");

    // Split model output tensors to multiple streams.
    auto& split_tensors_vector = graph.AddNode("SplitTensorVectorCalculator");
    ConfigureSplitTensorVectorCalculator(
        &split_tensors_vector
             .GetOptions<mediapipe::SplitVectorCalculatorOptions>());
    inference.Out("TENSORS") >> split_tensors_vector.In("");
    auto landmark_tensors = split_tensors_vector.Out(0);
    auto hand_flag_tensors = split_tensors_vector.Out(1);
    auto handedness_tensors = split_tensors_vector.Out(2);
    auto world_landmark_tensors = split_tensors_vector.Out(3);

    // Decodes the landmark tensors into a list of landmarks, where the landmark
    // coordinates are normalized by the size of the input image to the model.
    auto& tensors_to_landmarks = graph.AddNode("TensorsToLandmarksCalculator");
    ConfigureTensorsToLandmarksCalculator(
        image_tensor_specs, /* normalize = */ true,
        &tensors_to_landmarks
             .GetOptions<mediapipe::TensorsToLandmarksCalculatorOptions>());
    landmark_tensors >> tensors_to_landmarks.In("TENSORS");

    // Decodes the landmark tensors into a list of landmarks, where the landmark
    // coordinates are world coordinates in meters.
    auto& tensors_to_world_landmarks =
        graph.AddNode("TensorsToLandmarksCalculator");
    ConfigureTensorsToLandmarksCalculator(
        image_tensor_specs, /* normalize = */ false,
        &tensors_to_world_landmarks
             .GetOptions<mediapipe::TensorsToLandmarksCalculatorOptions>());
    world_landmark_tensors >> tensors_to_world_landmarks.In("TENSORS");

    // Converts the hand-flag tensor into a float that represents the confidence
    // score of hand presence.
    auto& tensors_to_hand_presence = graph.AddNode("TensorsToFloatsCalculator");
    hand_flag_tensors >> tensors_to_hand_presence.In("TENSORS");

    // Converts the handedness tensor into a float that represents the
    // classification score of handedness.
    auto& tensors_to_handedness =
        graph.AddNode("TensorsToClassificationCalculator");
    ConfigureTensorsToHandednessCalculator(
        &tensors_to_handedness.GetOptions<
            mediapipe::TensorsToClassificationCalculatorOptions>());
    handedness_tensors >> tensors_to_handedness.In("TENSORS");
    auto hand_presence_score = tensors_to_hand_presence[Output<float>("FLOAT")];
    auto handedness =
        tensors_to_handedness[Output<ClassificationList>("CLASSIFICATIONS")];

    // Applies a threshold to the confidence score to determine whether a
    // hand is present.
    auto& hand_presence_thresholding = graph.AddNode("ThresholdingCalculator");
    hand_presence_thresholding
        .GetOptions<mediapipe::ThresholdingCalculatorOptions>()
        .set_threshold(subgraph_options.min_detection_confidence());
    hand_presence_score >> hand_presence_thresholding.In("FLOAT");
    auto hand_presence = hand_presence_thresholding[Output<bool>("FLAG")];

    // Adjusts landmarks (already normalized to [0.f, 1.f]) on the letterboxed
    // hand image (after image transformation with the FIT scale mode) to the
    // corresponding locations on the same image with the letterbox removed
    // (hand image before image transformation).
    auto& landmark_letterbox_removal =
        graph.AddNode("LandmarkLetterboxRemovalCalculator");
    preprocessing.Out("LETTERBOX_PADDING") >>
        landmark_letterbox_removal.In("LETTERBOX_PADDING");
    tensors_to_landmarks.Out("NORM_LANDMARKS") >>
        landmark_letterbox_removal.In("LANDMARKS");

    // Projects the landmarks from the cropped hand image to the corresponding
    // locations on the full image before cropping (input to the graph).
    auto& landmark_projection = graph.AddNode("LandmarkProjectionCalculator");
    landmark_letterbox_removal.Out("LANDMARKS") >>
        landmark_projection.In("NORM_LANDMARKS");
    hand_rect >> landmark_projection.In("NORM_RECT");
    auto projected_landmarks =
        landmark_projection[Output<NormalizedLandmarkList>("NORM_LANDMARKS")];

    // Projects the world landmarks from the cropped pose image to the
    // corresponding locations on the full image before cropping (input to the
    // graph).
    auto& world_landmark_projection =
        graph.AddNode("WorldLandmarkProjectionCalculator");
    tensors_to_world_landmarks.Out("LANDMARKS") >>
        world_landmark_projection.In("LANDMARKS");
    hand_rect >> world_landmark_projection.In("NORM_RECT");
    auto projected_world_landmarks =
        world_landmark_projection[Output<LandmarkList>("LANDMARKS")];

    // Converts the hand landmarks into a rectangle (normalized by image size)
    // that encloses the hand.
    auto& hand_landmarks_to_rect =
        graph.AddNode("HandLandmarksToRectCalculator");
    image_size >> hand_landmarks_to_rect.In("IMAGE_SIZE");
    projected_landmarks >> hand_landmarks_to_rect.In("NORM_LANDMARKS");

    // Expands the hand rectangle so that in the next video frame it's likely to
    // still contain the hand even with some motion.
    auto& hand_rect_transformation =
        graph.AddNode("RectTransformationCalculator");
    ConfigureHandRectTransformationCalculator(
        &hand_rect_transformation
             .GetOptions<mediapipe::RectTransformationCalculatorOptions>());
    image_size >> hand_rect_transformation.In("IMAGE_SIZE");
    hand_landmarks_to_rect.Out("NORM_RECT") >>
        hand_rect_transformation.In("NORM_RECT");
    auto hand_rect_next_frame =
        hand_rect_transformation[Output<NormalizedRect>("")];

    return {{
        /* hand_landmarks= */ projected_landmarks,
        /* world_hand_landmarks= */ projected_world_landmarks,
        /* hand_rect_next_frame= */ hand_rect_next_frame,
        /* hand_presence= */ hand_presence,
        /* hand_presence_score= */ hand_presence_score,
        /* handedness= */ handedness,
        /* image_size= */ image_size,
    }};
  }
};

REGISTER_MEDIAPIPE_GRAPH(::mediapipe::tasks::vision::HandLandmarkDetectorGraph);

}  // namespace vision
}  // namespace tasks
}  // namespace mediapipe
