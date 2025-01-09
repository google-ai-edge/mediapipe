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
#include <type_traits>
#include <vector>

#include "absl/log/absl_log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "mediapipe/calculators/tensor/tensors_to_classification_calculator.pb.h"
#include "mediapipe/framework/api2/builder.h"
#include "mediapipe/framework/api2/port.h"
#include "mediapipe/framework/formats/classification.pb.h"
#include "mediapipe/framework/formats/landmark.pb.h"
#include "mediapipe/framework/formats/matrix.h"
#include "mediapipe/framework/formats/rect.pb.h"
#include "mediapipe/framework/formats/tensor.h"
#include "mediapipe/tasks/cc/common.h"
#include "mediapipe/tasks/cc/components/processors/classification_postprocessing_graph.h"
#include "mediapipe/tasks/cc/core/model_asset_bundle_resources.h"
#include "mediapipe/tasks/cc/core/model_resources.h"
#include "mediapipe/tasks/cc/core/model_resources_cache.h"
#include "mediapipe/tasks/cc/core/model_task_graph.h"
#include "mediapipe/tasks/cc/core/proto/base_options.pb.h"
#include "mediapipe/tasks/cc/core/proto/external_file.pb.h"
#include "mediapipe/tasks/cc/core/proto/inference_subgraph.pb.h"
#include "mediapipe/tasks/cc/core/utils.h"
#include "mediapipe/tasks/cc/metadata/utils/zip_utils.h"
#include "mediapipe/tasks/cc/vision/gesture_recognizer/calculators/combined_prediction_calculator.pb.h"
#include "mediapipe/tasks/cc/vision/gesture_recognizer/calculators/landmarks_to_matrix_calculator.pb.h"
#include "mediapipe/tasks/cc/vision/gesture_recognizer/proto/gesture_classifier_graph_options.pb.h"
#include "mediapipe/tasks/cc/vision/gesture_recognizer/proto/gesture_embedder_graph_options.pb.h"
#include "mediapipe/tasks/cc/vision/gesture_recognizer/proto/hand_gesture_recognizer_graph_options.pb.h"
#include "mediapipe/tasks/metadata/metadata_schema_generated.h"

namespace mediapipe {
namespace tasks {
namespace vision {
namespace gesture_recognizer {

namespace {

using ::mediapipe::NormalizedRect;
using ::mediapipe::api2::Input;
using ::mediapipe::api2::Output;
using ::mediapipe::api2::builder::Graph;
using ::mediapipe::api2::builder::Source;
using ::mediapipe::tasks::components::processors::
    ConfigureTensorsToClassificationCalculator;
using ::mediapipe::tasks::core::ModelAssetBundleResources;
using ::mediapipe::tasks::core::proto::BaseOptions;
using ::mediapipe::tasks::metadata::SetExternalFile;
using ::mediapipe::tasks::vision::gesture_recognizer::proto::
    HandGestureRecognizerGraphOptions;

constexpr char kHandednessTag[] = "HANDEDNESS";
constexpr char kLandmarksTag[] = "LANDMARKS";
constexpr char kWorldLandmarksTag[] = "WORLD_LANDMARKS";
constexpr char kImageSizeTag[] = "IMAGE_SIZE";
constexpr char kNormRectTag[] = "NORM_RECT";
constexpr char kHandTrackingIdsTag[] = "HAND_TRACKING_IDS";
constexpr char kHandGesturesTag[] = "HAND_GESTURES";
constexpr char kLandmarksMatrixTag[] = "LANDMARKS_MATRIX";
constexpr char kTensorsTag[] = "TENSORS";
constexpr char kHandednessMatrixTag[] = "HANDEDNESS_MATRIX";
constexpr char kCloneTag[] = "CLONE";
constexpr char kItemTag[] = "ITEM";
constexpr char kVectorTag[] = "VECTOR";
constexpr char kIndexTag[] = "INDEX";
constexpr char kIterableTag[] = "ITERABLE";
constexpr char kBatchEndTag[] = "BATCH_END";
constexpr char kPredictionTag[] = "PREDICTION";
constexpr char kBackgroundLabel[] = "None";
constexpr char kGestureEmbedderTFLiteName[] = "gesture_embedder.tflite";
constexpr char kCannedGestureClassifierTFLiteName[] =
    "canned_gesture_classifier.tflite";
constexpr char kCustomGestureClassifierTFLiteName[] =
    "custom_gesture_classifier.tflite";

struct SubTaskModelResources {
  const core::ModelResources* gesture_embedder_model_resource = nullptr;
  const core::ModelResources* canned_gesture_classifier_model_resource =
      nullptr;
  const core::ModelResources* custom_gesture_classifier_model_resource =
      nullptr;
};

Source<std::vector<Tensor>> ConvertMatrixToTensor(Source<Matrix> matrix,
                                                  Graph& graph) {
  auto& node = graph.AddNode("TensorConverterCalculator");
  matrix >> node.In("MATRIX");
  return node[Output<std::vector<Tensor>>{"TENSORS"}];
}

absl::Status ConfigureCombinedPredictionCalculator(
    CombinedPredictionCalculatorOptions* options) {
  options->set_background_label(kBackgroundLabel);
  return absl::OkStatus();
}

void PopulateAccelerationAndUseStreamMode(
    const BaseOptions& parent_base_options,
    BaseOptions* sub_task_base_options) {
  sub_task_base_options->mutable_acceleration()->CopyFrom(
      parent_base_options.acceleration());
  sub_task_base_options->set_use_stream_mode(
      parent_base_options.use_stream_mode());
}

}  // namespace

// A
// "mediapipe.tasks.vision.gesture_recognizer.SingleHandGestureRecognizerGraph"
// performs single hand gesture recognition. This graph is used as a building
// block for mediapipe.tasks.vision.GestureRecognizerGraph.
//
// Inputs:
//   HANDEDNESS - ClassificationList
//     Classification of handedness.
//   LANDMARKS - NormalizedLandmarkList
//     Detected hand landmarks in normalized image coordinates.
//   WORLD_LANDMARKS - LandmarkList
//     Detected hand landmarks in world coordinates.
//   IMAGE_SIZE - std::pair<int, int>
//     The size of image from which the landmarks detected from.
//   NORM_RECT - NormalizedRect
//     NormalizedRect whose 'rotation' field is used to rotate the
//     landmarks before processing them.
//
// Outputs:
//   HAND_GESTURES - ClassificationList
//     Recognized hand gestures with sorted order such that the winning label is
//     the first item in the list.
//
//
// Example:
// node {
//   calculator: "mediapipe.tasks.vision.SingleHandGestureRecognizerGraph"
//   input_stream: "HANDEDNESS:handedness"
//   input_stream: "LANDMARKS:landmarks"
//   input_stream: "WORLD_LANDMARKS:world_landmarks"
//   input_stream: "IMAGE_SIZE:image_size"
//   input_stream: "NORM_RECT:norm_rect"
//   output_stream: "HAND_GESTURES:hand_gestures"
//   options {
//     [mediapipe.tasks.vision.gesture_recognizer.proto.HandGestureRecognizerGraphOptions.ext]
//     {
//       base_options {
//         model_asset {
//           file_name: "hand_gesture.tflite"
//         }
//       }
//     }
//   }
// }
class SingleHandGestureRecognizerGraph : public core::ModelTaskGraph {
 public:
  absl::StatusOr<CalculatorGraphConfig> GetConfig(
      SubgraphContext* sc) override {
    if (sc->Options<HandGestureRecognizerGraphOptions>()
            .base_options()
            .has_model_asset()) {
      MP_ASSIGN_OR_RETURN(
          const auto* model_asset_bundle_resources,
          CreateModelAssetBundleResources<HandGestureRecognizerGraphOptions>(
              sc));
      // When the model resources cache service is available, filling in
      // the file pointer meta in the subtasks' base options. Otherwise,
      // providing the file contents instead.
      MP_RETURN_IF_ERROR(SetSubTaskBaseOptions(
          *model_asset_bundle_resources,
          sc->MutableOptions<HandGestureRecognizerGraphOptions>(),
          !sc->Service(::mediapipe::tasks::core::kModelResourcesCacheService)
               .IsAvailable()));
    }
    MP_ASSIGN_OR_RETURN(const auto sub_task_model_resources,
                        CreateSubTaskModelResources(sc));
    Graph graph;
    MP_ASSIGN_OR_RETURN(auto hand_gestures,
                        BuildGestureRecognizerGraph(
                            sc->Options<HandGestureRecognizerGraphOptions>(),
                            sub_task_model_resources,
                            graph[Input<ClassificationList>(kHandednessTag)],
                            graph[Input<NormalizedLandmarkList>(kLandmarksTag)],
                            graph[Input<LandmarkList>(kWorldLandmarksTag)],
                            graph[Input<std::pair<int, int>>(kImageSizeTag)],
                            graph[Input<NormalizedRect>(kNormRectTag)], graph));
    hand_gestures >> graph[Output<ClassificationList>(kHandGesturesTag)];
    return graph.GetConfig();
  }

 private:
  // Sets the base options in the sub tasks.
  absl::Status SetSubTaskBaseOptions(const ModelAssetBundleResources& resources,
                                     HandGestureRecognizerGraphOptions* options,
                                     bool is_copy) {
    MP_ASSIGN_OR_RETURN(const auto gesture_embedder_file,
                        resources.GetFile(kGestureEmbedderTFLiteName));
    auto* gesture_embedder_graph_options =
        options->mutable_gesture_embedder_graph_options();
    SetExternalFile(gesture_embedder_file,
                    gesture_embedder_graph_options->mutable_base_options()
                        ->mutable_model_asset(),
                    is_copy);
    PopulateAccelerationAndUseStreamMode(
        options->base_options(),
        gesture_embedder_graph_options->mutable_base_options());

    MP_ASSIGN_OR_RETURN(const auto canned_gesture_classifier_file,
                        resources.GetFile(kCannedGestureClassifierTFLiteName));
    auto* canned_gesture_classifier_graph_options =
        options->mutable_canned_gesture_classifier_graph_options();
    SetExternalFile(
        canned_gesture_classifier_file,
        canned_gesture_classifier_graph_options->mutable_base_options()
            ->mutable_model_asset(),
        is_copy);
    PopulateAccelerationAndUseStreamMode(
        options->base_options(),
        canned_gesture_classifier_graph_options->mutable_base_options());

    const auto custom_gesture_classifier_file =
        resources.GetFile(kCustomGestureClassifierTFLiteName);
    if (custom_gesture_classifier_file.ok()) {
      has_custom_gesture_classifier = true;
      auto* custom_gesture_classifier_graph_options =
          options->mutable_custom_gesture_classifier_graph_options();
      SetExternalFile(
          custom_gesture_classifier_file.value(),
          custom_gesture_classifier_graph_options->mutable_base_options()
              ->mutable_model_asset(),
          is_copy);
      PopulateAccelerationAndUseStreamMode(
          options->base_options(),
          custom_gesture_classifier_graph_options->mutable_base_options());
    } else {
      ABSL_LOG(INFO) << "Custom gesture classifier is not defined.";
    }
    return absl::OkStatus();
  }

  absl::StatusOr<SubTaskModelResources> CreateSubTaskModelResources(
      SubgraphContext* sc) {
    auto* options = sc->MutableOptions<HandGestureRecognizerGraphOptions>();
    SubTaskModelResources sub_task_model_resources;
    auto& gesture_embedder_model_asset =
        *options->mutable_gesture_embedder_graph_options()
             ->mutable_base_options()
             ->mutable_model_asset();
    MP_ASSIGN_OR_RETURN(
        sub_task_model_resources.gesture_embedder_model_resource,
        CreateModelResources(sc,
                             std::make_unique<core::proto::ExternalFile>(
                                 std::move(gesture_embedder_model_asset)),
                             "_gesture_embedder"));
    auto& canned_gesture_classifier_model_asset =
        *options->mutable_canned_gesture_classifier_graph_options()
             ->mutable_base_options()
             ->mutable_model_asset();
    MP_ASSIGN_OR_RETURN(
        sub_task_model_resources.canned_gesture_classifier_model_resource,
        CreateModelResources(
            sc,
            std::make_unique<core::proto::ExternalFile>(
                std::move(canned_gesture_classifier_model_asset)),
            "_canned_gesture_classifier"));
    if (has_custom_gesture_classifier) {
      auto& custom_gesture_classifier_model_asset =
          *options->mutable_custom_gesture_classifier_graph_options()
               ->mutable_base_options()
               ->mutable_model_asset();
      MP_ASSIGN_OR_RETURN(
          sub_task_model_resources.custom_gesture_classifier_model_resource,
          CreateModelResources(
              sc,
              std::make_unique<core::proto::ExternalFile>(
                  std::move(custom_gesture_classifier_model_asset)),
              "_custom_gesture_classifier"));
    }
    return sub_task_model_resources;
  }

  absl::StatusOr<Source<ClassificationList>> BuildGestureRecognizerGraph(
      const HandGestureRecognizerGraphOptions& graph_options,
      const SubTaskModelResources& sub_task_model_resources,
      Source<ClassificationList> handedness,
      Source<NormalizedLandmarkList> hand_landmarks,
      Source<LandmarkList> hand_world_landmarks,
      Source<std::pair<int, int>> image_size, Source<NormalizedRect> norm_rect,
      Graph& graph) {
    // Converts the ClassificationList to a matrix.
    auto& handedness_to_matrix = graph.AddNode("HandednessToMatrixCalculator");
    handedness >> handedness_to_matrix.In(kHandednessTag);
    auto handedness_matrix =
        handedness_to_matrix[Output<Matrix>(kHandednessMatrixTag)];

    // Converts the handedness matrix to a tensor for the inference
    // calculator.
    auto handedness_tensors = ConvertMatrixToTensor(handedness_matrix, graph);

    //  Converts the screen landmarks to a matrix.
    LandmarksToMatrixCalculatorOptions landmarks_options;
    landmarks_options.set_object_normalization(true);
    landmarks_options.set_object_normalization_origin_offset(0);
    auto& hand_landmarks_to_matrix =
        graph.AddNode("LandmarksToMatrixCalculator");
    hand_landmarks_to_matrix.GetOptions<LandmarksToMatrixCalculatorOptions>() =
        landmarks_options;
    hand_landmarks >> hand_landmarks_to_matrix.In(kLandmarksTag);
    image_size >> hand_landmarks_to_matrix.In(kImageSizeTag);
    norm_rect >> hand_landmarks_to_matrix.In(kNormRectTag);
    auto hand_landmarks_matrix =
        hand_landmarks_to_matrix[Output<Matrix>(kLandmarksMatrixTag)];

    // Converts the landmarks matrix to a tensor for the inference calculator.
    auto hand_landmarks_tensor =
        ConvertMatrixToTensor(hand_landmarks_matrix, graph);

    // Converts the world landmarks to a matrix.
    auto& hand_world_landmarks_to_matrix =
        graph.AddNode("LandmarksToMatrixCalculator");
    hand_world_landmarks_to_matrix
        .GetOptions<LandmarksToMatrixCalculatorOptions>() = landmarks_options;
    hand_world_landmarks >>
        hand_world_landmarks_to_matrix.In(kWorldLandmarksTag);
    image_size >> hand_world_landmarks_to_matrix.In(kImageSizeTag);
    norm_rect >> hand_world_landmarks_to_matrix.In(kNormRectTag);
    auto hand_world_landmarks_matrix =
        hand_world_landmarks_to_matrix[Output<Matrix>(kLandmarksMatrixTag)];

    // Converts the world landmarks matrix to a tensor for the inference
    // calculator.
    auto hand_world_landmarks_tensor =
        ConvertMatrixToTensor(hand_world_landmarks_matrix, graph);

    // Converts a tensor into a vector of tensors for the inference
    // calculator.
    auto& concatenate_tensor_vector =
        graph.AddNode("ConcatenateTensorVectorCalculator");
    hand_landmarks_tensor >> concatenate_tensor_vector.In(0);
    handedness_tensors >> concatenate_tensor_vector.In(1);
    hand_world_landmarks_tensor >> concatenate_tensor_vector.In(2);
    auto concatenated_tensors = concatenate_tensor_vector.Out("");

    // Inference for gesture embedder.
    auto& gesture_embedder_inference =
        AddInference(*sub_task_model_resources.gesture_embedder_model_resource,
                     graph_options.gesture_embedder_graph_options()
                         .base_options()
                         .acceleration(),
                     graph);
    concatenated_tensors >> gesture_embedder_inference.In(kTensorsTag);
    auto embedding_tensors =
        gesture_embedder_inference.Out(kTensorsTag).Cast<Tensor>();

    auto& combine_predictions = graph.AddNode("CombinedPredictionCalculator");
    MP_RETURN_IF_ERROR(ConfigureCombinedPredictionCalculator(
        &combine_predictions
             .GetOptions<CombinedPredictionCalculatorOptions>()));

    int classifier_nums = 0;
    // Inference for custom gesture classifier if it exists.
    if (has_custom_gesture_classifier) {
      MP_ASSIGN_OR_RETURN(
          auto gesture_classification_list,
          GetGestureClassificationList(
              sub_task_model_resources.custom_gesture_classifier_model_resource,
              graph_options.custom_gesture_classifier_graph_options(),
              embedding_tensors, graph));
      gesture_classification_list >> combine_predictions.In(classifier_nums++);
    }

    // Inference for canned gesture classifier.
    MP_ASSIGN_OR_RETURN(
        auto gesture_classification_list,
        GetGestureClassificationList(
            sub_task_model_resources.canned_gesture_classifier_model_resource,
            graph_options.canned_gesture_classifier_graph_options(),
            embedding_tensors, graph));
    gesture_classification_list >> combine_predictions.In(classifier_nums++);

    auto combined_classification_list =
        combine_predictions.Out(kPredictionTag).Cast<ClassificationList>();

    return combined_classification_list;
  }

  absl::StatusOr<Source<ClassificationList>> GetGestureClassificationList(
      const core::ModelResources* model_resources,
      const proto::GestureClassifierGraphOptions& options,
      Source<Tensor>& embedding_tensors, Graph& graph) {
    auto& gesture_classifier_inference = AddInference(
        *model_resources, options.base_options().acceleration(), graph);
    embedding_tensors >> gesture_classifier_inference.In(kTensorsTag);
    auto gesture_inference_out_tensors =
        gesture_classifier_inference.Out(kTensorsTag);
    auto& tensors_to_classification =
        graph.AddNode("TensorsToClassificationCalculator");
    MP_RETURN_IF_ERROR(ConfigureTensorsToClassificationCalculator(
        options.classifier_options(), *model_resources->GetMetadataExtractor(),
        0,
        &tensors_to_classification.GetOptions<
            mediapipe::TensorsToClassificationCalculatorOptions>()));
    gesture_inference_out_tensors >> tensors_to_classification.In(kTensorsTag);
    return tensors_to_classification.Out("CLASSIFICATIONS")
        .Cast<ClassificationList>();
  }

  bool has_custom_gesture_classifier = false;
};

// clang-format off
REGISTER_MEDIAPIPE_GRAPH(
  ::mediapipe::tasks::vision::gesture_recognizer::SingleHandGestureRecognizerGraph);  // NOLINT
// clang-format on

// A
// "mediapipe.tasks.vision.gesture_recognizer.MultipleHandGestureRecognizerGraph"
// performs multi hand gesture recognition. This graph is used as a building
// block for mediapipe.tasks.vision.gesture_recognizer.GestureRecognizerGraph.
//
// Inputs:
//   HANDEDNESS - std::vector<ClassificationList>
//     A vector of Classification of handedness.
//   LANDMARKS - std::vector<NormalizedLandmarkList>
//     A vector hand landmarks in normalized image coordinates.
//   WORLD_LANDMARKS - std::vector<LandmarkList>
//     A vector hand landmarks in world coordinates.
//   IMAGE_SIZE - std::pair<int, int>
//     The size of image from which the landmarks detected from.
//   NORM_RECT - NormalizedRect
//     NormalizedRect whose 'rotation' field is used to rotate the
//     landmarks before processing them.
//   HAND_TRACKING_IDS - std::vector<int>
//     A vector of the tracking ids of the hands. The tracking id is the vector
//     index corresponding to the same hand if the graph runs multiple times.
//
// Outputs:
//   HAND_GESTURES - std::vector<ClassificationList>
//     A vector of recognized hand gestures. Each vector element is the
//     ClassificationList of the hand in input vector.
//
//
// Example:
// node {
//   calculator:
//   "mediapipe.tasks.vision.gesture_recognizer.MultipleHandGestureRecognizerGraph"
//   input_stream: "HANDEDNESS:handedness"
//   input_stream: "LANDMARKS:landmarks"
//   input_stream: "WORLD_LANDMARKS:world_landmarks"
//   input_stream: "IMAGE_SIZE:image_size"
//   input_stream: "NORM_RECT:norm_rect"
//   input_stream: "HAND_TRACKING_IDS:hand_tracking_ids"
//   output_stream: "HAND_GESTURES:hand_gestures"
//   options {
//     [mediapipe.tasks.vision.gesture_recognizer.proto.MultipleHandGestureRecognizerGraph.ext]
//     {
//       base_options {
//         model_asset {
//           file_name: "hand_gesture.tflite"
//         }
//       }
//     }
//   }
// }
class MultipleHandGestureRecognizerGraph : public core::ModelTaskGraph {
 public:
  absl::StatusOr<CalculatorGraphConfig> GetConfig(
      SubgraphContext* sc) override {
    Graph graph;
    MP_ASSIGN_OR_RETURN(
        auto multi_hand_gestures,
        BuildMultiGestureRecognizerSubraph(
            sc->Options<HandGestureRecognizerGraphOptions>(),
            graph[Input<std::vector<ClassificationList>>(kHandednessTag)],
            graph[Input<std::vector<NormalizedLandmarkList>>(kLandmarksTag)],
            graph[Input<std::vector<LandmarkList>>(kWorldLandmarksTag)],
            graph[Input<std::pair<int, int>>(kImageSizeTag)],
            graph[Input<NormalizedRect>(kNormRectTag)],
            graph[Input<std::vector<int>>(kHandTrackingIdsTag)], graph));
    multi_hand_gestures >>
        graph[Output<std::vector<ClassificationList>>(kHandGesturesTag)];
    return graph.GetConfig();
  }

 private:
  absl::StatusOr<Source<std::vector<ClassificationList>>>
  BuildMultiGestureRecognizerSubraph(
      const HandGestureRecognizerGraphOptions& graph_options,
      Source<std::vector<ClassificationList>> multi_handedness,
      Source<std::vector<NormalizedLandmarkList>> multi_hand_landmarks,
      Source<std::vector<LandmarkList>> multi_hand_world_landmarks,
      Source<std::pair<int, int>> image_size, Source<NormalizedRect> norm_rect,
      Source<std::vector<int>> multi_hand_tracking_ids, Graph& graph) {
    auto& begin_loop_int = graph.AddNode("BeginLoopIntCalculator");
    image_size >> begin_loop_int.In(kCloneTag)[0];
    norm_rect >> begin_loop_int.In(kCloneTag)[1];
    multi_handedness >> begin_loop_int.In(kCloneTag)[2];
    multi_hand_landmarks >> begin_loop_int.In(kCloneTag)[3];
    multi_hand_world_landmarks >> begin_loop_int.In(kCloneTag)[4];
    multi_hand_tracking_ids >> begin_loop_int.In(kIterableTag);
    auto image_size_clone = begin_loop_int.Out(kCloneTag)[0];
    auto norm_rect_clone = begin_loop_int.Out(kCloneTag)[1];
    auto multi_handedness_clone = begin_loop_int.Out(kCloneTag)[2];
    auto multi_hand_landmarks_clone = begin_loop_int.Out(kCloneTag)[3];
    auto multi_hand_world_landmarks_clone = begin_loop_int.Out(kCloneTag)[4];
    auto hand_tracking_id = begin_loop_int.Out(kItemTag);
    auto batch_end = begin_loop_int.Out(kBatchEndTag);

    auto& get_handedness_at_index =
        graph.AddNode("GetClassificationListVectorItemCalculator");
    multi_handedness_clone >> get_handedness_at_index.In(kVectorTag);
    hand_tracking_id >> get_handedness_at_index.In(kIndexTag);
    auto handedness = get_handedness_at_index.Out(kItemTag);

    auto& get_landmarks_at_index =
        graph.AddNode("GetNormalizedLandmarkListVectorItemCalculator");
    multi_hand_landmarks_clone >> get_landmarks_at_index.In(kVectorTag);
    hand_tracking_id >> get_landmarks_at_index.In(kIndexTag);
    auto hand_landmarks = get_landmarks_at_index.Out(kItemTag);

    auto& get_world_landmarks_at_index =
        graph.AddNode("GetLandmarkListVectorItemCalculator");
    multi_hand_world_landmarks_clone >>
        get_world_landmarks_at_index.In(kVectorTag);
    hand_tracking_id >> get_world_landmarks_at_index.In(kIndexTag);
    auto hand_world_landmarks = get_world_landmarks_at_index.Out(kItemTag);

    auto& hand_gesture_recognizer_graph = graph.AddNode(
        "mediapipe.tasks.vision.gesture_recognizer."
        "SingleHandGestureRecognizerGraph");
    hand_gesture_recognizer_graph
        .GetOptions<HandGestureRecognizerGraphOptions>()
        .CopyFrom(graph_options);
    handedness >> hand_gesture_recognizer_graph.In(kHandednessTag);
    hand_landmarks >> hand_gesture_recognizer_graph.In(kLandmarksTag);
    hand_world_landmarks >>
        hand_gesture_recognizer_graph.In(kWorldLandmarksTag);
    image_size_clone >> hand_gesture_recognizer_graph.In(kImageSizeTag);
    norm_rect_clone >> hand_gesture_recognizer_graph.In(kNormRectTag);
    auto hand_gestures = hand_gesture_recognizer_graph.Out(kHandGesturesTag);

    auto& end_loop_classification_lists =
        graph.AddNode("EndLoopClassificationListCalculator");
    batch_end >> end_loop_classification_lists.In(kBatchEndTag);
    hand_gestures >> end_loop_classification_lists.In(kItemTag);
    auto multi_hand_gestures =
        end_loop_classification_lists[Output<std::vector<ClassificationList>>(
            kIterableTag)];

    return multi_hand_gestures;
  }
};

// clang-format off
REGISTER_MEDIAPIPE_GRAPH(
  ::mediapipe::tasks::vision::gesture_recognizer::MultipleHandGestureRecognizerGraph);  // NOLINT
// clang-format on

}  // namespace gesture_recognizer
}  // namespace vision
}  // namespace tasks
}  // namespace mediapipe
