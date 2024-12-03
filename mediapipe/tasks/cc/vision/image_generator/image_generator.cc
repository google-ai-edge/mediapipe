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

#include "mediapipe/tasks/cc/vision/image_generator/image_generator.h"

#include <memory>
#include <string>

#include "absl/status/status.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "absl/time/time.h"
#include "mediapipe/framework/api2/builder.h"
#include "mediapipe/framework/api2/port.h"
#include "mediapipe/framework/packet.h"
#include "mediapipe/framework/timestamp.h"
#include "mediapipe/tasks/cc/core/proto/external_file.pb.h"
#include "mediapipe/tasks/cc/core/task_runner.h"
#include "mediapipe/tasks/cc/vision/core/vision_task_api_factory.h"
#include "mediapipe/tasks/cc/vision/face_detector/proto/face_detector_graph_options.pb.h"
#include "mediapipe/tasks/cc/vision/face_landmarker/proto/face_landmarker_graph_options.pb.h"
#include "mediapipe/tasks/cc/vision/face_landmarker/proto/face_landmarks_detector_graph_options.pb.h"
#include "mediapipe/tasks/cc/vision/image_generator/diffuser/stable_diffusion_iterate_calculator.pb.h"
#include "mediapipe/tasks/cc/vision/image_generator/image_generator_result.h"
#include "mediapipe/tasks/cc/vision/image_generator/proto/conditioned_image_graph_options.pb.h"
#include "mediapipe/tasks/cc/vision/image_generator/proto/control_plugin_graph_options.pb.h"
#include "mediapipe/tasks/cc/vision/image_generator/proto/image_generator_graph_options.pb.h"
#include "mediapipe/tasks/cc/vision/image_segmenter/proto/image_segmenter_graph_options.pb.h"

namespace mediapipe {
namespace tasks {
namespace vision {
namespace image_generator {
namespace {

using ImageGeneratorGraphOptionsProto = ::mediapipe::tasks::vision::
    image_generator::proto::ImageGeneratorGraphOptions;
using ConditionedImageGraphOptionsProto = ::mediapipe::tasks::vision::
    image_generator::proto::ConditionedImageGraphOptions;
using ControlPluginGraphOptionsProto = ::mediapipe::tasks::vision::
    image_generator::proto::ControlPluginGraphOptions;
using FaceLandmarkerGraphOptionsProto = ::mediapipe::tasks::vision::
    face_landmarker::proto::FaceLandmarkerGraphOptions;

constexpr absl::string_view kImageTag = "IMAGE";
constexpr absl::string_view kImageOutName = "image_out";
constexpr absl::string_view kConditionImageTag = "CONDITION_IMAGE";
constexpr absl::string_view kConditionImageName = "condition_image";
constexpr absl::string_view kSourceConditionImageName =
    "source_condition_image";
constexpr absl::string_view kStepsTag = "STEPS";
constexpr absl::string_view kStepsName = "steps";
constexpr absl::string_view kIterationTag = "ITERATION";
constexpr absl::string_view kIterationName = "iteration";
constexpr absl::string_view kPromptTag = "PROMPT";
constexpr absl::string_view kPromptName = "prompt";
constexpr absl::string_view kRandSeedTag = "RAND_SEED";
constexpr absl::string_view kRandSeedName = "rand_seed";
constexpr absl::string_view kSelectTag = "SELECT";
constexpr absl::string_view kSelectName = "select";

constexpr char kImageGeneratorGraphTypeName[] =
    "mediapipe.tasks.vision.image_generator.ImageGeneratorGraph";

constexpr char kConditionedImageGraphContainerTypeName[] =
    "mediapipe.tasks.vision.image_generator.ConditionedImageGraphContainer";

// Creates a MediaPipe graph config that contains a subgraph node of
// "mediapipe.tasks.vision.image_generator.ImageGeneratorGraph".
CalculatorGraphConfig CreateImageGeneratorGraphConfig(
    std::unique_ptr<ImageGeneratorGraphOptionsProto> options,
    bool use_condition_image) {
  api2::builder::Graph graph;
  auto& subgraph = graph.AddNode(kImageGeneratorGraphTypeName);
  subgraph.GetOptions<ImageGeneratorGraphOptionsProto>().CopyFrom(*options);
  graph.In(kStepsTag).SetName(kStepsName) >> subgraph.In(kStepsTag);
  graph.In(kIterationTag).SetName(kIterationName) >> subgraph.In(kIterationTag);
  graph.In(kPromptTag).SetName(kPromptName) >> subgraph.In(kPromptTag);
  graph.In(kRandSeedTag).SetName(kRandSeedName) >> subgraph.In(kRandSeedTag);
  if (use_condition_image) {
    graph.In(kConditionImageTag).SetName(kConditionImageName) >>
        subgraph.In(kConditionImageTag);
    graph.In(kSelectTag).SetName(kSelectName) >> subgraph.In(kSelectTag);
  }
  subgraph.Out(kImageTag).SetName(kImageOutName) >>
      graph[api2::Output<Image>::Optional(kImageTag)];
  return graph.GetConfig();
}

// Creates a MediaPipe graph config that contains a subgraph node of
// "mediapipe.tasks.vision.image_generator.ConditionedImageGraphContainer".
CalculatorGraphConfig CreateConditionedImageGraphContainerConfig(
    std::unique_ptr<ImageGeneratorGraphOptionsProto> options) {
  api2::builder::Graph graph;
  auto& subgraph = graph.AddNode(kConditionedImageGraphContainerTypeName);
  subgraph.GetOptions<ImageGeneratorGraphOptionsProto>().CopyFrom(*options);
  graph.In(kImageTag).SetName(kSourceConditionImageName) >>
      subgraph.In(kImageTag);
  graph.In(kSelectTag).SetName(kSelectName) >> subgraph.In(kSelectTag);
  subgraph.Out(kConditionImageTag).SetName(kConditionImageName) >>
      graph.Out(kConditionImageTag).Cast<Image>();
  return graph.GetConfig();
}

absl::Status SetFaceConditionOptionsToProto(
    FaceConditionOptions& face_condition_options,
    ControlPluginGraphOptionsProto& options_proto) {
  // Configure face plugin model.
  auto plugin_base_options_proto =
      std::make_unique<tasks::core::proto::BaseOptions>(
          tasks::core::ConvertBaseOptionsToProto(
              &(face_condition_options.base_options)));
  options_proto.mutable_base_options()->Swap(plugin_base_options_proto.get());

  // Configure face landmarker graph.
  auto& face_landmarker_options =
      face_condition_options.face_landmarker_options;
  auto& face_landmarker_options_proto =
      *options_proto.mutable_conditioned_image_graph_options()
           ->mutable_face_condition_type_options()
           ->mutable_face_landmarker_graph_options();

  auto base_options_proto = std::make_unique<tasks::core::proto::BaseOptions>(
      tasks::core::ConvertBaseOptionsToProto(
          &(face_landmarker_options.base_options)));
  face_landmarker_options_proto.mutable_base_options()->Swap(
      base_options_proto.get());
  face_landmarker_options_proto.mutable_base_options()->set_use_stream_mode(
      false);

  // Configure face detector options.
  auto* face_detector_graph_options =
      face_landmarker_options_proto.mutable_face_detector_graph_options();
  face_detector_graph_options->set_num_faces(face_landmarker_options.num_faces);
  face_detector_graph_options->set_min_detection_confidence(
      face_landmarker_options.min_face_detection_confidence);

  // Configure face landmark detector options.
  face_landmarker_options_proto.set_min_tracking_confidence(
      face_landmarker_options.min_tracking_confidence);
  auto* face_landmarks_detector_graph_options =
      face_landmarker_options_proto
          .mutable_face_landmarks_detector_graph_options();
  face_landmarks_detector_graph_options->set_min_detection_confidence(
      face_landmarker_options.min_face_presence_confidence);
  return absl::OkStatus();
}

absl::Status SetDepthConditionOptionsToProto(
    DepthConditionOptions& depth_condition_options,
    ControlPluginGraphOptionsProto& options_proto) {
  // Configure face plugin model.
  auto plugin_base_options_proto =
      std::make_unique<tasks::core::proto::BaseOptions>(
          tasks::core::ConvertBaseOptionsToProto(
              &(depth_condition_options.base_options)));
  options_proto.mutable_base_options()->Swap(plugin_base_options_proto.get());

  auto& image_segmenter_graph_options =
      *options_proto.mutable_conditioned_image_graph_options()
           ->mutable_depth_condition_type_options()
           ->mutable_image_segmenter_graph_options();

  auto depth_base_options_proto =
      std::make_unique<tasks::core::proto::BaseOptions>(
          tasks::core::ConvertBaseOptionsToProto(
              &(depth_condition_options.image_segmenter_options.base_options)));
  image_segmenter_graph_options.mutable_base_options()->Swap(
      depth_base_options_proto.get());
  image_segmenter_graph_options.mutable_base_options()->set_use_stream_mode(
      false);
  image_segmenter_graph_options.set_display_names_locale(
      depth_condition_options.image_segmenter_options.display_names_locale);
  return absl::OkStatus();
}

absl::Status SetEdgeConditionOptionsToProto(
    EdgeConditionOptions& edge_condition_options,
    ControlPluginGraphOptionsProto& options_proto) {
  auto plugin_base_options_proto =
      std::make_unique<tasks::core::proto::BaseOptions>(
          tasks::core::ConvertBaseOptionsToProto(
              &(edge_condition_options.base_options)));
  options_proto.mutable_base_options()->Swap(plugin_base_options_proto.get());

  auto& edge_options_proto =
      *options_proto.mutable_conditioned_image_graph_options()
           ->mutable_edge_condition_type_options();
  edge_options_proto.set_threshold_1(edge_condition_options.threshold_1);
  edge_options_proto.set_threshold_2(edge_condition_options.threshold_2);
  edge_options_proto.set_aperture_size(edge_condition_options.aperture_size);
  edge_options_proto.set_l2_gradient(edge_condition_options.l2_gradient);
  return absl::OkStatus();
}

// Helper holder struct of image generator graph options and condition type
// index mapping.
struct ImageGeneratorOptionsProtoAndConditionTypeIndex {
  std::unique_ptr<ImageGeneratorGraphOptionsProto> options_proto;
  std::unique_ptr<std::map<ConditionOptions::ConditionType, int>>
      condition_type_index;
};

// Converts the user-facing ImageGeneratorOptions struct to the internal
// ImageGeneratorOptions proto.
absl::StatusOr<ImageGeneratorOptionsProtoAndConditionTypeIndex>
ConvertImageGeneratorGraphOptionsProto(
    ImageGeneratorOptions* image_generator_options,
    ConditionOptions* condition_options) {
  ImageGeneratorOptionsProtoAndConditionTypeIndex
      options_proto_and_condition_index;

  // Configure base image generator options.
  options_proto_and_condition_index.options_proto =
      std::make_unique<ImageGeneratorGraphOptionsProto>();
  auto& options_proto = *options_proto_and_condition_index.options_proto;
  options_proto.set_text2image_model_directory(
      image_generator_options->text2image_model_directory);
  options_proto.mutable_stable_diffusion_iterate_options()->set_file_folder(
      image_generator_options->text2image_model_directory);
  switch (image_generator_options->model_type) {
    case ImageGeneratorOptions::ModelType::SD_1:
      options_proto.mutable_stable_diffusion_iterate_options()->set_model_type(
          mediapipe::StableDiffusionIterateCalculatorOptions::SD_1);
      break;
    default:
      return absl::InvalidArgumentError(
          absl::StrFormat("Unsupported ImageGenerator model type: %d",
                          image_generator_options->model_type));
  }
  if (image_generator_options->lora_weights_file_path.has_value()) {
    options_proto.mutable_lora_weights_file()->set_file_name(
        *image_generator_options->lora_weights_file_path);
  }

  // Configure optional condition type options.
  if (condition_options != nullptr) {
    options_proto_and_condition_index.condition_type_index =
        std::make_unique<std::map<ConditionOptions::ConditionType, int>>();
    auto& condition_type_index =
        *options_proto_and_condition_index.condition_type_index;
    if (condition_options->face_condition_options.has_value()) {
      condition_type_index[ConditionOptions::FACE] =
          condition_type_index.size();
      auto& face_plugin_graph_options =
          *options_proto.add_control_plugin_graphs_options();
      RET_CHECK_OK(SetFaceConditionOptionsToProto(
          *condition_options->face_condition_options,
          face_plugin_graph_options));
    }
    if (condition_options->depth_condition_options.has_value()) {
      condition_type_index[ConditionOptions::DEPTH] =
          condition_type_index.size();
      auto& depth_plugin_graph_options =
          *options_proto.add_control_plugin_graphs_options();
      RET_CHECK_OK(SetDepthConditionOptionsToProto(
          *condition_options->depth_condition_options,
          depth_plugin_graph_options));
    }
    if (condition_options->edge_condition_options.has_value()) {
      condition_type_index[ConditionOptions::EDGE] =
          condition_type_index.size();
      auto& edge_plugin_graph_options =
          *options_proto.add_control_plugin_graphs_options();
      RET_CHECK_OK(SetEdgeConditionOptionsToProto(
          *condition_options->edge_condition_options,
          edge_plugin_graph_options));
    }
    if (condition_type_index.empty()) {
      return absl::InvalidArgumentError(
          "At least one condition type must be set.");
    }
  }
  return options_proto_and_condition_index;
}

}  // namespace

absl::StatusOr<std::unique_ptr<ImageGenerator>> ImageGenerator::Create(
    std::unique_ptr<ImageGeneratorOptions> image_generator_options,
    std::unique_ptr<ConditionOptions> condition_options) {
  bool use_condition_image = condition_options != nullptr;
  MP_ASSIGN_OR_RETURN(
      auto options_proto_and_condition_index,
      ConvertImageGeneratorGraphOptionsProto(image_generator_options.get(),
                                             condition_options.get()));
  std::unique_ptr<proto::ImageGeneratorGraphOptions>
      options_proto_for_condition_image_graphs_container;
  if (use_condition_image) {
    options_proto_for_condition_image_graphs_container =
        std::make_unique<proto::ImageGeneratorGraphOptions>();
    options_proto_for_condition_image_graphs_container->CopyFrom(
        *options_proto_and_condition_index.options_proto);
  }
  MP_ASSIGN_OR_RETURN(
      auto image_generator,
      (core::VisionTaskApiFactory::Create<ImageGenerator,
                                          ImageGeneratorGraphOptionsProto>(
          CreateImageGeneratorGraphConfig(
              std::move(options_proto_and_condition_index.options_proto),
              use_condition_image),
          std::make_unique<tasks::core::MediaPipeBuiltinOpResolver>(),
          core::RunningMode::IMAGE,
          /*result_callback=*/nullptr)));
  image_generator->use_condition_image_ = use_condition_image;
  if (use_condition_image) {
    image_generator->condition_type_index_ =
        std::move(options_proto_and_condition_index.condition_type_index);
    MP_ASSIGN_OR_RETURN(
        image_generator->condition_image_graphs_container_task_runner_,
        tasks::core::TaskRunner::Create(
            CreateConditionedImageGraphContainerConfig(
                std::move(options_proto_for_condition_image_graphs_container)),
            absl::make_unique<tasks::core::MediaPipeBuiltinOpResolver>()));
  }
  image_generator->init_timestamp_ = absl::Now();
  return image_generator;
}

absl::StatusOr<Image> ImageGenerator::CreateConditionImage(
    Image source_condition_image,
    ConditionOptions::ConditionType condition_type) {
  if (condition_type_index_->find(condition_type) ==
      condition_type_index_->end()) {
    return absl::InvalidArgumentError(
        "The condition type is not created during initialization.");
  }
  MP_ASSIGN_OR_RETURN(
      auto output_packets,
      condition_image_graphs_container_task_runner_->Process({
          {std::string(kSourceConditionImageName),
           MakePacket<Image>(std::move(source_condition_image))},
          {std::string(kSelectName),
           MakePacket<int>(condition_type_index_->at(condition_type))},
      }));
  return output_packets.at(std::string(kConditionImageName)).Get<Image>();
}

absl::StatusOr<ImageGeneratorResult> ImageGenerator::Generate(
    const std::string& prompt, int iterations, int seed) {
  if (use_condition_image_) {
    return absl::InvalidArgumentError(
        "ImageGenerator is created to use with conditioned image.");
  }
  return RunIterations(prompt, iterations, seed, std::nullopt);
}

absl::StatusOr<ImageGeneratorResult> ImageGenerator::Generate(
    const std::string& prompt, Image condition_image,
    ConditionOptions::ConditionType condition_type, int iterations, int seed) {
  if (!use_condition_image_) {
    return absl::InvalidArgumentError(
        "ImageGenerator is created to use without conditioned image.");
  }
  MP_ASSIGN_OR_RETURN(auto plugin_model_image,
                      CreateConditionImage(condition_image, condition_type));
  return RunIterations(
      prompt, iterations, seed,
      ConditionInputs{plugin_model_image,
                      condition_type_index_->at(condition_type)});
}

absl::StatusOr<ImageGeneratorResult> ImageGenerator::RunIterations(
    const std::string& prompt, int steps, int rand_seed,
    std::optional<ConditionInputs> condition_inputs) {
  tasks::core::PacketMap output_packets;
  ImageGeneratorResult result;
  auto timestamp = (absl::Now() - init_timestamp_) / absl::Milliseconds(1);
  for (int i = 0; i < steps; ++i) {
    tasks::core::PacketMap input_packets;
    if (i == 0 && condition_inputs.has_value()) {
      input_packets[std::string(kConditionImageName)] =
          MakePacket<Image>(condition_inputs->condition_image)
              .At(Timestamp(timestamp));
      input_packets[std::string(kSelectName)] =
          MakePacket<int>(condition_inputs->select).At(Timestamp(timestamp));
    }
    input_packets[std::string(kStepsName)] =
        MakePacket<int>(steps).At(Timestamp(timestamp));
    input_packets[std::string(kIterationName)] =
        MakePacket<int>(i).At(Timestamp(timestamp));
    input_packets[std::string(kPromptName)] =
        MakePacket<std::string>(prompt).At(Timestamp(timestamp));
    input_packets[std::string(kRandSeedName)] =
        MakePacket<int>(rand_seed).At(Timestamp(timestamp));
    MP_ASSIGN_OR_RETURN(output_packets, ProcessImageData(input_packets));
    timestamp += 1;
  }
  result.generated_image =
      output_packets.at(std::string(kImageOutName)).Get<Image>();
  if (condition_inputs.has_value()) {
    result.condition_image = condition_inputs->condition_image;
  }
  return result;
}

}  // namespace image_generator
}  // namespace vision
}  // namespace tasks
}  // namespace mediapipe
