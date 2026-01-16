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

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "absl/status/status.h"
#include "absl/strings/numbers.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_split.h"
#include "absl/strings/string_view.h"
#include "mediapipe/calculators/tensor/image_to_tensor_calculator.pb.h"
#include "mediapipe/calculators/tensor/inference_calculator.pb.h"
#include "mediapipe/framework/api2/builder.h"
#include "mediapipe/framework/api2/port.h"
#include "mediapipe/framework/calculator.pb.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/deps/file_path.h"
#include "mediapipe/framework/formats/image.h"
#include "mediapipe/framework/formats/tensor.h"
#include "mediapipe/framework/port/status_macros.h"
#include "mediapipe/framework/tool/switch_container.pb.h"
#include "mediapipe/tasks/cc/core/model_asset_bundle_resources.h"
#include "mediapipe/tasks/cc/core/model_resources.h"
#include "mediapipe/tasks/cc/core/model_task_graph.h"
#include "mediapipe/tasks/cc/core/proto/external_file.pb.h"
#include "mediapipe/tasks/cc/vision/image_generator/diffuser/stable_diffusion_iterate_calculator.pb.h"
#include "mediapipe/tasks/cc/vision/image_generator/proto/conditioned_image_graph_options.pb.h"
#include "mediapipe/tasks/cc/vision/image_generator/proto/control_plugin_graph_options.pb.h"
#include "mediapipe/tasks/cc/vision/image_generator/proto/image_generator_graph_options.pb.h"
#include "mediapipe/util/graph_builder_utils.h"

namespace mediapipe {
namespace tasks {
namespace vision {
namespace image_generator {

namespace {

using ::mediapipe::api2::Input;
using ::mediapipe::api2::Output;
using ::mediapipe::api2::builder::Graph;
using ::mediapipe::api2::builder::Source;

constexpr int kPluginsOutputSize = 512;
constexpr absl::string_view kTensorsTag = "TENSORS";
constexpr absl::string_view kImageTag = "IMAGE";
constexpr absl::string_view kImageCpuTag = "IMAGE_CPU";
constexpr absl::string_view kStepsTag = "STEPS";
constexpr absl::string_view kIterationTag = "ITERATION";
constexpr absl::string_view kPromptTag = "PROMPT";
constexpr absl::string_view kRandSeedTag = "RAND_SEED";
constexpr absl::string_view kPluginTensorsTag = "PLUGIN_TENSORS";
constexpr absl::string_view kConditionImageTag = "CONDITION_IMAGE";
constexpr absl::string_view kSelectTag = "SELECT";
constexpr absl::string_view kShowResultTag = "SHOW_RESULT";
constexpr absl::string_view kMetadataFilename = "metadata";
constexpr absl::string_view kLoraRankStr = "lora_rank";

struct ImageGeneratorInputs {
  Source<std::string> prompt;
  Source<int> steps;
  Source<int> iteration;
  Source<int> rand_seed;
  std::optional<Source<Image>> condition_image;
  std::optional<Source<int>> select_condition_type;
  std::optional<Source<bool>> show_result;
};

struct ImageGeneratorOutputs {
  Source<Image> generated_image;
};

}  // namespace

// A container graph containing several ConditionedImageGraph from which to
// choose specified condition type.
// Inputs:
//   IMAGE - Image
//     The source condition image, used to generate the condition image.
//   SELECT - int
//     The index of the selected conditioned image graph.
// Outputs:
//   CONDITION_IMAGE - Image
//     The condition image created from the specified condition type.
class ConditionedImageGraphContainer : public core::ModelTaskGraph {
 public:
  absl::StatusOr<CalculatorGraphConfig> GetConfig(
      SubgraphContext* sc) override {
    Graph graph;
    auto& graph_options =
        *sc->MutableOptions<proto::ImageGeneratorGraphOptions>();
    auto source_condition_image = graph.In(kImageTag).Cast<Image>();
    auto select_condition_type = graph.In(kSelectTag).Cast<int>();
    auto& switch_container = graph.AddNode("SwitchContainer");
    auto& switch_options =
        switch_container.GetOptions<mediapipe::SwitchContainerOptions>();
    for (auto& control_plugin_graph_options :
         *graph_options.mutable_control_plugin_graphs_options()) {
      auto& node = *switch_options.add_contained_node();
      node.set_calculator(
          "mediapipe.tasks.vision.image_generator.ConditionedImageGraph");
      node.mutable_node_options()->Add()->PackFrom(
          control_plugin_graph_options.conditioned_image_graph_options());
    }
    source_condition_image >> switch_container.In(kImageTag);
    select_condition_type >> switch_container.In(kSelectTag);
    auto condition_image = switch_container.Out(kImageTag).Cast<Image>();
    condition_image >> graph.Out(kConditionImageTag);
    return graph.GetConfig();
  }
};

// clang-format off
REGISTER_MEDIAPIPE_GRAPH(
  ::mediapipe::tasks::vision::image_generator::ConditionedImageGraphContainer); // NOLINT
// clang-format on

// A helper graph to convert condition image to Tensor using the control plugin
// model.
// Inputs:
//   CONDITION_IMAGE - Image
//     The condition image input to the control plugin model.
// Outputs:
//   PLUGIN_TENSORS - std::vector<Tensor>
//     The output tensors from the control plugin model. The tensors are used as
//     inputs to the image generation model.
class ControlPluginGraph : public core::ModelTaskGraph {
 public:
  absl::StatusOr<CalculatorGraphConfig> GetConfig(
      SubgraphContext* sc) override {
    Graph graph;
    auto& graph_options =
        *sc->MutableOptions<proto::ControlPluginGraphOptions>();

    auto condition_image = graph.In(kConditionImageTag).Cast<Image>();

    // Convert Image to ImageFrame.
    auto& from_image = graph.AddNode("FromImageCalculator");
    condition_image >> from_image.In(kImageTag);
    auto image_frame = from_image.Out(kImageCpuTag);

    // Convert ImageFrame to Tensor.
    auto& image_to_tensor = graph.AddNode("ImageToTensorCalculator");
    auto& image_to_tensor_options =
        image_to_tensor.GetOptions<mediapipe::ImageToTensorCalculatorOptions>();
    image_to_tensor_options.set_output_tensor_width(kPluginsOutputSize);
    image_to_tensor_options.set_output_tensor_height(kPluginsOutputSize);
    image_to_tensor_options.mutable_output_tensor_float_range()->set_min(-1);
    image_to_tensor_options.mutable_output_tensor_float_range()->set_max(1);
    image_to_tensor_options.set_keep_aspect_ratio(true);
    image_frame >> image_to_tensor.In(kImageTag);

    // Create the plugin model resource.
    MP_ASSIGN_OR_RETURN(
        const core::ModelResources* plugin_model_resources,
        CreateModelResources(
            sc,
            std::make_unique<tasks::core::proto::ExternalFile>(
                *graph_options.mutable_base_options()->mutable_model_asset())));

    // Add control plugin model inference.
    auto& plugins_inference =
        AddInference(*plugin_model_resources,
                     graph_options.base_options().acceleration(), graph);
    image_to_tensor.Out(kTensorsTag) >> plugins_inference.In(kTensorsTag);
    // The plugins model is not runnable on OpenGL. Error message:
    // TfLiteGpuDelegate Prepare: Batch size mismatch, expected 1 but got 64
    // Node number 67 (TfLiteGpuDelegate) failed to prepare.
    plugins_inference.GetOptions<mediapipe::InferenceCalculatorOptions>()
        .mutable_delegate()
        ->mutable_xnnpack();
    plugins_inference.Out(kTensorsTag).Cast<std::vector<Tensor>>() >>
        graph.Out(kPluginTensorsTag);
    return graph.GetConfig();
  }
};

REGISTER_MEDIAPIPE_GRAPH(
    ::mediapipe::tasks::vision::image_generator::ControlPluginGraph);

// A "mediapipe.tasks.vision.image_generator.ImageGeneratorGraph" performs image
// generation from a text prompt, and a optional condition image.
//
// Inputs:
//   PROMPT - std::string
//     The prompt describing the image to be generated.
//   STEPS - int
//     The total steps to generate the image.
//   ITERATION - int
//     The current iteration in the generating steps. Must be less than STEPS.
//   RAND_SEED - int
//     The randaom seed input to the image generation model.
//   CONDITION_IMAGE - Image
//     The condition image used as a guidance for the image generation. Only
//     valid, if condtrol plugin graph options are set in the graph options.
//   SELECT - int
//     The index of the selected the control plugin graph.
//   SHOW_RESULT - bool @Optional
//     Whether to show the diffusion result at the current step. If this stream
//     is not empty, regardless show_every_n_iteration in the options.
//
// Outputs:
//   IMAGE - Image
//     The generated image.
//   STEPS - int @optional
//    The total steps to generate the image. The same as STEPS input.
//   ITERATION - int @optional
//    The current iteration in the generating steps. The same as ITERATION
//    input.
//   SHOW_RESULT - bool @Optional
//     Whether to show the diffusion result at the current step. The same as
//     input SHOW_RESULT.
class ImageGeneratorGraph : public core::ModelTaskGraph {
 public:
  absl::StatusOr<CalculatorGraphConfig> GetConfig(
      SubgraphContext* sc) override {
    Graph graph;
    auto* subgraph_options =
        sc->MutableOptions<proto::ImageGeneratorGraphOptions>();
    std::optional<const core::ModelAssetBundleResources*> lora_resources;
    // Create LoRA weights asset bundle resources.
    if (subgraph_options->has_lora_weights_file()) {
      auto external_file = std::make_unique<tasks::core::proto::ExternalFile>();
      external_file->Swap(subgraph_options->mutable_lora_weights_file());
      MP_ASSIGN_OR_RETURN(lora_resources, CreateModelAssetBundleResources(
                                              sc, std::move(external_file)));
    }
    std::optional<Source<Image>> condition_image;
    std::optional<Source<int>> select_condition_type;
    if (!subgraph_options->control_plugin_graphs_options().empty()) {
      condition_image = graph.In(kConditionImageTag).Cast<Image>();
      select_condition_type = graph.In(kSelectTag).Cast<int>();
    }
    std::optional<Source<bool>> show_result;
    if (HasInput(sc->OriginalNode(), kShowResultTag)) {
      show_result = graph.In(kShowResultTag).Cast<bool>();
    }
    MP_ASSIGN_OR_RETURN(
        auto outputs,
        BuildImageGeneratorGraph(
            *sc->MutableOptions<proto::ImageGeneratorGraphOptions>(),
            lora_resources,
            ImageGeneratorInputs{
                /*prompt=*/graph.In(kPromptTag).Cast<std::string>(),
                /*steps=*/graph.In(kStepsTag).Cast<int>(),
                /*iteration=*/graph.In(kIterationTag).Cast<int>(),
                /*rand_seed=*/graph.In(kRandSeedTag).Cast<int>(),
                /*condition_image*/ condition_image,
                /*select_condition_type*/ select_condition_type,
                /*show_result*/ show_result,
            },
            graph));
    outputs.generated_image >> graph.Out(kImageTag).Cast<Image>();

    // Optional outputs to provide the current iteration.
    auto& pass_through = graph.AddNode("PassThroughCalculator");
    graph.In(kIterationTag) >> pass_through.In(0);
    graph.In(kStepsTag) >> pass_through.In(1);
    pass_through.Out(0) >> graph[Output<int>::Optional(kIterationTag)];
    pass_through.Out(1) >> graph[Output<int>::Optional(kStepsTag)];
    if (HasOutput(sc->OriginalNode(), kShowResultTag)) {
      graph.In(kShowResultTag) >> pass_through.In(2);
      pass_through.Out(2) >> graph[Output<bool>::Optional(kShowResultTag)];
    }
    return graph.GetConfig();
  }

  absl::StatusOr<ImageGeneratorOutputs> BuildImageGeneratorGraph(
      proto::ImageGeneratorGraphOptions& subgraph_options,
      std::optional<const core::ModelAssetBundleResources*> lora_resources,
      ImageGeneratorInputs inputs, Graph& graph) {
    auto& stable_diff = graph.AddNode("StableDiffusionIterateCalculator");
    if (inputs.condition_image.has_value()) {
      // Add switch container for multiple control plugin graphs.
      auto& switch_container = graph.AddNode("SwitchContainer");
      auto& switch_options =
          switch_container.GetOptions<mediapipe::SwitchContainerOptions>();
      for (auto& control_plugin_graph_options :
           *subgraph_options.mutable_control_plugin_graphs_options()) {
        auto& node = *switch_options.add_contained_node();
        node.set_calculator(
            "mediapipe.tasks.vision.image_generator.ControlPluginGraph");
        node.mutable_node_options()->Add()->PackFrom(
            control_plugin_graph_options);
      }
      *inputs.condition_image >> switch_container.In(kConditionImageTag);
      *inputs.select_condition_type >> switch_container.In(kSelectTag);
      auto plugin_tensors = switch_container.Out(kPluginTensorsTag);

      // Additional diffusion plugins calculator to pass tensors to diffusion
      // iterator.
      auto& plugins_output = graph.AddNode("DiffusionPluginsOutputCalculator");
      plugin_tensors >> plugins_output.In(kTensorsTag);
      inputs.steps >> plugins_output.In(kStepsTag);
      inputs.iteration >> plugins_output.In(kIterationTag);
      plugins_output.Out(kTensorsTag) >> stable_diff.In(kPluginTensorsTag);
    }

    inputs.prompt >> stable_diff.In(kPromptTag);
    inputs.steps >> stable_diff.In(kStepsTag);
    inputs.iteration >> stable_diff.In(kIterationTag);
    inputs.rand_seed >> stable_diff.In(kRandSeedTag);
    if (inputs.show_result.has_value()) {
      *inputs.show_result >> stable_diff.In(kShowResultTag);
    }
    mediapipe::StableDiffusionIterateCalculatorOptions& options =
        stable_diff
            .GetOptions<mediapipe::StableDiffusionIterateCalculatorOptions>();
    if (subgraph_options.has_stable_diffusion_iterate_options()) {
      options = subgraph_options.stable_diffusion_iterate_options();
    } else {
      options.set_base_seed(0);
      options.set_output_image_height(kPluginsOutputSize);
      options.set_output_image_width(kPluginsOutputSize);
      options.set_file_folder(subgraph_options.text2image_model_directory());
      options.set_show_every_n_iteration(100);
      options.set_emit_empty_packet(true);
    }
    if (lora_resources.has_value()) {
      auto& lora_layer_weights_mapping =
          *options.mutable_lora_weights_layer_mapping();
      for (const auto& file_path : (*lora_resources)->ListFiles()) {
        auto basename = file::Basename(file_path);
        MP_ASSIGN_OR_RETURN(auto file_content,
                            (*lora_resources)->GetFile(std::string(file_path)));
        if (file_path == kMetadataFilename) {
          MP_RETURN_IF_ERROR(
              ParseLoraMetadataAndConfigOptions(file_content, options));
        } else {
          lora_layer_weights_mapping[basename] =
              reinterpret_cast<uint64_t>(file_content.data());
        }
      }
    }

    auto& to_image = graph.AddNode("ToImageCalculator");
    stable_diff.Out(kImageTag) >> to_image.In(kImageCpuTag);

    return {{to_image.Out(kImageTag).Cast<Image>()}};
  }

 private:
  absl::Status ParseLoraMetadataAndConfigOptions(
      absl::string_view contents,
      mediapipe::StableDiffusionIterateCalculatorOptions& options) {
    std::vector<absl::string_view> lines =
        absl::StrSplit(contents, '\n', absl::SkipEmpty());
    for (const auto& line : lines) {
      std::vector<absl::string_view> values = absl::StrSplit(line, ',');
      if (values[0] == kLoraRankStr) {
        int lora_rank;
        if (values.size() != 2 || !absl::SimpleAtoi(values[1], &lora_rank)) {
          return absl::InvalidArgumentError(
              absl::StrCat("Error parsing LoRA weights metadata. ", line));
        }
        options.set_lora_rank(lora_rank);
      }
    }
    return absl::OkStatus();
  }
};

REGISTER_MEDIAPIPE_GRAPH(
    ::mediapipe::tasks::vision::image_generator::ImageGeneratorGraph);

}  // namespace image_generator
}  // namespace vision
}  // namespace tasks
}  // namespace mediapipe
