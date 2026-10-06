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
#include <utility>

#include "absl/container/flat_hash_map.h"
#include "absl/log/absl_check.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/match.h"
#include "absl/strings/string_view.h"
#include "mediapipe/calculators/tensor/inference_calculator.pb.h"
#include "mediapipe/framework/api2/builder.h"
#include "mediapipe/framework/api2/port.h"
#include "mediapipe/framework/calculator.pb.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/port/status_macros.h"
#include "mediapipe/tasks/cc/components/calculators/tensors_to_embeddings_calculator.pb.h"
#include "mediapipe/tasks/cc/components/containers/proto/embeddings.pb.h"
#include "mediapipe/tasks/cc/components/processors/embedding_postprocessing_graph.h"
#include "mediapipe/tasks/cc/components/processors/proto/embedding_postprocessing_graph_options.pb.h"
#include "mediapipe/tasks/cc/components/processors/proto/text_model_type.pb.h"
#include "mediapipe/tasks/cc/components/processors/proto/text_preprocessing_graph_options.pb.h"
#include "mediapipe/tasks/cc/components/processors/text_preprocessing_graph.h"
#include "mediapipe/tasks/cc/core/external_file_handler.h"
#include "mediapipe/tasks/cc/core/model_asset_bundle_resources.h"
#include "mediapipe/tasks/cc/core/model_resources.h"
#include "mediapipe/tasks/cc/core/model_resources_cache.h"
#include "mediapipe/tasks/cc/core/model_task_graph.h"
#include "mediapipe/tasks/cc/core/proto/model_resources_calculator.pb.h"
#include "mediapipe/tasks/cc/metadata/utils/zip_utils.h"
#include "mediapipe/tasks/cc/text/text_embedder/proto/text_embedder_graph_options.pb.h"
#include "mediapipe/tasks/cc/text/utils/text_model_utils.h"

namespace mediapipe::tasks::text::text_embedder {
namespace {

using ::mediapipe::api2::Input;
using ::mediapipe::api2::Output;
using ::mediapipe::api2::builder::Graph;
using ::mediapipe::api2::builder::Source;
using ::mediapipe::tasks::components::containers::proto::EmbeddingResult;
using ::mediapipe::tasks::components::processors::proto::TextModelType;
using ::mediapipe::tasks::core::kModelResourcesCacheService;
using ::mediapipe::tasks::core::ModelResources;
using ::mediapipe::tasks::metadata::SetExternalFile;
using ::mediapipe::tasks::text::utils::GetModelType;

constexpr char kEmbeddingsTag[] = "EMBEDDINGS";
constexpr char kTextTag[] = "TEXT";
constexpr char kMetadataExtractorTag[] = "METADATA_EXTRACTOR";
constexpr char kTensorsTag[] = "TENSORS";

constexpr char kGeckoTFLiteName[] = "gecko.tflite";
constexpr char kSentencePieceModelName[] = "sentencepiece.model";

constexpr char kUSEQueryTensorName[] = "query_encoding";

}  // namespace

// A "mediapipe.tasks.text.TextEmbedderGraph" performs text embedding
// extraction.
// - Accepts input text and outputs embeddings on CPU.
//
// Inputs:
//   TEXT - std::string
//     Input text to perform embedding extraction on.
//
// Outputs:
//   EMBEDDINGS - EmbeddingResult
//     The embedding result.
//
// Example:
// node {
//   calculator: "mediapipe.tasks.text.TextEmbedderGraph"
//   input_stream: "TEXT:text_in"
//   output_stream: "EMBEDDINGS:embedding_result_out"
//   options {
//     [mediapipe.tasks.text.text_embedder.proto.TextEmbedderGraphOptions.ext] {
//       base_options {
//         model_asset {
//           file_name: "/path/to/model.tflite"
//         }
//       }
//     }
//   }
// }
class TextEmbedderGraph : public core::ModelTaskGraph {
 public:
  absl::StatusOr<CalculatorGraphConfig> GetConfig(
      SubgraphContext* sc) override {
    ABSL_CHECK(sc != nullptr);

    const ModelResources* model_resources = nullptr;
    MP_RETURN_IF_ERROR(MaybeHandleModelBundle(sc, &model_resources));

    if (model_resources == nullptr) {
      MP_ASSIGN_OR_RETURN(
          model_resources,
          CreateModelResources<proto::TextEmbedderGraphOptions>(sc));
    }

    Graph graph;
    MP_ASSIGN_OR_RETURN(
        Source<EmbeddingResult> embedding_result_out,
        BuildTextEmbedderTask(sc->Options<proto::TextEmbedderGraphOptions>(),
                              *model_resources,
                              graph[Input<std::string>(kTextTag)], graph));
    embedding_result_out >> graph[Output<EmbeddingResult>(kEmbeddingsTag)];
    return graph.GetConfig();
  }

 private:
  // Handles a model bundle by extracting the .tflite and
  // sentencepiece.model files from the bundle and creating ModelResources for
  // the .tflite file. If the model asset is not a model bundle, does
  // nothing and does not update the model_resources pointer.
  absl::Status MaybeHandleModelBundle(SubgraphContext* sc,
                                      const ModelResources** model_resources) {
    *model_resources = nullptr;
    if (!sc->Options<proto::TextEmbedderGraphOptions>()
             .base_options()
             .has_model_asset()) {
      return absl::OkStatus();
    }
    const auto& model_asset = sc->Options<proto::TextEmbedderGraphOptions>()
                                  .base_options()
                                  .model_asset();
    bool is_model_bundle = false;
    std::string tflite_model_name;
    auto handler_or =
        tasks::core::ExternalFileHandler::CreateFromExternalFile(&model_asset);
    if (handler_or.ok()) {
      absl::flat_hash_map<std::string, absl::string_view> files;
      if (tasks::metadata::ExtractFilesfromZipFile(
              handler_or.value()->GetFileContent().data(),
              handler_or.value()->GetFileContent().size(), &files)
              .ok()) {
        for (const auto& [name, content] : files) {
          if (absl::EndsWith(name, ".tflite")) {
            tflite_model_name = name;
            break;
          }
        }
        if (!tflite_model_name.empty() &&
            files.find(kSentencePieceModelName) != files.end()) {
          is_model_bundle = true;
        }
      }
    }

    if (is_model_bundle) {
      MP_ASSIGN_OR_RETURN(
          const auto* bundle_resources,
          CreateModelAssetBundleResources<proto::TextEmbedderGraphOptions>(sc));
      // Extract tflite model
      MP_ASSIGN_OR_RETURN(auto tflite_model_file,
                          bundle_resources->GetFile(tflite_model_name));
      // Create ModelResources for the tflite model
      auto tflite_external_file =
          std::make_unique<tasks::core::proto::ExternalFile>();
      SetExternalFile(tflite_model_file, tflite_external_file.get(),
                      /*is_copy=*/
                      !sc->Service(kModelResourcesCacheService).IsAvailable());

      std::string model_id = tflite_model_name == kGeckoTFLiteName
                                 ? "gecko_tflite"
                                 : "gemma_tflite";
      MP_ASSIGN_OR_RETURN(
          *model_resources,
          CreateModelResources(sc, std::move(tflite_external_file), model_id));

      // Extract sentencepiece.model
      auto sp_model_file_or =
          bundle_resources->GetFile(kSentencePieceModelName);
      if (sp_model_file_or.ok()) {
        SetExternalFile(
            sp_model_file_or.value(),
            sc->MutableOptions<proto::TextEmbedderGraphOptions>()
                ->mutable_sentence_piece_model(),
            /*is_copy=*/
            !sc->Service(kModelResourcesCacheService).IsAvailable());
      }
    }
    return absl::OkStatus();
  }

  // Adds a mediapipe TextEmbedder task graph into the provided
  // builder::Graph instance. The TextEmbedder task takes an input
  // text (std::string) and returns an embedding result.
  //
  // task_options: the mediapipe tasks TextEmbedderGraphOptions proto.
  // model_resources: the ModelResources object initialized from a
  //   TextEmbedder model file with model metadata.
  // text_in: (std::string) stream to run embedding extraction on.
  // graph: the mediapipe builder::Graph instance to be updated.
  absl::StatusOr<Source<EmbeddingResult>> BuildTextEmbedderTask(
      const proto::TextEmbedderGraphOptions& task_options,
      const ModelResources& model_resources, Source<std::string> text_in,
      Graph& graph) {
    // Adds preprocessing calculators and connects them to the text input
    // stream.
    auto& preprocessing = graph.AddNode(
        "mediapipe.tasks.components.processors.TextPreprocessingGraph");
    auto* preproc_options = &preprocessing.GetOptions<
        components::processors::proto::TextPreprocessingGraphOptions>();
    if (task_options.has_sentence_piece_model()) {
      preproc_options->mutable_sentence_piece_model()->CopyFrom(
          task_options.sentence_piece_model());
    }
    MP_RETURN_IF_ERROR(components::processors::ConfigureTextPreprocessingGraph(
        model_resources, *preproc_options));
    text_in >> preprocessing.In(kTextTag);

    // Adds both InferenceCalculator and ModelResourcesCalculator.
    auto& inference = AddInference(
        model_resources, task_options.base_options().acceleration(), graph);
    // The metadata extractor side-output comes from the
    // ModelResourcesCalculator.
    inference.SideOut(kMetadataExtractorTag) >>
        preprocessing.SideIn(kMetadataExtractorTag);
    preprocessing.Out(kTensorsTag) >> inference.In(kTensorsTag);

    // Adds postprocessing calculators and connects its input stream to the
    // inference results.
    auto& postprocessing = graph.AddNode(
        "mediapipe.tasks.components.processors.EmbeddingPostprocessingGraph");
    auto* postprocessing_options = &postprocessing.GetOptions<
        components::processors::proto::EmbeddingPostprocessingGraphOptions>();

    // The UniversalSentenceEncoder model has an extraneous output head.
    std::vector<absl::string_view> filtered_head_names;
    MP_ASSIGN_OR_RETURN(TextModelType::ModelType model_type,
                        GetModelType(model_resources));
    if (model_type == TextModelType::USE_MODEL) {
      postprocessing_options->mutable_tensors_to_embeddings_options()
          ->add_ignored_head_names(kUSEQueryTensorName);
    }

    MP_RETURN_IF_ERROR(
        components::processors::ConfigureEmbeddingPostprocessingGraph(
            model_resources, task_options.embedder_options(),
            postprocessing_options));
    inference.Out(kTensorsTag) >> postprocessing.In(kTensorsTag);

    // Outputs the embedding result.
    return postprocessing[Output<EmbeddingResult>(kEmbeddingsTag)];
  }
};

REGISTER_MEDIAPIPE_GRAPH(
    ::mediapipe::tasks::text::text_embedder::TextEmbedderGraph);

}  // namespace mediapipe::tasks::text::text_embedder
