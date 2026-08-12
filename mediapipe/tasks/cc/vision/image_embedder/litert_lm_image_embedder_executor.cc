/* Copyright 2026 The MediaPipe Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
you may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "mediapipe/tasks/cc/vision/image_embedder/litert_lm_image_embedder_executor.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <functional>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "mediapipe/framework/formats/image_frame.h"
#include "mediapipe/framework/formats/rect.pb.h"
#include "mediapipe/framework/port/status_macros.h"
#include "mediapipe/tasks/cc/components/containers/embedding_result.h"
#include "mediapipe/tasks/cc/vision/image_embedder/image_embedder_executor.h"
#include "odml/litert_lm/runtime/core/embedding_engine_impl.h"  // from @odml
#include "odml/litert_lm/runtime/engine/embedding_engine.h"     // from @odml
#include "odml/litert_lm/runtime/engine/embedding_engine_settings.h"  // from @odml
#include "odml/litert_lm/runtime/engine/io_types.h"  // from @odml
#include "odml/litert_lm/runtime/executor/executor_settings_base.h"  // from @odml
#include "odml/litert_lm/runtime/util/memory_mapped_file.h"  // from @odml
#include "odml/litert_lm/support/util/io_types.h"            // from @odml

namespace mediapipe::tasks::vision::image_embedder {
namespace {

using ::litert::lm::EmbeddingEngineImpl;
using ::litert::lm::EmbeddingEngineSettings;
using ::litert::lm::EmbeddingOptions;
using ::litert::lm::InputData;
using ::litert::lm::MemoryMappedFile;
using ::litert::lm::ModelAssets;
using ::mediapipe::tasks::components::containers::EmbeddingResult;

}  // namespace

LiteRtLmImageEmbedderExecutor::LiteRtLmImageEmbedderExecutor(
    std::shared_ptr<MemoryMappedFile> shared_mmap,
    std::unique_ptr<::litert::lm::EmbeddingEngine> engine, bool l2_normalize,
    bool quantize,
    std::function<void(absl::StatusOr<EmbeddingResult>, const Image&, int64_t)>
        result_callback)
    : shared_mmap_(std::move(shared_mmap)),
      engine_(std::move(engine)),
      l2_normalize_(l2_normalize),
      quantize_(quantize),
      result_callback_(std::move(result_callback)) {}

LiteRtLmImageEmbedderExecutor::~LiteRtLmImageEmbedderExecutor() = default;

absl::StatusOr<::litert::support::InputImage>
LiteRtLmImageEmbedderExecutor::PreprocessImage(const Image& image) {
  auto image_frame = image.GetImageFrameSharedPtr();
  if (!image_frame) {
    return absl::InternalError("Failed to obtain ImageFrame from image.");
  }

  // Ensure image is in SRGB/RGB format (3 channels, 1 byte per pixel)
  if (image_frame->Format() != mediapipe::ImageFormat::SRGB) {
    return absl::InvalidArgumentError("Image format must be SRGB.");
  }

  const int width = image_frame->Width();
  const int height = image_frame->Height();

  // Create PPM format in memory to leverage the stb image preprocessor's
  // decoding, rescaling, normalizing, and patchifying pipeline.
  std::string ppm_data;
  ppm_data.reserve(100 + height * width * 3);
  absl::StrAppendFormat(&ppm_data, "P6\n%d %d\n255\n", width, height);

  const uint8_t* src_ptr = image_frame->PixelData();
  const int row_bytes = width * 3;

  for (int y = 0; y < height; ++y) {
    ppm_data.append(
        reinterpret_cast<const char*>(src_ptr + y * image_frame->WidthStep()),
        row_bytes);
  }

  return ::litert::support::InputImage(std::move(ppm_data));
}

absl::StatusOr<ImageEmbedderResult> LiteRtLmImageEmbedderExecutor::Embed(
    Image image, const NormalizedRect& norm_rect) {
  ABSL_ASSIGN_OR_RETURN(auto preprocessed_image, PreprocessImage(image));

  std::vector<InputData> contents;
  contents.push_back(std::move(preprocessed_image));

  EmbeddingOptions options;
  options.normalize = l2_normalize_;
  options.insert_special_tokens = true;

  ABSL_ASSIGN_OR_RETURN(auto response,
                        engine_->ComputeEmbedding(contents, options),
                        _ << "Failed to compute embeddings.");

  ImageEmbedderResult result;
  components::containers::Embedding embedding;

  if (quantize_) {
    std::string quantized;
    quantized.resize(response.embedding.size());
    for (size_t i = 0; i < response.embedding.size(); ++i) {
      float val = response.embedding[i];
      int unclamped_value = static_cast<int>(roundf(val * 128));
      quantized[i] =
          static_cast<char>(std::max(-128, std::min(unclamped_value, 127)));
    }
    embedding.quantized_embedding = std::move(quantized);
  } else {
    embedding.float_embedding = std::move(response.embedding);
  }

  result.embeddings.push_back(std::move(embedding));
  return result;
}

absl::StatusOr<ImageEmbedderResult>
LiteRtLmImageEmbedderExecutor::EmbedForVideo(Image image,
                                             const NormalizedRect& norm_rect,
                                             int64_t timestamp_ms) {
  return Embed(std::move(image), norm_rect);
}

absl::Status LiteRtLmImageEmbedderExecutor::EmbedAsync(
    Image image, const NormalizedRect& norm_rect, int64_t timestamp_ms) {
  if (!result_callback_) {
    return absl::FailedPreconditionError("Result callback is not registered.");
  }
  Image callback_image = image;
  auto result_or = Embed(std::move(image), norm_rect);
  result_callback_(result_or, callback_image, timestamp_ms);
  return absl::OkStatus();
}

absl::Status LiteRtLmImageEmbedderExecutor::Close() { return absl::OkStatus(); }

absl::StatusOr<std::unique_ptr<LiteRtLmImageEmbedderExecutor>>
LiteRtLmImageEmbedderExecutor::Create(
    const BaseOptions& base_options, const EmbedderOptions& embedder_options,
    std::function<void(absl::StatusOr<EmbeddingResult>, const Image&, int64_t)>
        result_callback) {
  if (base_options.model_asset_path.empty()) {
    return absl::FailedPreconditionError("No model asset path specified.");
  }

  ::litert::lm::Backend backend = ::litert::lm::Backend::CPU;
  if (base_options.delegate == BaseOptions::GPU) {
    backend = ::litert::lm::Backend::GPU;
  }

  // Load directly as a raw `.litertlm` model file.
  ABSL_ASSIGN_OR_RETURN(
      auto mmap_file, MemoryMappedFile::Create(base_options.model_asset_path));
  auto shared_mmap = std::shared_ptr<MemoryMappedFile>(std::move(mmap_file));

  ABSL_ASSIGN_OR_RETURN(
      auto model_assets,
      ModelAssets::Create(shared_mmap, base_options.model_asset_path));
  ABSL_ASSIGN_OR_RETURN(auto settings,
                        EmbeddingEngineSettings::CreateDefault(
                            std::move(model_assets), backend, backend));

  ABSL_ASSIGN_OR_RETURN(auto engine,
                        EmbeddingEngineImpl::Create(std::move(settings)));

  return std::make_unique<LiteRtLmImageEmbedderExecutor>(
      std::move(shared_mmap), std::move(engine), embedder_options.l2_normalize,
      embedder_options.quantize, std::move(result_callback));
}

}  // namespace mediapipe::tasks::vision::image_embedder
