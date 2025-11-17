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

#include "mediapipe/tasks/c/vision/image_embedder/image_embedder.h"

#include <cstdint>
#include <cstdlib>
#include <memory>
#include <optional>
#include <utility>

#include "absl/log/absl_log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "mediapipe/framework/formats/image.h"
#include "mediapipe/tasks/c/components/containers/embedding_result.h"
#include "mediapipe/tasks/c/components/containers/embedding_result_converter.h"
#include "mediapipe/tasks/c/components/processors/embedder_options_converter.h"
#include "mediapipe/tasks/c/core/base_options_converter.h"
#include "mediapipe/tasks/c/core/mp_status.h"
#include "mediapipe/tasks/c/core/mp_status_converter.h"
#include "mediapipe/tasks/c/vision/core/common.h"
#include "mediapipe/tasks/c/vision/core/image.h"
#include "mediapipe/tasks/c/vision/core/image_frame_util.h"
#include "mediapipe/tasks/c/vision/core/image_processing_options.h"
#include "mediapipe/tasks/c/vision/core/image_processing_options_converter.h"
#include "mediapipe/tasks/cc/components/containers/embedding_result.h"
#include "mediapipe/tasks/cc/vision/core/image_processing_options.h"
#include "mediapipe/tasks/cc/vision/core/running_mode.h"
#include "mediapipe/tasks/cc/vision/image_embedder/image_embedder.h"

struct MpImageEmbedderInternal {
  std::unique_ptr<::mediapipe::tasks::vision::image_embedder::ImageEmbedder>
      embedder;
};

namespace mediapipe::tasks::c::vision::image_embedder {

namespace {

using ::mediapipe::tasks::c::components::containers::CppCloseEmbeddingResult;
using ::mediapipe::tasks::c::components::containers::CppConvertToCppEmbedding;
using ::mediapipe::tasks::c::components::containers::
    CppConvertToEmbeddingResult;
using ::mediapipe::tasks::c::components::processors::
    CppConvertToEmbedderOptions;
using ::mediapipe::tasks::c::core::CppConvertToBaseOptions;
using ::mediapipe::tasks::c::core::ToMpStatus;
using ::mediapipe::tasks::c::vision::core::CppConvertToImageProcessingOptions;
using ::mediapipe::tasks::vision::core::RunningMode;
using ::mediapipe::tasks::vision::image_embedder::ImageEmbedder;
typedef ::mediapipe::tasks::components::containers::Embedding CppEmbedding;
typedef ::mediapipe::tasks::vision::image_embedder::ImageEmbedderResult
    CppImageEmbedderResult;
using CppImageProcessingOptions =
    ::mediapipe::tasks::vision::core::ImageProcessingOptions;

const Image& ToImage(const MpImagePtr mp_image) { return mp_image->image; }

}  // namespace

MpStatus CppImageEmbedderCreate(const ImageEmbedderOptions& options,
                                MpImageEmbedderPtr* embedder) {
  auto cpp_options = std::make_unique<
      ::mediapipe::tasks::vision::image_embedder::ImageEmbedderOptions>();

  CppConvertToBaseOptions(options.base_options, &cpp_options->base_options);
  CppConvertToEmbedderOptions(options.embedder_options,
                              &cpp_options->embedder_options);
  cpp_options->running_mode = static_cast<RunningMode>(options.running_mode);

  // Enable callback for processing live stream data when the running mode is
  // set to RunningMode::LIVE_STREAM.
  if (cpp_options->running_mode == RunningMode::LIVE_STREAM) {
    if (options.result_callback == nullptr) {
      const absl::Status status = absl::InvalidArgumentError(
          "Provided null pointer to callback function.");
      ABSL_LOG(ERROR) << "Failed to create ImageEmbedder: " << status;
      return ToMpStatus(status);
    }

    ImageEmbedderOptions::result_callback_fn result_callback =
        options.result_callback;
    cpp_options->result_callback =
        [result_callback](absl::StatusOr<CppImageEmbedderResult> cpp_result,
                          const Image& image, int64_t timestamp) {
          MpImageInternal mp_image({.image = image});
          if (!cpp_result.ok()) {
            result_callback(ToMpStatus(cpp_result.status()), nullptr, &mp_image,
                            timestamp);
            return;
          }
          ImageEmbedderResult result;
          CppConvertToEmbeddingResult(*cpp_result, &result);
          result_callback(kMpOk, &result, &mp_image, timestamp);
          CppCloseEmbeddingResult(&result);
        };
  }

  auto cpp_embedder = ImageEmbedder::Create(std::move(cpp_options));
  if (!cpp_embedder.ok()) {
    ABSL_LOG(ERROR) << "Failed to create ImageEmbedder: "
                    << cpp_embedder.status();
    return ToMpStatus(cpp_embedder.status());
  }
  *embedder = new MpImageEmbedderInternal{.embedder = std::move(*cpp_embedder)};
  return kMpOk;
}

MpStatus CppImageEmbedderEmbed(
    MpImageEmbedderPtr embedder, MpImagePtr image,
    const ImageProcessingOptions* image_processing_options,
    ImageEmbedderResult* result) {
  ImageEmbedder* cpp_embedder = embedder->embedder.get();
  std::optional<CppImageProcessingOptions> cpp_image_processing_options;
  if (image_processing_options) {
    CppImageProcessingOptions options;
    CppConvertToImageProcessingOptions(*image_processing_options, &options);
    cpp_image_processing_options = options;
  }
  auto cpp_result =
      cpp_embedder->Embed(ToImage(image), cpp_image_processing_options);
  if (!cpp_result.ok()) {
    ABSL_LOG(ERROR) << "Embedding extraction failed: " << cpp_result.status();
    return ToMpStatus(cpp_result.status());
  }
  CppConvertToEmbeddingResult(*cpp_result, result);
  return kMpOk;
}

MpStatus CppImageEmbedderEmbedForVideo(
    MpImageEmbedderPtr embedder, MpImagePtr image,
    const ImageProcessingOptions* image_processing_options,
    int64_t timestamp_ms, ImageEmbedderResult* result) {
  ImageEmbedder* cpp_embedder = embedder->embedder.get();
  std::optional<CppImageProcessingOptions> cpp_image_processing_options;
  if (image_processing_options) {
    CppImageProcessingOptions options;
    CppConvertToImageProcessingOptions(*image_processing_options, &options);
    cpp_image_processing_options = options;
  }
  auto cpp_result = cpp_embedder->EmbedForVideo(ToImage(image), timestamp_ms,
                                                cpp_image_processing_options);
  if (!cpp_result.ok()) {
    ABSL_LOG(ERROR) << "Embedding extraction failed: " << cpp_result.status();
    return ToMpStatus(cpp_result.status());
  }
  CppConvertToEmbeddingResult(*cpp_result, result);
  return kMpOk;
}

MpStatus CppImageEmbedderEmbedAsync(
    MpImageEmbedderPtr embedder, MpImagePtr image,
    const ImageProcessingOptions* image_processing_options,
    int64_t timestamp_ms) {
  ImageEmbedder* cpp_embedder = embedder->embedder.get();
  std::optional<CppImageProcessingOptions> cpp_image_processing_options;
  if (image_processing_options) {
    CppImageProcessingOptions options;
    CppConvertToImageProcessingOptions(*image_processing_options, &options);
    cpp_image_processing_options = options;
  }
  auto cpp_result = cpp_embedder->EmbedAsync(ToImage(image), timestamp_ms,
                                             cpp_image_processing_options);
  if (!cpp_result.ok()) {
    ABSL_LOG(ERROR) << "Data preparation for the embedding extraction failed: "
                    << cpp_result;
    return ToMpStatus(cpp_result);
  }
  return kMpOk;
}

void CppImageEmbedderCloseResult(ImageEmbedderResult* result) {
  CppCloseEmbeddingResult(result);
}

MpStatus CppImageEmbedderClose(MpImageEmbedderPtr embedder) {
  auto cpp_embedder = embedder->embedder.get();
  auto result = cpp_embedder->Close();
  if (!result.ok()) {
    ABSL_LOG(ERROR) << "Failed to close ImageEmbedder: " << result;
    return ToMpStatus(result);
  }
  delete embedder;
  return kMpOk;
}

MpStatus CppImageEmbedderCosineSimilarity(const Embedding& u,
                                          const Embedding& v,
                                          double* similarity) {
  CppEmbedding cpp_u;
  CppConvertToCppEmbedding(u, &cpp_u);
  CppEmbedding cpp_v;
  CppConvertToCppEmbedding(v, &cpp_v);
  auto status_or_similarity =
      mediapipe::tasks::vision::image_embedder::ImageEmbedder::CosineSimilarity(
          cpp_u, cpp_v);
  if (!status_or_similarity.ok()) {
    ABSL_LOG(ERROR) << "Cannot compute cosine similarity: "
                    << status_or_similarity.status();
    return ToMpStatus(status_or_similarity.status());
  }
  *similarity = status_or_similarity.value();
  return kMpOk;
}

}  // namespace mediapipe::tasks::c::vision::image_embedder

extern "C" {

MP_EXPORT MpStatus MpImageEmbedderCreate(struct ImageEmbedderOptions* options,
                                         MpImageEmbedderPtr* embedder_out) {
  return mediapipe::tasks::c::vision::image_embedder::CppImageEmbedderCreate(
      *options, embedder_out);
}

MP_EXPORT MpStatus MpImageEmbedderEmbedImage(
    MpImageEmbedderPtr embedder, MpImagePtr image,
    const ImageProcessingOptions* image_processing_options,
    ImageEmbedderResult* result) {
  return mediapipe::tasks::c::vision::image_embedder::CppImageEmbedderEmbed(
      embedder, image, image_processing_options, result);
}

MP_EXPORT MpStatus MpImageEmbedderEmbedForVideo(
    MpImageEmbedderPtr embedder, MpImagePtr image,
    const ImageProcessingOptions* image_processing_options,
    int64_t timestamp_ms, ImageEmbedderResult* result) {
  return mediapipe::tasks::c::vision::image_embedder::
      CppImageEmbedderEmbedForVideo(embedder, image, image_processing_options,
                                    timestamp_ms, result);
}

MP_EXPORT MpStatus MpImageEmbedderEmbedAsync(
    MpImageEmbedderPtr embedder, MpImagePtr image,
    const ImageProcessingOptions* image_processing_options,
    int64_t timestamp_ms) {
  return mediapipe::tasks::c::vision::image_embedder::
      CppImageEmbedderEmbedAsync(embedder, image, image_processing_options,
                                 timestamp_ms);
}

MP_EXPORT void MpImageEmbedderCloseResult(ImageEmbedderResult* result) {
  mediapipe::tasks::c::vision::image_embedder::CppImageEmbedderCloseResult(
      result);
}

MP_EXPORT MpStatus MpImageEmbedderClose(MpImageEmbedderPtr embedder) {
  return mediapipe::tasks::c::vision::image_embedder::CppImageEmbedderClose(
      embedder);
}

MP_EXPORT MpStatus MpImageEmbedderCosineSimilarity(const Embedding& u,
                                                   const Embedding& v,
                                                   double* similarity_out) {
  return mediapipe::tasks::c::vision::image_embedder::
      CppImageEmbedderCosineSimilarity(u, v, similarity_out);
}

}  // extern "C"
