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
#include <utility>

#include "absl/log/absl_log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "mediapipe/framework/formats/image.h"
#include "mediapipe/framework/formats/image_frame.h"
#include "mediapipe/tasks/c/components/containers/embedding_result.h"
#include "mediapipe/tasks/c/components/containers/embedding_result_converter.h"
#include "mediapipe/tasks/c/components/processors/embedder_options_converter.h"
#include "mediapipe/tasks/c/core/base_options_converter.h"
#include "mediapipe/tasks/c/vision/core/common.h"
#include "mediapipe/tasks/cc/components/containers/embedding_result.h"
#include "mediapipe/tasks/cc/vision/core/running_mode.h"
#include "mediapipe/tasks/cc/vision/image_embedder/image_embedder.h"
#include "mediapipe/tasks/cc/vision/utils/image_utils.h"

namespace mediapipe::tasks::c::vision::image_embedder {

namespace {

using ::mediapipe::tasks::c::components::containers::CppCloseEmbeddingResult;
using ::mediapipe::tasks::c::components::containers::CppConvertToCppEmbedding;
using ::mediapipe::tasks::c::components::containers::
    CppConvertToEmbeddingResult;
using ::mediapipe::tasks::c::components::processors::
    CppConvertToEmbedderOptions;
using ::mediapipe::tasks::c::core::CppConvertToBaseOptions;
using ::mediapipe::tasks::vision::CreateImageFromBuffer;
using ::mediapipe::tasks::vision::core::RunningMode;
using ::mediapipe::tasks::vision::image_embedder::ImageEmbedder;
typedef ::mediapipe::tasks::components::containers::Embedding CppEmbedding;
typedef ::mediapipe::tasks::vision::image_embedder::ImageEmbedderResult
    CppImageEmbedderResult;

int CppProcessError(absl::Status status, char** error_msg) {
  if (error_msg) {
    *error_msg = strdup(status.ToString().c_str());
  }
  return status.raw_code();
}

}  // namespace

ImageEmbedder* CppImageEmbedderCreate(const ImageEmbedderOptions& options,
                                      char** error_msg) {
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
      CppProcessError(status, error_msg);
      return nullptr;
    }

    ImageEmbedderOptions::result_callback_fn result_callback =
        options.result_callback;
    cpp_options->result_callback =
        [result_callback](absl::StatusOr<CppImageEmbedderResult> cpp_result,
                          const Image& image, int64_t timestamp) {
          char* error_msg = nullptr;

          if (!cpp_result.ok()) {
            ABSL_LOG(ERROR)
                << "Embedding extraction failed: " << cpp_result.status();
            CppProcessError(cpp_result.status(), &error_msg);
            result_callback(nullptr, nullptr, timestamp, error_msg);
            free(error_msg);
            return;
          }

          // Result is valid for the lifetime of the callback function.
          auto result = std::make_unique<ImageEmbedderResult>();
          CppConvertToEmbeddingResult(*cpp_result, result.get());

          const auto& image_frame = image.GetImageFrameSharedPtr();
          const MpImage mp_image = {
              .type = MpImage::IMAGE_FRAME,
              .image_frame = {
                  .format = static_cast<::ImageFormat>(image_frame->Format()),
                  .image_buffer = image_frame->PixelData(),
                  .width = image_frame->Width(),
                  .height = image_frame->Height()}};

          result_callback(result.release(), &mp_image, timestamp,
                          /* error_msg= */ nullptr);
        };
  }

  auto embedder = ImageEmbedder::Create(std::move(cpp_options));
  if (!embedder.ok()) {
    ABSL_LOG(ERROR) << "Failed to create ImageEmbedder: " << embedder.status();
    CppProcessError(embedder.status(), error_msg);
    return nullptr;
  }
  return embedder->release();
}

int CppImageEmbedderEmbed(void* embedder, const MpImage* image,
                          ImageEmbedderResult* result, char** error_msg) {
  if (image->type == MpImage::GPU_BUFFER) {
    const absl::Status status =
        absl::InvalidArgumentError("GPU Buffer not supported yet.");

    ABSL_LOG(ERROR) << "Embedding extraction failed: " << status.message();
    return CppProcessError(status, error_msg);
  }

  const auto img = CreateImageFromBuffer(
      static_cast<ImageFormat::Format>(image->image_frame.format),
      image->image_frame.image_buffer, image->image_frame.width,
      image->image_frame.height);

  if (!img.ok()) {
    ABSL_LOG(ERROR) << "Failed to create Image: " << img.status();
    return CppProcessError(img.status(), error_msg);
  }

  auto cpp_embedder = static_cast<ImageEmbedder*>(embedder);
  auto cpp_result = cpp_embedder->Embed(*img);
  if (!cpp_result.ok()) {
    ABSL_LOG(ERROR) << "Embedding extraction failed: " << cpp_result.status();
    return CppProcessError(cpp_result.status(), error_msg);
  }
  CppConvertToEmbeddingResult(*cpp_result, result);
  return 0;
}

int CppImageEmbedderEmbedForVideo(void* embedder, const MpImage* image,
                                  int64_t timestamp_ms,
                                  ImageEmbedderResult* result,
                                  char** error_msg) {
  if (image->type == MpImage::GPU_BUFFER) {
    absl::Status status =
        absl::InvalidArgumentError("GPU Buffer not supported yet");

    ABSL_LOG(ERROR) << "Embedding extraction failed: " << status.message();
    return CppProcessError(status, error_msg);
  }

  const auto img = CreateImageFromBuffer(
      static_cast<ImageFormat::Format>(image->image_frame.format),
      image->image_frame.image_buffer, image->image_frame.width,
      image->image_frame.height);

  if (!img.ok()) {
    ABSL_LOG(ERROR) << "Failed to create Image: " << img.status();
    return CppProcessError(img.status(), error_msg);
  }

  auto cpp_embedder = static_cast<ImageEmbedder*>(embedder);
  auto cpp_result = cpp_embedder->EmbedForVideo(*img, timestamp_ms);
  if (!cpp_result.ok()) {
    ABSL_LOG(ERROR) << "Embedding extraction failed: " << cpp_result.status();
    return CppProcessError(cpp_result.status(), error_msg);
  }
  CppConvertToEmbeddingResult(*cpp_result, result);
  return 0;
}

int CppImageEmbedderEmbedAsync(void* embedder, const MpImage* image,
                               int64_t timestamp_ms, char** error_msg) {
  if (image->type == MpImage::GPU_BUFFER) {
    absl::Status status =
        absl::InvalidArgumentError("GPU Buffer not supported yet");

    ABSL_LOG(ERROR) << "Embedding extraction failed: " << status.message();
    return CppProcessError(status, error_msg);
  }

  const auto img = CreateImageFromBuffer(
      static_cast<ImageFormat::Format>(image->image_frame.format),
      image->image_frame.image_buffer, image->image_frame.width,
      image->image_frame.height);

  if (!img.ok()) {
    ABSL_LOG(ERROR) << "Failed to create Image: " << img.status();
    return CppProcessError(img.status(), error_msg);
  }

  auto cpp_embedder = static_cast<ImageEmbedder*>(embedder);
  auto cpp_result = cpp_embedder->EmbedAsync(*img, timestamp_ms);
  if (!cpp_result.ok()) {
    ABSL_LOG(ERROR) << "Data preparation for the embedding extraction failed: "
                    << cpp_result;
    return CppProcessError(cpp_result, error_msg);
  }
  return 0;
}

void CppImageEmbedderCloseResult(ImageEmbedderResult* result) {
  CppCloseEmbeddingResult(result);
}

int CppImageEmbedderClose(void* embedder, char** error_msg) {
  auto cpp_embedder = static_cast<ImageEmbedder*>(embedder);
  auto result = cpp_embedder->Close();
  if (!result.ok()) {
    ABSL_LOG(ERROR) << "Failed to close ImageEmbedder: " << result;
    return CppProcessError(result, error_msg);
  }
  delete cpp_embedder;
  return 0;
}

int CppImageEmbedderCosineSimilarity(const Embedding& u, const Embedding& v,
                                     double* similarity, char** error_msg) {
  CppEmbedding cpp_u;
  CppConvertToCppEmbedding(u, &cpp_u);
  CppEmbedding cpp_v;
  CppConvertToCppEmbedding(v, &cpp_v);
  auto status_or_similarity =
      mediapipe::tasks::vision::image_embedder::ImageEmbedder::CosineSimilarity(
          cpp_u, cpp_v);
  if (status_or_similarity.ok()) {
    *similarity = status_or_similarity.value();
  } else {
    ABSL_LOG(ERROR) << "Cannot compute cosine similarity.";
    return CppProcessError(status_or_similarity.status(), error_msg);
  }
  return 0;
}

}  // namespace mediapipe::tasks::c::vision::image_embedder

extern "C" {

void* image_embedder_create(struct ImageEmbedderOptions* options,
                            char** error_msg) {
  return mediapipe::tasks::c::vision::image_embedder::CppImageEmbedderCreate(
      *options, error_msg);
}

int image_embedder_embed_image(void* embedder, const MpImage* image,
                               ImageEmbedderResult* result, char** error_msg) {
  return mediapipe::tasks::c::vision::image_embedder::CppImageEmbedderEmbed(
      embedder, image, result, error_msg);
}

int image_embedder_embed_for_video(void* embedder, const MpImage* image,
                                   int64_t timestamp_ms,
                                   ImageEmbedderResult* result,
                                   char** error_msg) {
  return mediapipe::tasks::c::vision::image_embedder::
      CppImageEmbedderEmbedForVideo(embedder, image, timestamp_ms, result,
                                    error_msg);
}

int image_embedder_embed_async(void* embedder, const MpImage* image,
                               int64_t timestamp_ms, char** error_msg) {
  return mediapipe::tasks::c::vision::image_embedder::
      CppImageEmbedderEmbedAsync(embedder, image, timestamp_ms, error_msg);
}

void image_embedder_close_result(ImageEmbedderResult* result) {
  mediapipe::tasks::c::vision::image_embedder::CppImageEmbedderCloseResult(
      result);
}

int image_embedder_close(void* embedder, char** error_msg) {
  return mediapipe::tasks::c::vision::image_embedder::CppImageEmbedderClose(
      embedder, error_msg);
}

int image_embedder_cosine_similarity(const Embedding& u, const Embedding& v,
                                     double* similarity, char** error_msg) {
  return mediapipe::tasks::c::vision::image_embedder::
      CppImageEmbedderCosineSimilarity(u, v, similarity, error_msg);
}

}  // extern "C"
