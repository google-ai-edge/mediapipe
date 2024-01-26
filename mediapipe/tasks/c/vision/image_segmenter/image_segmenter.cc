/* Copyright 2024 The MediaPipe Authors.

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

#include "mediapipe/tasks/c/vision/image_segmenter/image_segmenter.h"

#include <cstdint>
#include <cstdlib>
#include <memory>
#include <utility>

#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "mediapipe/framework/formats/image.h"
#include "mediapipe/framework/formats/image_frame.h"
#include "mediapipe/tasks/c/core/base_options_converter.h"
#include "mediapipe/tasks/c/vision/core/common.h"
#include "mediapipe/tasks/c/vision/image_segmenter/image_segmenter_result.h"
#include "mediapipe/tasks/c/vision/image_segmenter/image_segmenter_result_converter.h"
#include "mediapipe/tasks/cc/vision/core/running_mode.h"
#include "mediapipe/tasks/cc/vision/image_segmenter/image_segmenter.h"
#include "mediapipe/tasks/cc/vision/image_segmenter/image_segmenter_result.h"
#include "mediapipe/tasks/cc/vision/utils/image_utils.h"

namespace mediapipe::tasks::c::vision::image_segmenter {

namespace {

using ::mediapipe::tasks::c::components::containers::
    CppCloseImageSegmenterResult;
using ::mediapipe::tasks::c::components::containers::
    CppConvertToImageSegmenterResult;
using ::mediapipe::tasks::c::core::CppConvertToBaseOptions;
using ::mediapipe::tasks::vision::CreateImageFromBuffer;
using ::mediapipe::tasks::vision::core::RunningMode;
using ::mediapipe::tasks::vision::image_segmenter::ImageSegmenter;
typedef ::mediapipe::tasks::vision::image_segmenter::ImageSegmenterResult
    CppImageSegmenterResult;

int CppProcessError(absl::Status status, char** error_msg) {
  if (error_msg) {
    *error_msg = strdup(status.ToString().c_str());
  }
  return status.raw_code();
}

}  // namespace

void CppConvertToImageSegmenterOptions(
    const ImageSegmenterOptions& in,
    mediapipe::tasks::vision::image_segmenter::ImageSegmenterOptions* out) {
  out->display_names_locale = in.display_names_locale;
  out->output_confidence_masks = in.output_confidence_masks;
  out->output_category_mask = in.output_category_mask;
}

ImageSegmenter* CppImageSegmenterCreate(const ImageSegmenterOptions& options,
                                        char** error_msg) {
  auto cpp_options = std::make_unique<
      ::mediapipe::tasks::vision::image_segmenter::ImageSegmenterOptions>();

  CppConvertToBaseOptions(options.base_options, &cpp_options->base_options);
  CppConvertToImageSegmenterOptions(options, cpp_options.get());
  cpp_options->running_mode = static_cast<RunningMode>(options.running_mode);

  // Enable callback for processing live stream data when the running mode is
  // set to RunningMode::LIVE_STREAM.
  if (cpp_options->running_mode == RunningMode::LIVE_STREAM) {
    if (options.result_callback == nullptr) {
      const absl::Status status = absl::InvalidArgumentError(
          "Provided null pointer to callback function.");
      LOG(ERROR) << "Failed to create ImageSegmenter: " << status;
      CppProcessError(status, error_msg);
      return nullptr;
    }

    ImageSegmenterOptions::result_callback_fn result_callback =
        options.result_callback;
    cpp_options->result_callback =
        [result_callback](absl::StatusOr<CppImageSegmenterResult> cpp_result,
                          const Image& image, int64_t timestamp) {
          char* error_msg = nullptr;

          if (!cpp_result.ok()) {
            LOG(ERROR) << "Segmentation failed: " << cpp_result.status();
            CppProcessError(cpp_result.status(), &error_msg);
            result_callback(nullptr, MpImage(), timestamp, error_msg);
            free(error_msg);
            return;
          }

          // Result is valid for the lifetime of the callback function.
          ImageSegmenterResult result;
          CppConvertToImageSegmenterResult(*cpp_result, &result);

          const auto& image_frame = image.GetImageFrameSharedPtr();
          const MpImage mp_image = {
              .type = MpImage::IMAGE_FRAME,
              .image_frame = {
                  .format = static_cast<::ImageFormat>(image_frame->Format()),
                  .image_buffer = image_frame->PixelData(),
                  .width = image_frame->Width(),
                  .height = image_frame->Height()}};

          result_callback(&result, mp_image, timestamp,
                          /* error_msg= */ nullptr);

          CppCloseImageSegmenterResult(&result);
        };
  }

  auto segmenter = ImageSegmenter::Create(std::move(cpp_options));
  if (!segmenter.ok()) {
    LOG(ERROR) << "Failed to create ImageSegmenter: " << segmenter.status();
    CppProcessError(segmenter.status(), error_msg);
    return nullptr;
  }
  return segmenter->release();
}

int CppImageSegmenterSegment(void* segmenter, const MpImage& image,
                             ImageSegmenterResult* result, char** error_msg) {
  if (image.type == MpImage::GPU_BUFFER) {
    const absl::Status status =
        absl::InvalidArgumentError("GPU Buffer not supported yet.");

    LOG(ERROR) << "Segmentation failed: " << status.message();
    return CppProcessError(status, error_msg);
  }

  const auto img = CreateImageFromBuffer(
      static_cast<ImageFormat::Format>(image.image_frame.format),
      image.image_frame.image_buffer, image.image_frame.width,
      image.image_frame.height);

  if (!img.ok()) {
    LOG(ERROR) << "Failed to create Image: " << img.status();
    return CppProcessError(img.status(), error_msg);
  }

  auto cpp_segmenter = static_cast<ImageSegmenter*>(segmenter);
  auto cpp_result = cpp_segmenter->Segment(*img);
  if (!cpp_result.ok()) {
    LOG(ERROR) << "Segmentation failed: " << cpp_result.status();
    return CppProcessError(cpp_result.status(), error_msg);
  }
  CppConvertToImageSegmenterResult(*cpp_result, result);
  return 0;
}

int CppImageSegmenterSegmentForVideo(void* segmenter, const MpImage& image,
                                     int64_t timestamp_ms,
                                     ImageSegmenterResult* result,
                                     char** error_msg) {
  if (image.type == MpImage::GPU_BUFFER) {
    absl::Status status =
        absl::InvalidArgumentError("GPU Buffer not supported yet");

    LOG(ERROR) << "Segmentation failed: " << status.message();
    return CppProcessError(status, error_msg);
  }

  const auto img = CreateImageFromBuffer(
      static_cast<ImageFormat::Format>(image.image_frame.format),
      image.image_frame.image_buffer, image.image_frame.width,
      image.image_frame.height);

  if (!img.ok()) {
    LOG(ERROR) << "Failed to create Image: " << img.status();
    return CppProcessError(img.status(), error_msg);
  }

  auto cpp_segmenter = static_cast<ImageSegmenter*>(segmenter);
  auto cpp_result = cpp_segmenter->SegmentForVideo(*img, timestamp_ms);
  if (!cpp_result.ok()) {
    LOG(ERROR) << "Segmentation failed: " << cpp_result.status();
    return CppProcessError(cpp_result.status(), error_msg);
  }
  CppConvertToImageSegmenterResult(*cpp_result, result);
  return 0;
}

int CppImageSegmenterSegmentAsync(void* segmenter, const MpImage& image,
                                  int64_t timestamp_ms, char** error_msg) {
  if (image.type == MpImage::GPU_BUFFER) {
    absl::Status status =
        absl::InvalidArgumentError("GPU Buffer not supported yet");

    LOG(ERROR) << "Segmentation failed: " << status.message();
    return CppProcessError(status, error_msg);
  }

  const auto img = CreateImageFromBuffer(
      static_cast<ImageFormat::Format>(image.image_frame.format),
      image.image_frame.image_buffer, image.image_frame.width,
      image.image_frame.height);

  if (!img.ok()) {
    LOG(ERROR) << "Failed to create Image: " << img.status();
    return CppProcessError(img.status(), error_msg);
  }

  auto cpp_segmenter = static_cast<ImageSegmenter*>(segmenter);
  auto cpp_result = cpp_segmenter->SegmentAsync(*img, timestamp_ms);
  if (!cpp_result.ok()) {
    LOG(ERROR) << "Data preparation for the image segmentation failed: "
               << cpp_result;
    return CppProcessError(cpp_result, error_msg);
  }
  return 0;
}

void CppImageSegmenterCloseResult(ImageSegmenterResult* result) {
  CppCloseImageSegmenterResult(result);
}

int CppImageSegmenterClose(void* segmenter, char** error_msg) {
  auto cpp_segmenter = static_cast<ImageSegmenter*>(segmenter);
  auto result = cpp_segmenter->Close();
  if (!result.ok()) {
    LOG(ERROR) << "Failed to close ImageSegmenter: " << result;
    return CppProcessError(result, error_msg);
  }
  delete cpp_segmenter;
  return 0;
}

}  // namespace mediapipe::tasks::c::vision::image_segmenter

extern "C" {

void* image_segmenter_create(struct ImageSegmenterOptions* options,
                             char** error_msg) {
  return mediapipe::tasks::c::vision::image_segmenter::CppImageSegmenterCreate(
      *options, error_msg);
}

int image_segmenter_segment_image(void* segmenter, const MpImage& image,
                                  ImageSegmenterResult* result,
                                  char** error_msg) {
  return mediapipe::tasks::c::vision::image_segmenter::CppImageSegmenterSegment(
      segmenter, image, result, error_msg);
}

int image_segmenter_segment_for_video(void* segmenter, const MpImage& image,
                                      int64_t timestamp_ms,
                                      ImageSegmenterResult* result,
                                      char** error_msg) {
  return mediapipe::tasks::c::vision::image_segmenter::
      CppImageSegmenterSegmentForVideo(segmenter, image, timestamp_ms, result,
                                       error_msg);
}

int image_segmenter_segment_async(void* segmenter, const MpImage& image,
                                  int64_t timestamp_ms, char** error_msg) {
  return mediapipe::tasks::c::vision::image_segmenter::
      CppImageSegmenterSegmentAsync(segmenter, image, timestamp_ms, error_msg);
}

void image_segmenter_close_result(ImageSegmenterResult* result) {
  mediapipe::tasks::c::vision::image_segmenter::CppImageSegmenterCloseResult(
      result);
}

int image_segmenter_close(void* segmenter, char** error_ms) {
  return mediapipe::tasks::c::vision::image_segmenter::CppImageSegmenterClose(
      segmenter, error_ms);
}

}  // extern "C"
