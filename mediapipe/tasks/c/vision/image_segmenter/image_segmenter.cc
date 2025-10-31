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
#include "mediapipe/tasks/c/core/base_options_converter.h"
#include "mediapipe/tasks/c/vision/core/image.h"
#include "mediapipe/tasks/c/vision/core/image_frame_util.h"
#include "mediapipe/tasks/c/vision/core/image_processing_options.h"
#include "mediapipe/tasks/c/vision/core/image_processing_options_converter.h"
#include "mediapipe/tasks/c/vision/image_segmenter/image_segmenter_result.h"
#include "mediapipe/tasks/c/vision/image_segmenter/image_segmenter_result_converter.h"
#include "mediapipe/tasks/cc/vision/core/image_processing_options.h"
#include "mediapipe/tasks/cc/vision/core/running_mode.h"
#include "mediapipe/tasks/cc/vision/image_segmenter/image_segmenter.h"
#include "mediapipe/tasks/cc/vision/image_segmenter/image_segmenter_result.h"

struct MpImageSegmenterInternal {
  std::unique_ptr<::mediapipe::tasks::vision::image_segmenter::ImageSegmenter>
      segmenter;
};

namespace mediapipe::tasks::c::vision::image_segmenter {

namespace {

using ::mediapipe::tasks::c::components::containers::
    CppCloseImageSegmenterResult;
using ::mediapipe::tasks::c::components::containers::
    CppConvertToImageSegmenterResult;
using ::mediapipe::tasks::c::core::CppConvertToBaseOptions;
using ::mediapipe::tasks::c::vision::core::CppConvertToImageProcessingOptions;
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

const Image& ToImage(const MpImagePtr mp_image) { return mp_image->image; }

}  // namespace

void CppConvertToImageSegmenterOptions(
    const ImageSegmenterOptions& in,
    mediapipe::tasks::vision::image_segmenter::ImageSegmenterOptions* out) {
  out->display_names_locale = in.display_names_locale;
  out->output_confidence_masks = in.output_confidence_masks;
  out->output_category_mask = in.output_category_mask;
}

MpImageSegmenterPtr CppImageSegmenterCreate(
    const ImageSegmenterOptions& options, char** error_msg) {
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
            result_callback(nullptr, nullptr, timestamp, error_msg);
            free(error_msg);
            return;
          }

          // Result is valid for the lifetime of the callback function.
          auto result = std::make_unique<ImageSegmenterResult>();
          CppConvertToImageSegmenterResult(*cpp_result, result.get());

          MpImageInternal mp_image = {.image = image};

          result_callback(result.release(), &mp_image, timestamp,
                          /* error_msg= */ nullptr);
        };
  }

  auto segmenter = ImageSegmenter::Create(std::move(cpp_options));
  if (!segmenter.ok()) {
    LOG(ERROR) << "Failed to create ImageSegmenter: " << segmenter.status();
    CppProcessError(segmenter.status(), error_msg);
    return nullptr;
  }
  return new MpImageSegmenterInternal{.segmenter = std::move(*segmenter)};
}

int CppImageSegmenterSegment(MpImageSegmenterPtr segmenter, MpImagePtr image,
                             const ImageProcessingOptions* options,
                             ImageSegmenterResult* result, char** error_msg) {
  auto cpp_segmenter = segmenter->segmenter.get();
  absl::StatusOr<CppImageSegmenterResult> cpp_result;
  if (options) {
    ::mediapipe::tasks::vision::core::ImageProcessingOptions cpp_options;
    CppConvertToImageProcessingOptions(*options, &cpp_options);
    cpp_result = cpp_segmenter->Segment(ToImage(image), cpp_options);
  } else {
    cpp_result = cpp_segmenter->Segment(ToImage(image));
  }

  if (!cpp_result.ok()) {
    LOG(ERROR) << "Segmentation failed: " << cpp_result.status();
    return CppProcessError(cpp_result.status(), error_msg);
  }
  CppConvertToImageSegmenterResult(*cpp_result, result);
  return 0;
}

int CppImageSegmenterSegmentForVideo(MpImageSegmenterPtr segmenter,
                                     MpImagePtr image,
                                     const ImageProcessingOptions* options,
                                     int64_t timestamp_ms,
                                     ImageSegmenterResult* result,
                                     char** error_msg) {
  auto cpp_segmenter = segmenter->segmenter.get();
  absl::StatusOr<CppImageSegmenterResult> cpp_result;
  if (options) {
    ::mediapipe::tasks::vision::core::ImageProcessingOptions cpp_options;
    CppConvertToImageProcessingOptions(*options, &cpp_options);
    cpp_result = cpp_segmenter->SegmentForVideo(ToImage(image), timestamp_ms,
                                                cpp_options);
  } else {
    cpp_result = cpp_segmenter->SegmentForVideo(ToImage(image), timestamp_ms);
  }

  if (!cpp_result.ok()) {
    LOG(ERROR) << "Segmentation failed: " << cpp_result.status();
    return CppProcessError(cpp_result.status(), error_msg);
  }
  CppConvertToImageSegmenterResult(*cpp_result, result);
  return 0;
}

int CppImageSegmenterSegmentAsync(MpImageSegmenterPtr segmenter,
                                  MpImagePtr image,
                                  const ImageProcessingOptions* options,
                                  int64_t timestamp_ms, char** error_msg) {
  auto cpp_segmenter = segmenter->segmenter.get();
  absl::Status cpp_result;
  if (options) {
    ::mediapipe::tasks::vision::core::ImageProcessingOptions cpp_options;
    CppConvertToImageProcessingOptions(*options, &cpp_options);
    cpp_result =
        cpp_segmenter->SegmentAsync(ToImage(image), timestamp_ms, cpp_options);
  } else {
    cpp_result = cpp_segmenter->SegmentAsync(ToImage(image), timestamp_ms);
  }

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

int CppImageSegmenterClose(MpImageSegmenterPtr segmenter, char** error_msg) {
  auto cpp_segmenter = segmenter->segmenter.get();
  auto result = cpp_segmenter->Close();
  if (!result.ok()) {
    LOG(ERROR) << "Failed to close ImageSegmenter: " << result;
    return CppProcessError(result, error_msg);
  }
  delete segmenter;
  return 0;
}

}  // namespace mediapipe::tasks::c::vision::image_segmenter

extern "C" {

MP_EXPORT MpImageSegmenterPtr image_segmenter_create(
    struct ImageSegmenterOptions* options, char** error_msg) {
  return mediapipe::tasks::c::vision::image_segmenter::CppImageSegmenterCreate(
      *options, error_msg);
}

MP_EXPORT int image_segmenter_segment_image(MpImageSegmenterPtr segmenter,
                                            MpImagePtr image,
                                            ImageSegmenterResult* result,
                                            char** error_msg) {
  return mediapipe::tasks::c::vision::image_segmenter::CppImageSegmenterSegment(
      segmenter, image, /*options=*/nullptr, result, error_msg);
}

MP_EXPORT int image_segmenter_segment_image_with_options(
    MpImageSegmenterPtr segmenter, MpImagePtr image,
    const ImageProcessingOptions* options, ImageSegmenterResult* result,
    char** error_msg) {
  return mediapipe::tasks::c::vision::image_segmenter::CppImageSegmenterSegment(
      segmenter, image, options, result, error_msg);
}

MP_EXPORT int image_segmenter_segment_for_video(MpImageSegmenterPtr segmenter,
                                                MpImagePtr image,
                                                int64_t timestamp_ms,
                                                ImageSegmenterResult* result,
                                                char** error_msg) {
  return mediapipe::tasks::c::vision::image_segmenter::
      CppImageSegmenterSegmentForVideo(segmenter, image, /*options=*/nullptr,
                                       timestamp_ms, result, error_msg);
}

MP_EXPORT int image_segmenter_segment_for_video_with_options(
    MpImageSegmenterPtr segmenter, MpImagePtr image,
    const ImageProcessingOptions* options, int64_t timestamp_ms,
    ImageSegmenterResult* result, char** error_msg) {
  return mediapipe::tasks::c::vision::image_segmenter::
      CppImageSegmenterSegmentForVideo(segmenter, image, options, timestamp_ms,
                                       result, error_msg);
}

MP_EXPORT int image_segmenter_segment_async(MpImageSegmenterPtr segmenter,
                                            MpImagePtr image,
                                            int64_t timestamp_ms,
                                            char** error_msg) {
  return mediapipe::tasks::c::vision::image_segmenter::
      CppImageSegmenterSegmentAsync(segmenter, image, /*options=*/nullptr,
                                    timestamp_ms, error_msg);
}

MP_EXPORT int image_segmenter_segment_async_with_options(
    MpImageSegmenterPtr segmenter, MpImagePtr image,
    const ImageProcessingOptions* options, int64_t timestamp_ms,
    char** error_msg) {
  return mediapipe::tasks::c::vision::image_segmenter::
      CppImageSegmenterSegmentAsync(segmenter, image, options, timestamp_ms,
                                    error_msg);
}

MP_EXPORT void image_segmenter_close_result(ImageSegmenterResult* result) {
  mediapipe::tasks::c::vision::image_segmenter::CppImageSegmenterCloseResult(
      result);
}

MP_EXPORT int image_segmenter_close(MpImageSegmenterPtr segmenter,
                                    char** error_ms) {
  return mediapipe::tasks::c::vision::image_segmenter::CppImageSegmenterClose(
      segmenter, error_ms);
}

}  // extern "C"
