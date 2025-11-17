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
#include <optional>
#include <utility>

#include "absl/log/absl_log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "mediapipe/framework/formats/image.h"
#include "mediapipe/tasks/c/core/base_options_converter.h"
#include "mediapipe/tasks/c/core/mp_status.h"
#include "mediapipe/tasks/c/core/mp_status_converter.h"
#include "mediapipe/tasks/c/vision/core/common.h"
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
using ::mediapipe::tasks::c::core::ToMpStatus;
using ::mediapipe::tasks::c::vision::core::CppConvertToImageProcessingOptions;
using ::mediapipe::tasks::vision::core::RunningMode;
using ::mediapipe::tasks::vision::image_segmenter::ImageSegmenter;
using CppImageSegmenterResult =
    ::mediapipe::tasks::vision::image_segmenter::ImageSegmenterResult;
using CppImageProcessingOptions =
    ::mediapipe::tasks::vision::core::ImageProcessingOptions;

const Image& ToImage(const MpImagePtr mp_image) { return mp_image->image; }

}  // namespace

void CppConvertToImageSegmenterOptions(
    const ImageSegmenterOptions& in,
    mediapipe::tasks::vision::image_segmenter::ImageSegmenterOptions* out) {
  out->display_names_locale = in.display_names_locale;
  out->output_confidence_masks = in.output_confidence_masks;
  out->output_category_mask = in.output_category_mask;
}

MpStatus CppImageSegmenterCreate(const ImageSegmenterOptions& options,
                                 MpImageSegmenterPtr* segmenter) {
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
      ABSL_LOG(ERROR) << "Failed to create ImageSegmenter: " << status;
      return ToMpStatus(status);
    }

    ImageSegmenterOptions::result_callback_fn result_callback =
        options.result_callback;
    cpp_options->result_callback =
        [result_callback](absl::StatusOr<CppImageSegmenterResult> cpp_result,
                          const Image& image, int64_t timestamp) {
          MpImageInternal mp_image({.image = image});
          if (!cpp_result.ok()) {
            result_callback(ToMpStatus(cpp_result.status()), nullptr, &mp_image,
                            timestamp);
            return;
          }
          ImageSegmenterResult result;
          CppConvertToImageSegmenterResult(*cpp_result, &result);
          result_callback(kMpOk, &result, &mp_image, timestamp);
          CppCloseImageSegmenterResult(&result);
        };
  }

  auto cpp_segmenter = ImageSegmenter::Create(std::move(cpp_options));
  if (!cpp_segmenter.ok()) {
    ABSL_LOG(ERROR) << "Failed to create ImageSegmenter: "
                    << cpp_segmenter.status();
    return ToMpStatus(cpp_segmenter.status());
  }
  *segmenter =
      new MpImageSegmenterInternal{.segmenter = std::move(*cpp_segmenter)};
  return kMpOk;
}

MpStatus CppImageSegmenterSegment(
    MpImageSegmenterPtr segmenter, MpImagePtr image,
    const ImageProcessingOptions* image_processing_options,
    ImageSegmenterResult* result) {
  auto cpp_segmenter = segmenter->segmenter.get();
  std::optional<CppImageProcessingOptions> cpp_image_processing_options;
  if (image_processing_options) {
    CppImageProcessingOptions options;
    CppConvertToImageProcessingOptions(*image_processing_options, &options);
    cpp_image_processing_options = options;
  }
  auto cpp_result =
      cpp_segmenter->Segment(ToImage(image), cpp_image_processing_options);

  if (!cpp_result.ok()) {
    ABSL_LOG(ERROR) << "Segmentation failed: " << cpp_result.status();
    return ToMpStatus(cpp_result.status());
  }
  CppConvertToImageSegmenterResult(*cpp_result, result);
  return kMpOk;
}

MpStatus CppImageSegmenterSegmentForVideo(
    MpImageSegmenterPtr segmenter, MpImagePtr image,
    const ImageProcessingOptions* image_processing_options,
    int64_t timestamp_ms, ImageSegmenterResult* result) {
  auto cpp_segmenter = segmenter->segmenter.get();
  std::optional<CppImageProcessingOptions> cpp_image_processing_options;
  if (image_processing_options) {
    CppImageProcessingOptions options;
    CppConvertToImageProcessingOptions(*image_processing_options, &options);
    cpp_image_processing_options = options;
  }
  auto cpp_result = cpp_segmenter->SegmentForVideo(
      ToImage(image), timestamp_ms, cpp_image_processing_options);

  if (!cpp_result.ok()) {
    LOG(ERROR) << "Segmentation failed: " << cpp_result.status();
    return ToMpStatus(cpp_result.status());
  }
  CppConvertToImageSegmenterResult(*cpp_result, result);
  return kMpOk;
}

MpStatus CppImageSegmenterSegmentAsync(
    MpImageSegmenterPtr segmenter, MpImagePtr image,
    const ImageProcessingOptions* image_processing_options,
    int64_t timestamp_ms) {
  auto cpp_segmenter = segmenter->segmenter.get();
  std::optional<CppImageProcessingOptions> cpp_image_processing_options;
  if (image_processing_options) {
    CppImageProcessingOptions options;
    CppConvertToImageProcessingOptions(*image_processing_options, &options);
    cpp_image_processing_options = options;
  }
  auto cpp_result = cpp_segmenter->SegmentAsync(ToImage(image), timestamp_ms,
                                                cpp_image_processing_options);

  if (!cpp_result.ok()) {
    LOG(ERROR) << "Data preparation for the image segmentation failed: "
               << cpp_result;
    return ToMpStatus(cpp_result);
  }
  return kMpOk;
}

void CppImageSegmenterCloseResult(ImageSegmenterResult* result) {
  CppCloseImageSegmenterResult(result);
}

MpStatus CppImageSegmenterClose(MpImageSegmenterPtr segmenter) {
  auto cpp_segmenter = segmenter->segmenter.get();
  auto result = cpp_segmenter->Close();
  if (!result.ok()) {
    LOG(ERROR) << "Failed to close ImageSegmenter: " << result;
    return ToMpStatus(result);
  }
  delete segmenter;
  return kMpOk;
}

MpStatus CppImageSegmenterGetLabels(MpImageSegmenterPtr segmenter,
                                    MpStringList* label_list) {
  const auto& cpp_labels = segmenter->segmenter->GetLabels();
  if (cpp_labels.empty()) {
    label_list->strings = nullptr;
    label_list->num_strings = 0;
    return kMpOk;
  }

  label_list->num_strings = cpp_labels.size();
  label_list->strings = (char**)malloc(sizeof(char*) * label_list->num_strings);
  for (int i = 0; i < label_list->num_strings; ++i) {
    label_list->strings[i] = strdup(cpp_labels[i].c_str());
  }
  return kMpOk;
}

}  // namespace mediapipe::tasks::c::vision::image_segmenter

extern "C" {

MP_EXPORT MpStatus MpImageSegmenterCreate(struct ImageSegmenterOptions* options,
                                          MpImageSegmenterPtr* segmenter) {
  return mediapipe::tasks::c::vision::image_segmenter::CppImageSegmenterCreate(
      *options, segmenter);
}

MP_EXPORT MpStatus MpImageSegmenterSegmentImage(
    MpImageSegmenterPtr segmenter, MpImagePtr image,
    const ImageProcessingOptions* options, ImageSegmenterResult* result) {
  return mediapipe::tasks::c::vision::image_segmenter::CppImageSegmenterSegment(
      segmenter, image, options, result);
}

MP_EXPORT MpStatus MpImageSegmenterSegmentForVideo(
    MpImageSegmenterPtr segmenter, MpImagePtr image,
    const ImageProcessingOptions* options, int64_t timestamp_ms,
    ImageSegmenterResult* result) {
  return mediapipe::tasks::c::vision::image_segmenter::
      CppImageSegmenterSegmentForVideo(segmenter, image, options, timestamp_ms,
                                       result);
}

MP_EXPORT MpStatus MpImageSegmenterSegmentAsync(
    MpImageSegmenterPtr segmenter, MpImagePtr image,
    const ImageProcessingOptions* options, int64_t timestamp_ms) {
  return mediapipe::tasks::c::vision::image_segmenter::
      CppImageSegmenterSegmentAsync(segmenter, image, options, timestamp_ms);
}

MP_EXPORT void MpImageSegmenterCloseResult(ImageSegmenterResult* result) {
  mediapipe::tasks::c::vision::image_segmenter::CppImageSegmenterCloseResult(
      result);
}

MP_EXPORT MpStatus MpImageSegmenterClose(MpImageSegmenterPtr segmenter) {
  return mediapipe::tasks::c::vision::image_segmenter::CppImageSegmenterClose(
      segmenter);
}

MP_EXPORT MpStatus MpImageSegmenterGetLabels(MpImageSegmenterPtr segmenter,
                                             MpStringList* label_list) {
  return mediapipe::tasks::c::vision::image_segmenter::
      CppImageSegmenterGetLabels(segmenter, label_list);
}

}  // extern "C"
