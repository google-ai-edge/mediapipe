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

#include "mediapipe/tasks/c/vision/interactive_segmenter/interactive_segmenter.h"

#include <memory>
#include <utility>
#include <vector>

#include "absl/log/absl_log.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "mediapipe/framework/formats/image.h"
#include "mediapipe/framework/formats/image_frame.h"
#include "mediapipe/tasks/c/core/base_options_converter.h"
#include "mediapipe/tasks/c/vision/core/common.h"
#include "mediapipe/tasks/c/vision/image_segmenter/image_segmenter_result.h"
#include "mediapipe/tasks/c/vision/image_segmenter/image_segmenter_result_converter.h"
#include "mediapipe/tasks/cc/components/containers/keypoint.h"
#include "mediapipe/tasks/cc/vision/interactive_segmenter/interactive_segmenter.h"
#include "mediapipe/tasks/cc/vision/utils/image_utils.h"

namespace mediapipe::tasks::c::vision::interactive_segmenter {

namespace {

using ::mediapipe::tasks::c::components::containers::
    CppCloseImageSegmenterResult;
using ::mediapipe::tasks::c::components::containers::
    CppConvertToImageSegmenterResult;
using ::mediapipe::tasks::c::core::CppConvertToBaseOptions;
using ::mediapipe::tasks::vision::CreateImageFromBuffer;
using ::mediapipe::tasks::vision::interactive_segmenter::InteractiveSegmenter;
typedef ::mediapipe::tasks::vision::interactive_segmenter::RegionOfInterest::
    Format CppRegionOfInterestFormat;
typedef ::mediapipe::tasks::components::containers::NormalizedKeypoint
    CppNormalizedKeypoint;

int CppProcessError(absl::Status status, char** error_msg) {
  if (error_msg) {
    *error_msg = strdup(status.ToString().c_str());
  }
  return status.raw_code();
}

}  // namespace

void CppConvertToRegionOfInterest(
    const RegionOfInterest& in,
    mediapipe::tasks::vision::interactive_segmenter::RegionOfInterest* out) {
  // Convert format
  switch (in.format) {
    case RegionOfInterest::kKeypoint:
      out->format = CppRegionOfInterestFormat::kKeyPoint;
      break;
    case RegionOfInterest::kScribble:
      out->format = CppRegionOfInterestFormat::kScribble;
      break;
    default:
      out->format = CppRegionOfInterestFormat::kUnspecified;
  }

  // Convert keypoint
  if (in.format == RegionOfInterest::kKeypoint) {
    out->keypoint = CppNormalizedKeypoint{in.keypoint->x, in.keypoint->y};
  }

  // Convert scribble
  if (in.format == RegionOfInterest::kScribble) {
    out->scribble = std::vector<CppNormalizedKeypoint>();
    for (int i = 0; i < in.scribble_count; ++i) {
      out->scribble->emplace_back(
          CppNormalizedKeypoint{in.scribble[i].x, in.scribble[i].y});
    }
  }
}

void CppConvertToInteractiveSegmenterOptions(
    const InteractiveSegmenterOptions& in,
    mediapipe::tasks::vision::interactive_segmenter::
        InteractiveSegmenterOptions* out) {
  out->output_confidence_masks = in.output_confidence_masks;
  out->output_category_mask = in.output_category_mask;
}

InteractiveSegmenter* CppInteractiveSegmenterCreate(
    const InteractiveSegmenterOptions& options, char** error_msg) {
  auto cpp_options =
      std::make_unique<::mediapipe::tasks::vision::interactive_segmenter::
                           InteractiveSegmenterOptions>();

  CppConvertToBaseOptions(options.base_options, &cpp_options->base_options);
  CppConvertToInteractiveSegmenterOptions(options, cpp_options.get());

  auto segmenter = InteractiveSegmenter::Create(std::move(cpp_options));
  if (!segmenter.ok()) {
    ABSL_LOG(ERROR) << "Failed to create InteractiveSegmenter: "
                    << segmenter.status();
    CppProcessError(segmenter.status(), error_msg);
    return nullptr;
  }
  return segmenter->release();
}

int CppInteractiveSegmenterSegment(void* segmenter, const MpImage& image,
                                   const RegionOfInterest& region_of_interest,
                                   ImageSegmenterResult* result,
                                   char** error_msg) {
  if (image.type == MpImage::GPU_BUFFER) {
    const absl::Status status =
        absl::InvalidArgumentError("GPU Buffer not supported yet.");

    ABSL_LOG(ERROR) << "Segmentation failed: " << status.message();
    return CppProcessError(status, error_msg);
  }

  const auto img = CreateImageFromBuffer(
      static_cast<ImageFormat::Format>(image.image_frame.format),
      image.image_frame.image_buffer, image.image_frame.width,
      image.image_frame.height);

  mediapipe::tasks::vision::interactive_segmenter::RegionOfInterest roi;
  CppConvertToRegionOfInterest(region_of_interest, &roi);

  if (!img.ok()) {
    ABSL_LOG(ERROR) << "Failed to create Image: " << img.status();
    return CppProcessError(img.status(), error_msg);
  }

  auto cpp_segmenter = static_cast<InteractiveSegmenter*>(segmenter);
  auto cpp_result = cpp_segmenter->Segment(*img, roi);
  if (!cpp_result.ok()) {
    ABSL_LOG(ERROR) << "Segmentation failed: " << cpp_result.status();
    return CppProcessError(cpp_result.status(), error_msg);
  }
  CppConvertToImageSegmenterResult(*cpp_result, result);
  return 0;
}

void CppImageSegmenterCloseResult(ImageSegmenterResult* result) {
  CppCloseImageSegmenterResult(result);
}

int CppInteractiveSegmenterClose(void* segmenter, char** error_msg) {
  auto cpp_segmenter = static_cast<InteractiveSegmenter*>(segmenter);
  auto result = cpp_segmenter->Close();
  if (!result.ok()) {
    ABSL_LOG(ERROR) << "Failed to close InteractiveSegmenter: " << result;
    return CppProcessError(result, error_msg);
  }
  delete cpp_segmenter;
  return 0;
}

}  // namespace mediapipe::tasks::c::vision::interactive_segmenter

extern "C" {

void* interactive_segmenter_create(struct InteractiveSegmenterOptions* options,
                                   char** error_msg) {
  return mediapipe::tasks::c::vision::interactive_segmenter::
      CppInteractiveSegmenterCreate(*options, error_msg);
}

int interactive_segmenter_segment_image(void* segmenter, const MpImage& image,
                                        const RegionOfInterest& roi,
                                        ImageSegmenterResult* result,
                                        char** error_msg) {
  return mediapipe::tasks::c::vision::interactive_segmenter::
      CppInteractiveSegmenterSegment(segmenter, image, roi, result, error_msg);
}

void interactive_segmenter_close_result(ImageSegmenterResult* result) {
  mediapipe::tasks::c::vision::interactive_segmenter::
      CppImageSegmenterCloseResult(result);
}

int interactive_segmenter_close(void* segmenter, char** error_ms) {
  return mediapipe::tasks::c::vision::interactive_segmenter::
      CppInteractiveSegmenterClose(segmenter, error_ms);
}

}  // extern "C"
