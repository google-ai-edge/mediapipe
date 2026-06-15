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

#include "mediapipe/tasks/c/vision/interactive_segmenter_legacy/interactive_segmenter_legacy.h"

#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include "absl/log/absl_check.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "mediapipe/framework/formats/image.h"
#include "mediapipe/tasks/c/core/base_options_converter.h"
#include "mediapipe/tasks/c/core/common.h"
#include "mediapipe/tasks/c/core/mp_status.h"
#include "mediapipe/tasks/c/core/mp_status_converter.h"
#include "mediapipe/tasks/c/vision/core/image.h"
#include "mediapipe/tasks/c/vision/core/image_frame_util.h"
#include "mediapipe/tasks/c/vision/core/image_processing_options.h"
#include "mediapipe/tasks/c/vision/core/image_processing_options_converter.h"
#include "mediapipe/tasks/c/vision/image_segmenter/image_segmenter_result.h"
#include "mediapipe/tasks/c/vision/image_segmenter/image_segmenter_result_converter.h"
#include "mediapipe/tasks/cc/components/containers/keypoint.h"
#include "mediapipe/tasks/cc/vision/core/image_processing_options.h"
#include "mediapipe/tasks/cc/vision/interactive_segmenter_legacy/interactive_segmenter_legacy.h"

struct MpInteractiveSegmenterLegacyInternal {
  std::unique_ptr<::mediapipe::tasks::vision::interactive_segmenter_legacy::
                      InteractiveSegmenterLegacy>
      instance;
};

namespace mediapipe::tasks::c::vision::interactive_segmenter_legacy {

namespace {

using ::mediapipe::tasks::c::components::containers::
    CppCloseImageSegmenterResult;
using ::mediapipe::tasks::c::components::containers::
    CppConvertToImageSegmenterResult;
using ::mediapipe::tasks::c::core::CppConvertToBaseOptions;
using ::mediapipe::tasks::c::vision::core::CppConvertToImageProcessingOptions;
using ::mediapipe::tasks::vision::interactive_segmenter_legacy::
    InteractiveSegmenterLegacy;
typedef ::mediapipe::tasks::vision::interactive_segmenter_legacy::
    RegionOfInterest::Format CppRegionOfInterestFormat;
typedef ::mediapipe::tasks::components::containers::NormalizedKeypoint
    CppNormalizedKeypoint;
using CppImageProcessingOptions =
    ::mediapipe::tasks::vision::core::ImageProcessingOptions;
using ::mediapipe::tasks::c::core::ToMpStatus;

const Image& ToImage(const MpImagePtr mp_image) { return mp_image->image; }

InteractiveSegmenterLegacy* GetCppSegmenter(
    MpInteractiveSegmenterLegacyPtr wrapper) {
  ABSL_CHECK(wrapper != nullptr) << "InteractiveSegmenterLegacy is null.";
  return wrapper->instance.get();
}

}  // namespace

void CppConvertToRegionOfInterest(
    const MpRegionOfInterest* in,
    mediapipe::tasks::vision::interactive_segmenter_legacy::RegionOfInterest*
        out) {
  // Convert format
  switch (in->format) {
    case MP_REGION_OF_INTEREST_FORMAT_KEYPOINT:
      out->format = CppRegionOfInterestFormat::kKeyPoint;
      break;
    case MP_REGION_OF_INTEREST_FORMAT_SCRIBBLE:
      out->format = CppRegionOfInterestFormat::kScribble;
      break;
    default:
      out->format = CppRegionOfInterestFormat::kUnspecified;
  }

  // Convert keypoint
  if (in->format == MP_REGION_OF_INTEREST_FORMAT_KEYPOINT) {
    out->keypoint = CppNormalizedKeypoint{in->keypoint->x, in->keypoint->y};
  }

  // Convert scribble
  if (in->format == MP_REGION_OF_INTEREST_FORMAT_SCRIBBLE) {
    out->scribble = std::vector<CppNormalizedKeypoint>();
    for (int i = 0; i < in->scribble_count; ++i) {
      out->scribble->emplace_back(
          CppNormalizedKeypoint{in->scribble[i].x, in->scribble[i].y});
    }
  }
}

void CppConvertToInteractiveSegmenterLegacyOptions(
    const MpInteractiveSegmenterLegacyOptions& in,
    mediapipe::tasks::vision::interactive_segmenter_legacy::
        InteractiveSegmenterLegacyOptions* out) {
  out->output_confidence_masks = in.output_confidence_masks;
  out->output_category_mask = in.output_category_mask;
}

absl::Status CppInteractiveSegmenterLegacyCreate(
    const MpInteractiveSegmenterLegacyOptions& options,
    MpInteractiveSegmenterLegacyPtr* segmenter) {
  auto cpp_options = std::make_unique<
      ::mediapipe::tasks::vision::interactive_segmenter_legacy::
          InteractiveSegmenterLegacyOptions>();

  CppConvertToBaseOptions(options.base_options, &cpp_options->base_options);
  CppConvertToInteractiveSegmenterLegacyOptions(options, cpp_options.get());

  auto cpp_segmenter =
      InteractiveSegmenterLegacy::Create(std::move(cpp_options));
  if (!cpp_segmenter.ok()) {
    return cpp_segmenter.status();
  }
  *segmenter = new MpInteractiveSegmenterLegacyInternal{
      .instance = std::move(*cpp_segmenter)};
  return absl::OkStatus();
}

absl::Status CppInteractiveSegmenterLegacySegment(
    MpInteractiveSegmenterLegacyPtr segmenter, MpImagePtr image,
    const MpRegionOfInterest* region_of_interest,
    const MpImageProcessingOptions* image_processing_options,
    MpImageSegmenterResult* result) {
  auto cpp_segmenter = GetCppSegmenter(segmenter);
  std::optional<CppImageProcessingOptions> cpp_image_processing_options;
  if (image_processing_options) {
    CppImageProcessingOptions options;
    CppConvertToImageProcessingOptions(*image_processing_options, &options);
    cpp_image_processing_options = options;
  }
  mediapipe::tasks::vision::interactive_segmenter_legacy::RegionOfInterest
      cpp_roi;
  CppConvertToRegionOfInterest(region_of_interest, &cpp_roi);
  auto cpp_result = cpp_segmenter->Segment(ToImage(image), cpp_roi,
                                           cpp_image_processing_options);
  if (!cpp_result.ok()) {
    return cpp_result.status();
  }
  CppConvertToImageSegmenterResult(*cpp_result, result);
  return absl::OkStatus();
}

void CppImageSegmenterCloseResult(MpImageSegmenterResult* result) {
  CppCloseImageSegmenterResult(result);
}

absl::Status CppInteractiveSegmenterLegacyClose(
    MpInteractiveSegmenterLegacyPtr segmenter) {
  auto cpp_segmenter = GetCppSegmenter(segmenter);
  auto result = cpp_segmenter->Close();
  if (!result.ok()) {
    return result;
  }
  delete segmenter;
  return absl::OkStatus();
}

}  // namespace mediapipe::tasks::c::vision::interactive_segmenter_legacy

extern "C" {

MP_EXPORT MpStatus MpInteractiveSegmenterLegacyCreate(
    struct MpInteractiveSegmenterLegacyOptions* options,
    MpInteractiveSegmenterLegacyPtr* segmenter, char** error_msg) {
  absl::Status status =
      mediapipe::tasks::c::vision::interactive_segmenter_legacy::
          CppInteractiveSegmenterLegacyCreate(*options, segmenter);
  return mediapipe::tasks::c::core::HandleStatus(status, error_msg);
}

MP_EXPORT MpStatus MpInteractiveSegmenterLegacySegmentImage(
    MpInteractiveSegmenterLegacyPtr segmenter, MpImagePtr image,
    const MpRegionOfInterest* roi,
    const MpImageProcessingOptions* image_processing_options,
    MpImageSegmenterResult* result, char** error_msg) {
  absl::Status status = mediapipe::tasks::c::vision::
      interactive_segmenter_legacy::CppInteractiveSegmenterLegacySegment(
          segmenter, image, roi, image_processing_options, result);
  return mediapipe::tasks::c::core::HandleStatus(status, error_msg);
}

MP_EXPORT void MpInteractiveSegmenterLegacyCloseResult(
    MpImageSegmenterResult* result) {
  mediapipe::tasks::c::vision::interactive_segmenter_legacy::
      CppImageSegmenterCloseResult(result);
}

MP_EXPORT MpStatus MpInteractiveSegmenterLegacyClose(
    MpInteractiveSegmenterLegacyPtr segmenter, char** error_msg) {
  absl::Status status =
      mediapipe::tasks::c::vision::interactive_segmenter_legacy::
          CppInteractiveSegmenterLegacyClose(segmenter);
  return mediapipe::tasks::c::core::HandleStatus(status, error_msg);
}

}  // extern "C"
