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
#include <optional>
#include <utility>
#include <vector>

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
#include "mediapipe/tasks/cc/components/containers/keypoint.h"
#include "mediapipe/tasks/cc/vision/core/image_processing_options.h"
#include "mediapipe/tasks/cc/vision/interactive_segmenter/interactive_segmenter.h"

struct MpInteractiveSegmenterInternal {
  std::unique_ptr<
      ::mediapipe::tasks::vision::interactive_segmenter::InteractiveSegmenter>
      segmenter;
};

namespace mediapipe::tasks::c::vision::interactive_segmenter {

namespace {

using ::mediapipe::tasks::c::components::containers::
    CppCloseImageSegmenterResult;
using ::mediapipe::tasks::c::components::containers::
    CppConvertToImageSegmenterResult;
using ::mediapipe::tasks::c::core::CppConvertToBaseOptions;
using ::mediapipe::tasks::c::vision::core::CppConvertToImageProcessingOptions;
using ::mediapipe::tasks::vision::interactive_segmenter::InteractiveSegmenter;
typedef ::mediapipe::tasks::vision::interactive_segmenter::RegionOfInterest::
    Format CppRegionOfInterestFormat;
typedef ::mediapipe::tasks::components::containers::NormalizedKeypoint
    CppNormalizedKeypoint;
using CppImageProcessingOptions =
    ::mediapipe::tasks::vision::core::ImageProcessingOptions;
using ::mediapipe::tasks::c::core::ToMpStatus;

const Image& ToImage(const MpImagePtr mp_image) { return mp_image->image; }

}  // namespace

void CppConvertToRegionOfInterest(
    const RegionOfInterest* in,
    mediapipe::tasks::vision::interactive_segmenter::RegionOfInterest* out) {
  // Convert format
  switch (in->format) {
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
  if (in->format == RegionOfInterest::kKeypoint) {
    out->keypoint = CppNormalizedKeypoint{in->keypoint->x, in->keypoint->y};
  }

  // Convert scribble
  if (in->format == RegionOfInterest::kScribble) {
    out->scribble = std::vector<CppNormalizedKeypoint>();
    for (int i = 0; i < in->scribble_count; ++i) {
      out->scribble->emplace_back(
          CppNormalizedKeypoint{in->scribble[i].x, in->scribble[i].y});
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

MpStatus CppInteractiveSegmenterCreate(
    const InteractiveSegmenterOptions& options,
    MpInteractiveSegmenterPtr* segmenter) {
  auto cpp_options =
      std::make_unique<::mediapipe::tasks::vision::interactive_segmenter::
                           InteractiveSegmenterOptions>();

  CppConvertToBaseOptions(options.base_options, &cpp_options->base_options);
  CppConvertToInteractiveSegmenterOptions(options, cpp_options.get());

  auto cpp_segmenter = InteractiveSegmenter::Create(std::move(cpp_options));
  if (!cpp_segmenter.ok()) {
    ABSL_LOG(ERROR) << "Failed to create InteractiveSegmenter: "
                    << cpp_segmenter.status();
    return ToMpStatus(cpp_segmenter.status());
  }
  *segmenter = new MpInteractiveSegmenterInternal{
      .segmenter = std::move(*cpp_segmenter)};
  return kMpOk;
}

MpStatus CppInteractiveSegmenterSegment(
    MpInteractiveSegmenterPtr segmenter, MpImagePtr image,
    const RegionOfInterest* region_of_interest,
    const ImageProcessingOptions* image_processing_options,
    ImageSegmenterResult* result) {
  auto cpp_segmenter = segmenter->segmenter.get();
  std::optional<CppImageProcessingOptions> cpp_image_processing_options;
  if (image_processing_options) {
    CppImageProcessingOptions options;
    CppConvertToImageProcessingOptions(*image_processing_options, &options);
    cpp_image_processing_options = options;
  }
  mediapipe::tasks::vision::interactive_segmenter::RegionOfInterest cpp_roi;
  CppConvertToRegionOfInterest(region_of_interest, &cpp_roi);
  auto cpp_result = cpp_segmenter->Segment(ToImage(image), cpp_roi,
                                           cpp_image_processing_options);
  if (!cpp_result.ok()) {
    ABSL_LOG(ERROR) << "Segmentation failed: " << cpp_result.status();
    return ToMpStatus(cpp_result.status());
  }
  CppConvertToImageSegmenterResult(*cpp_result, result);
  return kMpOk;
}

void CppImageSegmenterCloseResult(ImageSegmenterResult* result) {
  CppCloseImageSegmenterResult(result);
}

MpStatus CppInteractiveSegmenterClose(MpInteractiveSegmenterPtr segmenter) {
  auto cpp_segmenter = segmenter->segmenter.get();
  auto result = cpp_segmenter->Close();
  if (!result.ok()) {
    ABSL_LOG(ERROR) << "Failed to close InteractiveSegmenter: " << result;
    return ToMpStatus(result);
  }
  delete segmenter;
  return kMpOk;
}

}  // namespace mediapipe::tasks::c::vision::interactive_segmenter

extern "C" {

MP_EXPORT MpStatus
MpInteractiveSegmenterCreate(struct InteractiveSegmenterOptions* options,
                             MpInteractiveSegmenterPtr* segmenter) {
  return mediapipe::tasks::c::vision::interactive_segmenter::
      CppInteractiveSegmenterCreate(*options, segmenter);
}

MP_EXPORT MpStatus MpInteractiveSegmenterSegmentImage(
    MpInteractiveSegmenterPtr segmenter, MpImagePtr image,
    const RegionOfInterest* roi,
    const ImageProcessingOptions* image_processing_options,
    ImageSegmenterResult* result) {
  return mediapipe::tasks::c::vision::interactive_segmenter::
      CppInteractiveSegmenterSegment(segmenter, image, roi,
                                     image_processing_options, result);
}

MP_EXPORT void MpInteractiveSegmenterCloseResult(ImageSegmenterResult* result) {
  mediapipe::tasks::c::vision::interactive_segmenter::
      CppImageSegmenterCloseResult(result);
}

MP_EXPORT MpStatus
MpInteractiveSegmenterClose(MpInteractiveSegmenterPtr segmenter) {
  return mediapipe::tasks::c::vision::interactive_segmenter::
      CppInteractiveSegmenterClose(segmenter);
}

}  // extern "C"
