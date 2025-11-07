// Copyright 2020-2021 The MediaPipe Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef MEDIAPIPE_TASKS_C_VISION_CORE_IMAGE_FRAME_UTIL_H_
#define MEDIAPIPE_TASKS_C_VISION_CORE_IMAGE_FRAME_UTIL_H_

#include <cstddef>
#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

#include "absl/memory/memory.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "mediapipe/framework/formats/image.h"
#include "mediapipe/framework/formats/image_format.pb.h"
#include "mediapipe/framework/formats/image_frame.h"

// C API wrapper for MediaPipe Image, which can optionally store cached
// contiguous image data to allow efficient access from Python.
typedef struct MpImageInternal {
  mediapipe::Image image;
  mutable std::vector<uint8_t> cached_contiguous_data;
} MpImageInternal;

namespace mediapipe {
namespace tasks {
namespace vision {
namespace core {

template <typename T>
std::unique_ptr<ImageFrame> CreateImageFrame(
    mediapipe::ImageFormat::Format format, T* data, const int rows,
    const int cols, bool copy = true) {
  int width_step = ImageFrame::NumberOfChannelsForFormat(format) *
                   ImageFrame::ByteDepthForFormat(format) * cols;
  if (copy) {
    auto image_frame = absl::make_unique<ImageFrame>(
        format, /*width=*/cols, /*height=*/rows, width_step,
        static_cast<uint8_t*>(data), ImageFrame::PixelDataDeleter::kNone);
    auto image_frame_copy = absl::make_unique<ImageFrame>();
    // Set alignment_boundary to kGlDefaultAlignmentBoundary so that both
    // GPU and CPU can process it.
    image_frame_copy->CopyFrom(*image_frame,
                               ImageFrame::kGlDefaultAlignmentBoundary);
    return image_frame_copy;
  }
  auto image_frame = absl::make_unique<ImageFrame>(
      format, /*width=*/cols, /*height=*/rows, width_step,
      static_cast<uint8_t*>(data),
      /*deleter=*/[data](uint8_t*) { free(data); });
  return image_frame;
}

template <typename T>
inline const T* GenerateContiguousDataArray(const MpImageInternal* image) {
  std::shared_ptr<ImageFrame> image_frame =
      image->image.GetImageFrameSharedPtr();
  if (image_frame->IsContiguous()) {
    return reinterpret_cast<const T*>(image_frame->PixelData());
  } else {
    size_t buffer_size = image_frame->PixelDataSizeStoredContiguously();
    std::vector<uint8_t> contiguous_data_copy(buffer_size);
    image_frame->CopyToBuffer(contiguous_data_copy.data(), buffer_size);
    image->cached_contiguous_data = std::move(contiguous_data_copy);
    return reinterpret_cast<const T*>(image->cached_contiguous_data.data());
  }
}

// Generates a contiguous data pyarray object on demand.
// This function only accepts an image frame object that already stores
// contiguous data. The output py::array points to the raw pixel data array of
// the image frame object directly.
template <typename T>
inline absl::StatusOr<const T*> GenerateDataArrayOnDemand(
    const MpImageInternal* image) {
  std::shared_ptr<ImageFrame> image_frame =
      image->image.GetImageFrameSharedPtr();
  if (!image_frame->IsContiguous()) {
    return absl::InvalidArgumentError(
        "GenerateDataArrayOnDemand must take an ImageFrame "
        "object that stores contiguous data.");
  }
  return GenerateContiguousDataArray<T>(image);
}

// Gets the a pointer to a contiguous data array that stores the image data.
//
// If the image frame is already contiguous, the function returns a pointer to
// the raw pixel data array of the image frame object directly. Otherwise, the
// function returns a pointer to the cached contiguous data array or generates
// the contiguous data array and stores the result for efficient access in
// future calls
template <typename T>
inline absl::StatusOr<const T*> GetCachedContiguousDataAttr(
    const MpImageInternal* image) {
  std::shared_ptr<ImageFrame> image_frame =
      image->image.GetImageFrameSharedPtr();
  if (image_frame->IsContiguous()) {
    return reinterpret_cast<const T*>(image_frame->PixelData());
  }
  if (image_frame->IsEmpty()) {
    return absl::InvalidArgumentError("ImageFrame is unallocated.");
  }
  // If cached_contiguous_data attr doesn't store data yet, generates the
  // contiguous data array object and caches the result.
  if (image->cached_contiguous_data.empty()) {
    GenerateContiguousDataArray<T>(image);
  }
  return reinterpret_cast<const T*>(image->cached_contiguous_data.data());
}

template <typename T>
absl::StatusOr<T> GetValue(const MpImageInternal* image,
                           const std::vector<int>& pos) {
  std::shared_ptr<ImageFrame> image_frame =
      image->image.GetImageFrameSharedPtr();
  const uint8_t* output_array = image_frame->PixelData();
  size_t offset = 0;
  if (pos.size() == 2) {
    offset =
        pos[0] * image_frame->WidthStep() + pos[1] * image_frame->ByteDepth();
  } else if (pos.size() == 3) {
    offset =
        pos[0] * image_frame->WidthStep() +
        pos[1] * image_frame->NumberOfChannels() * image_frame->ByteDepth() +
        pos[2] * image_frame->ByteDepth();
  } else {
    return absl::InvalidArgumentError(
        absl::StrCat("Invalid index dimension: ", pos.size()));
  }
  return *reinterpret_cast<const T*>(output_array + offset);
}

}  // namespace core
}  // namespace vision
}  // namespace tasks
}  // namespace mediapipe

#endif  // MEDIAPIPE_TASKS_C_VISION_CORE_IMAGE_FRAME_UTIL_H_
