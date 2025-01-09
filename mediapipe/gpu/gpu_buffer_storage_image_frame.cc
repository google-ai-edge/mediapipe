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

#include "mediapipe/gpu/gpu_buffer_storage_image_frame.h"

#include <memory>
#include <vector>

#include "absl/log/absl_check.h"
#include "mediapipe/framework/formats/frame_buffer.h"
#include "mediapipe/framework/formats/image_frame.h"

namespace mediapipe {

namespace {

FrameBuffer::Format FrameBufferFormatForImageFrameFormat(
    ImageFormat::Format format) {
  switch (format) {
    case ImageFormat::SRGB:
      return FrameBuffer::Format::kRGB;
    case ImageFormat::SRGBA:
      return FrameBuffer::Format::kRGBA;
    case ImageFormat::GRAY8:
      return FrameBuffer::Format::kGRAY;
    default:
      return FrameBuffer::Format::kUNKNOWN;
  }
}

std::shared_ptr<FrameBuffer> ImageFrameToFrameBuffer(
    std::shared_ptr<ImageFrame> image_frame) {
  FrameBuffer::Format format =
      FrameBufferFormatForImageFrameFormat(image_frame->Format());
  ABSL_CHECK(format != FrameBuffer::Format::kUNKNOWN)
      << "Invalid format. Only SRGB, SRGBA and GRAY8 are supported.";
  const FrameBuffer::Dimension dimension{/*width=*/image_frame->Width(),
                                         /*height=*/image_frame->Height()};
  const FrameBuffer::Stride stride{
      /*row_stride_bytes=*/image_frame->WidthStep(),
      /*pixel_stride_bytes=*/image_frame->ByteDepth() *
          image_frame->NumberOfChannels()};
  const std::vector<FrameBuffer::Plane> planes{
      {image_frame->MutablePixelData(), stride}};
  return std::make_shared<FrameBuffer>(planes, dimension, format);
}

}  // namespace

std::shared_ptr<const FrameBuffer> GpuBufferStorageImageFrame::GetReadView(
    internal::types<FrameBuffer>) const {
  return ImageFrameToFrameBuffer(image_frame_);
}

std::shared_ptr<FrameBuffer> GpuBufferStorageImageFrame::GetWriteView(
    internal::types<FrameBuffer>) {
  return ImageFrameToFrameBuffer(image_frame_);
}

}  // namespace mediapipe
