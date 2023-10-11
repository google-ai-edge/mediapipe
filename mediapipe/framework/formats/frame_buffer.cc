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

#include "mediapipe/framework/formats/frame_buffer.h"

#include "absl/status/status.h"
#include "absl/status/statusor.h"

namespace mediapipe {

namespace {

// Returns whether the input `format` is a supported YUV format.
bool IsSupportedYuvFormat(FrameBuffer::Format format) {
  return format == FrameBuffer::Format::kNV21 ||
         format == FrameBuffer::Format::kNV12 ||
         format == FrameBuffer::Format::kYV12 ||
         format == FrameBuffer::Format::kYV21;
}

// Returns supported 1-plane FrameBuffer in YuvData structure.
absl::StatusOr<FrameBuffer::YuvData> GetYuvDataFromOnePlaneFrameBuffer(
    const FrameBuffer& source) {
  if (!IsSupportedYuvFormat(source.format())) {
    return absl::InvalidArgumentError(
        "The source FrameBuffer format is not part of YUV420 family.");
  }

  FrameBuffer::YuvData result;
  const int y_buffer_size =
      source.plane(0).stride().row_stride_bytes * source.dimension().height;
  const int uv_buffer_size =
      ((source.plane(0).stride().row_stride_bytes + 1) / 2) *
      ((source.dimension().height + 1) / 2);
  result.y_buffer = source.plane(0).buffer();
  result.y_row_stride = source.plane(0).stride().row_stride_bytes;
  result.uv_row_stride = result.y_row_stride;

  if (source.format() == FrameBuffer::Format::kNV21) {
    result.v_buffer = result.y_buffer + y_buffer_size;
    result.u_buffer = result.v_buffer + 1;
    result.uv_pixel_stride = 2;
    // If y_row_stride equals to the frame width and is an odd value,
    // uv_row_stride = y_row_stride + 1, otherwise uv_row_stride = y_row_stride.
    if (result.y_row_stride == source.dimension().width &&
        result.y_row_stride % 2 == 1) {
      result.uv_row_stride = (result.y_row_stride + 1) / 2 * 2;
    }
  } else if (source.format() == FrameBuffer::Format::kNV12) {
    result.u_buffer = result.y_buffer + y_buffer_size;
    result.v_buffer = result.u_buffer + 1;
    result.uv_pixel_stride = 2;
    // If y_row_stride equals to the frame width and is an odd value,
    // uv_row_stride = y_row_stride + 1, otherwise uv_row_stride = y_row_stride.
    if (result.y_row_stride == source.dimension().width &&
        result.y_row_stride % 2 == 1) {
      result.uv_row_stride = (result.y_row_stride + 1) / 2 * 2;
    }
  } else if (source.format() == FrameBuffer::Format::kYV21) {
    result.u_buffer = result.y_buffer + y_buffer_size;
    result.v_buffer = result.u_buffer + uv_buffer_size;
    result.uv_pixel_stride = 1;
    result.uv_row_stride = (result.y_row_stride + 1) / 2;
  } else if (source.format() == FrameBuffer::Format::kYV12) {
    result.v_buffer = result.y_buffer + y_buffer_size;
    result.u_buffer = result.v_buffer + uv_buffer_size;
    result.uv_pixel_stride = 1;
    result.uv_row_stride = (result.y_row_stride + 1) / 2;
  }
  return result;
}

// Returns supported 2-plane FrameBuffer in YuvData structure.
absl::StatusOr<FrameBuffer::YuvData> GetYuvDataFromTwoPlaneFrameBuffer(
    const FrameBuffer& source) {
  if (source.format() != FrameBuffer::Format::kNV12 &&
      source.format() != FrameBuffer::Format::kNV21) {
    return absl::InvalidArgumentError("Unsupported YUV planar format.");
  }

  FrameBuffer::YuvData result;
  // Y plane
  result.y_buffer = source.plane(0).buffer();
  // All plane strides
  result.y_row_stride = source.plane(0).stride().row_stride_bytes;
  result.uv_row_stride = source.plane(1).stride().row_stride_bytes;
  result.uv_pixel_stride = 2;

  if (source.format() == FrameBuffer::Format::kNV12) {
    // Y and UV interleaved format
    result.u_buffer = source.plane(1).buffer();
    result.v_buffer = result.u_buffer + 1;
  } else {
    // Y and VU interleaved format
    result.v_buffer = source.plane(1).buffer();
    result.u_buffer = result.v_buffer + 1;
  }
  return result;
}

// Returns supported 3-plane FrameBuffer in YuvData structure. Note that NV21
// and NV12 are included in the supported Yuv formats. Technically, NV21 and
// NV12 should not be described by the 3-plane format. Historically, NV21 is
// used loosely such that it can also be used to describe YV21 format. For
// backwards compatibility, FrameBuffer supports NV21/NV12 with 3-plane format
// but such usage is discouraged
absl::StatusOr<FrameBuffer::YuvData> GetYuvDataFromThreePlaneFrameBuffer(
    const FrameBuffer& source) {
  if (!IsSupportedYuvFormat(source.format())) {
    return absl::InvalidArgumentError(
        "The source FrameBuffer format is not part of YUV420 family.");
  }

  if (source.plane(1).stride().row_stride_bytes !=
          source.plane(2).stride().row_stride_bytes ||
      source.plane(1).stride().pixel_stride_bytes !=
          source.plane(2).stride().pixel_stride_bytes) {
    return absl::InternalError("Unsupported YUV planar format.");
  }
  FrameBuffer::YuvData result;
  if (source.format() == FrameBuffer::Format::kNV21 ||
      source.format() == FrameBuffer::Format::kYV12) {
    // Y follow by VU order. The VU chroma planes can be interleaved or
    // planar.
    result.y_buffer = source.plane(0).buffer();
    result.v_buffer = source.plane(1).buffer();
    result.u_buffer = source.plane(2).buffer();
    result.y_row_stride = source.plane(0).stride().row_stride_bytes;
    result.uv_row_stride = source.plane(1).stride().row_stride_bytes;
    result.uv_pixel_stride = source.plane(1).stride().pixel_stride_bytes;
  } else {
    // Y follow by UV order. The UV chroma planes can be interleaved or
    // planar.
    result.y_buffer = source.plane(0).buffer();
    result.u_buffer = source.plane(1).buffer();
    result.v_buffer = source.plane(2).buffer();
    result.y_row_stride = source.plane(0).stride().row_stride_bytes;
    result.uv_row_stride = source.plane(1).stride().row_stride_bytes;
    result.uv_pixel_stride = source.plane(1).stride().pixel_stride_bytes;
  }
  return result;
}

}  // namespace

absl::StatusOr<FrameBuffer::YuvData> FrameBuffer::GetYuvDataFromFrameBuffer(
    const FrameBuffer& source) {
  if (!IsSupportedYuvFormat(source.format())) {
    return absl::InvalidArgumentError(
        "The source FrameBuffer format is not part of YUV420 family.");
  }

  if (source.plane_count() == 1) {
    return GetYuvDataFromOnePlaneFrameBuffer(source);
  } else if (source.plane_count() == 2) {
    return GetYuvDataFromTwoPlaneFrameBuffer(source);
  } else if (source.plane_count() == 3) {
    return GetYuvDataFromThreePlaneFrameBuffer(source);
  }
  return absl::InvalidArgumentError(
      "The source FrameBuffer must be consisted by 1, 2, or 3 planes");
}

}  // namespace mediapipe
