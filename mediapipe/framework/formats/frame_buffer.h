/* Copyright 2022 The MediaPipe Authors. All Rights Reserved.

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

#ifndef MEDIAPIPE_FRAMEWORK_FORMATS_FRAME_BUFFER_H_
#define MEDIAPIPE_FRAMEWORK_FORMATS_FRAME_BUFFER_H_

#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/memory/memory.h"
#include "absl/status/statusor.h"
#include "mediapipe/framework/port/integral_types.h"

namespace mediapipe {

// A `FrameBuffer` provides a view into the provided backing buffer (e.g. camera
// frame or still image) with buffer format information. FrameBuffer doesn't
// take ownership of the provided backing buffer. The caller is responsible to
// manage the backing buffer lifecycle for the lifetime of the FrameBuffer.
//
// Examples:
//
// // Create an metadata instance with no backing buffer.
// auto buffer = FrameBuffer::Create(/*planes=*/{}, dimension, kRGBA,
//                                   KTopLeft);
//
// // Create an RGBA instance with backing buffer on single plane.
// FrameBuffer::Plane plane =
//     {rgba_buffer, /*stride=*/{dimension.width * 4, 4}};
// auto buffer = FrameBuffer::Create({plane}, dimension, kRGBA, kTopLeft);
//
// // Create an YUV instance with planar backing buffer.
// FrameBuffer::Plane y_plane = {y_buffer, /*stride=*/{dimension.width , 1}};
// FrameBuffer::Plane uv_plane = {u_buffer, /*stride=*/{dimension.width, 2}};
// auto buffer = FrameBuffer::Create({y_plane, uv_plane}, dimension, kNV21,
//                                   kLeftTop);
class FrameBuffer {
 public:
  // Colorspace formats.
  enum class Format {
    kRGBA,
    kRGB,
    kNV12,
    kNV21,
    kYV12,
    kYV21,
    kGRAY,
    kUNKNOWN
  };

  // Stride information.
  struct Stride {
    // The row stride in bytes. This is the distance between the start pixels of
    // two consecutive rows in the image.
    int row_stride_bytes;
    // This is the distance between two consecutive pixel values in a row of
    // pixels in bytes. It may be larger than the size of a single pixel to
    // account for interleaved image data or padded formats.
    int pixel_stride_bytes;

    bool operator==(const Stride& other) const {
      return row_stride_bytes == other.row_stride_bytes &&
             pixel_stride_bytes == other.pixel_stride_bytes;
    }

    bool operator!=(const Stride& other) const { return !operator==(other); }
  };

  // YUV data structure.
  struct YuvData {
    const uint8* y_buffer;
    const uint8* u_buffer;
    const uint8* v_buffer;
    // Y buffer row stride in bytes.
    int y_row_stride;
    // U/V buffer row stride in bytes.
    int uv_row_stride;
    // U/V pixel stride in bytes. This is the distance between two consecutive
    // u/v pixel values in a row.
    int uv_pixel_stride;
  };

  // FrameBuffer content orientation follows EXIF specification. The name of
  // each enum value defines the position of the 0th row and the 0th column of
  // the image content. See http://jpegclub.org/exif_orientation.html for
  // details.
  enum class Orientation {
    kTopLeft = 1,
    kTopRight = 2,
    kBottomRight = 3,
    kBottomLeft = 4,
    kLeftTop = 5,
    kRightTop = 6,
    kRightBottom = 7,
    kLeftBottom = 8
  };

  // Plane encapsulates buffer and stride information.
  struct Plane {
    const uint8* buffer;
    Stride stride;
  };

  // Dimension information for the whole frame or a cropped portion of it.
  struct Dimension {
    // The width dimension in pixel unit.
    int width;
    // The height dimension in pixel unit.
    int height;

    bool operator==(const Dimension& other) const {
      return width == other.width && height == other.height;
    }

    bool operator!=(const Dimension& other) const {
      return width != other.width || height != other.height;
    }

    bool operator>=(const Dimension& other) const {
      return width >= other.width && height >= other.height;
    }

    bool operator<=(const Dimension& other) const {
      return width <= other.width && height <= other.height;
    }

    // Swaps width and height.
    void Swap() {
      using std::swap;
      swap(width, height);
    }

    // Returns area represented by width * height.
    int Size() const { return width * height; }
  };

  // Factory method for creating a FrameBuffer object from row-major backing
  // buffers.
  static std::unique_ptr<FrameBuffer> Create(const std::vector<Plane>& planes,
                                             Dimension dimension, Format format,
                                             Orientation orientation) {
    return absl::make_unique<FrameBuffer>(planes, dimension, format,
                                          orientation);
  }

  // Factory method for creating a FrameBuffer object from row-major movable
  // backing buffers.
  static std::unique_ptr<FrameBuffer> Create(std::vector<Plane>&& planes,
                                             Dimension dimension, Format format,
                                             Orientation orientation) {
    return absl::make_unique<FrameBuffer>(std::move(planes), dimension, format,
                                          orientation);
  }

  // Returns YuvData which contains the Y, U, and V buffer and their
  // stride info from the input `source` FrameBuffer which is in the YUV family
  // formats (e.g NV12, NV21, YV12, and YV21).
  static absl::StatusOr<YuvData> GetYuvDataFromFrameBuffer(
      const FrameBuffer& source);

  // Builds a FrameBuffer object from a row-major backing buffer.
  //
  // The FrameBuffer does not take ownership of the backing buffer. The backing
  // buffer is read-only and the caller is responsible for maintaining the
  // backing buffer lifecycle for the lifetime of FrameBuffer.
  FrameBuffer(const std::vector<Plane>& planes, Dimension dimension,
              Format format, Orientation orientation)
      : planes_(planes),
        dimension_(dimension),
        format_(format),
        orientation_(orientation) {}

  // Builds a FrameBuffer object from a movable row-major backing buffer.
  //
  // The FrameBuffer does not take ownership of the backing buffer. The backing
  // buffer is read-only and the caller is responsible for maintaining the
  // backing buffer lifecycle for the lifetime of FrameBuffer.
  FrameBuffer(std::vector<Plane>&& planes, Dimension dimension, Format format,
              Orientation orientation)
      : planes_(std::move(planes)),
        dimension_(dimension),
        format_(format),
        orientation_(orientation) {}

  // Copy constructor.
  //
  // FrameBuffer does not take ownership of the backing buffer. The copy
  // constructor behaves the same way to only copy the buffer pointer and not
  // take ownership of the backing buffer.
  FrameBuffer(const FrameBuffer& frame_buffer) {
    planes_.clear();
    for (int i = 0; i < frame_buffer.plane_count(); i++) {
      planes_.push_back(
          FrameBuffer::Plane{.buffer = frame_buffer.plane(i).buffer,
                             .stride = frame_buffer.plane(i).stride});
    }
    dimension_ = frame_buffer.dimension();
    format_ = frame_buffer.format();
    orientation_ = frame_buffer.orientation();
  }

  // Returns number of planes.
  int plane_count() const { return planes_.size(); }

  // Returns plane indexed by the input `index`.
  Plane plane(int index) const {
    if (index > -1 && static_cast<size_t>(index) < planes_.size()) {
      return planes_[index];
    }
    return {};
  }

  // Returns FrameBuffer dimension.
  Dimension dimension() const { return dimension_; }

  // Returns FrameBuffer format.
  Format format() const { return format_; }

  // Returns FrameBuffer orientation.
  Orientation orientation() const { return orientation_; }

 private:
  std::vector<Plane> planes_;
  Dimension dimension_;
  Format format_;
  Orientation orientation_;
};

}  // namespace mediapipe

#endif  // MEDIAPIPE_FRAMEWORK_FORMATS_FRAME_BUFFER_H_
