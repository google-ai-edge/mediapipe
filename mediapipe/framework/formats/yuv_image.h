// Copyright 2019 The MediaPipe Authors.
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

#ifndef MEDIAPIPE_FRAMEWORK_FORMATS_YUV_IMAGE_H_
#define MEDIAPIPE_FRAMEWORK_FORMATS_YUV_IMAGE_H_

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <utility>

#include "absl/base/attributes.h"
#include "absl/functional/any_invocable.h"
#include "absl/types/span.h"
#include "libyuv/video_common.h"

namespace mediapipe {

// Generic data structure for representing various 8-bit YUV image formats with
// pixel format specification in FourCC. The class is also capable of
// representing higher bit depth YUV image formats (10-bit, 12-bit, or 16-bit)
// where each format uses the lower bits of a uint16_t. For these high bit depth
// configurations, only the fully planar representation (i.e., u/v are not
// interleaved) with chroma subsampling of 420 is supported. Although there are
// high bit depth fourcc codes, none of them are defined or supported by libyuv,
// and there does not appear to be a standard code for the fully planar 10-bit
// format we use (this format is efficient for in memory manipulation but not
// necessarily for transport). Therefore, when bit_depth > 8, the only allowable
// chroma subsampling is 420 and the corresponding fourc_cc will be FOURCC_ANY.
//
// This class is primarily designed as a wrapper around 8-bit YUV image formats
// used by Android (NV21, YV12) and FFmpeg (I420 a.k.a. YCbCr420P).
//
// Note that YUV and YCbCr, although often used interchangeably, are different.
// The YUV color space was developed for analog systems and is not defined
// precisely in the technical and scientific literature; instead, it refers to a
// whole family of luminance/chrominance color spaces.  On the other hand, the
// YCbCr color space is defined in the ITU-R BT.601-5 and ITU-R BT.709-5
// standards of ITU (International Telecommunication Union) for digital systems.
// Thus, YCbCr420P is referring to a specific digital color space and a specific
// storage format.
//
// Class takes ownership of the pixel data buffers provided as input to the
// constructor or Initialize().
//
// A typical FFmpeg usage would be:
//
//   AVFrame frame;
//   avcodec_decode_video2(&codec_context, &frame, &got_frame, &av_packet);
//   const size_t y_size = frame.linesize[0] * height;
//   const size_t u_size = frame.linesize[1] * ((height + 1) / 2);
//   const size_t v_size = frame.linesize[2] * ((height + 1) / 2);
//   const size_t total_size = y_size + u_size + v_size;
//   auto data = std::make_unique<uint8_t[]>(total_size);
//   auto data_span = absl::MakeSpan(data.get(), total_size);
//   auto y = data_span.subspan(0, y_size);
//   auto u = data_span.subspan(y_size, u_size);
//   auto v = data_span.subspan(y_size + u_size, v_size);
//   libyuv::I420Copy(frame.data[0], frame.linesize[0],
//                    frame.data[1], frame.linesize[1],
//                    frame.data[2], frame.linesize[2],
//                    y.data(), frame.linesize[0],
//                    u.data(), frame.linesize[1],
//                    v.data(), frame.linesize[2],
//                    width, height);
//   auto yuv_image = std::make_unique<YUVImage>();
//   yuv_image->Initialize(
//       libyuv::FOURCC_I420,
//       [data = std::move(data)]() mutable { data.reset(); },  //
//       y, frame.linesize[0],                                  //
//       u, frame.linesize[1],                                  //
//       v, frame.linesize[2],                                  //
//       width, height);
//   Outputs().Tag("VIDEO")->Add(yuv_image.release(), timestamp);
//
// Note that for formats with subsampled U and V channels, like I420, the
// dimensions of the U and V channels are half the dimensions of the Y channel,
// rounded up. Rounding up can be accomplished by adding one to the Y dimensions
// before dividing by 2.
//
// Please do not add new constructors unless it is unavoidable; the default
// constructor followed by Initialize() should cover most of the use cases.
class YUVImage {
 public:
  // The matrix coefficients used (e.g., defines the conversion matrix from
  // Ycbcr
  // to RGB).
  enum ColorMatrixCoefficients {
    COLOR_MATRIX_COEFFICIENTS_RGB = 0,
    // Also ITU-R BT1361 / IEC 61966-2-4 xvYCC709 / SMPTE RP177 Annex B.
    COLOR_MATRIX_COEFFICIENTS_BT709 = 1,
    COLOR_MATRIX_COEFFICIENTS_UNSPECIFIED = 2,
    COLOR_MATRIX_COEFFICIENTS_FCC = 4,
    // Also ITU-R BT601-6 625 / ITU-R BT1358 625 / ITU-R BT1700 625 PAL &
    /// SECAM / IEC 61966-2-4 xvYCC601.
    COLOR_MATRIX_COEFFICIENTS_BT470BG = 5,
    // Also ITU-R BT601-6 525 / ITU-R BT1358 525 / ITU-R BT1700 NTSC /
    /// functionally identical to above.
    COLOR_MATRIX_COEFFICIENTS_SMPTE170M = 6,
    COLOR_MATRIX_COEFFICIENTS_SMPTE240M = 7,
    // Used by Dirac / VC-2 and H.264 FRext, see ITU-T SG16.
    COLOR_MATRIX_COEFFICIENTS_YCOCG = 8,
    // ITU-R BT2020 non-constant luminance system.
    COLOR_MATRIX_COEFFICIENTS_BT2020_NCL = 9,
    // ITU-R BT2020 constant luminance system.
    COLOR_MATRIX_COEFFICIENTS_BT2020_CL = 10,
    // SMPTE 2085, Y'D'zD'x
    COLOR_MATRIX_COEFFICIENTS_SMPTE2085 = 11,
    // Chromaticity-derived non-constant luminance.
    COLOR_MATRIX_COEFFICIENTS_CHROMA_DERIVED_NCL = 12,
    // Chromaticity-derived constant luminance.
    COLOR_MATRIX_COEFFICIENTS_CHROMA_DERIVED_CL = 13,
    // ITU-R BT.[HDR-TV] ICtCp
    COLOR_MATRIX_COEFFICIENTS_ICTCP = 14,
  };

  YUVImage() = default;
  ~YUVImage() { Clear(); }

  // YUVImage is move-only.
  YUVImage(const YUVImage&) = delete;
  YUVImage& operator=(const YUVImage&) = delete;
  YUVImage(YUVImage&& b) { *this = std::move(b); }

  YUVImage& operator=(YUVImage&& b) {
    if (this != &b) {
      Clear();
      deallocation_function_ = std::exchange(b.deallocation_function_, nullptr);
      fourcc_ = std::exchange(b.fourcc_, libyuv::FOURCC_ANY);
      std::swap_ranges(data_, data_ + kMaxNumPlanes, b.data_);
      std::swap_ranges(stride_, stride_ + kMaxNumPlanes, b.stride_);
      width_ = std::exchange(b.width_, 0);
      height_ = std::exchange(b.height_, 0);
      bit_depth_ = std::exchange(b.bit_depth_, 0);
      matrix_coefficients_ = std::exchange(
          b.matrix_coefficients_, COLOR_MATRIX_COEFFICIENTS_UNSPECIFIED);
      full_range_ = std::exchange(b.full_range_, false);
    }
    return *this;
  }

  // Convenience constructor
  ABSL_DEPRECATED("Use the span-based Initialize instead.")
  YUVImage(libyuv::FourCC fourcc,                     //
           std::unique_ptr<uint8_t[]> data_location,  //
           uint8_t* data0, int stride0,               //
           uint8_t* data1, int stride1,               //
           uint8_t* data2, int stride2,               //
           int width, int height, int bit_depth = 8) {
    absl::AnyInvocable<void()> deallocation_function =
        [data = std::move(data_location)]() mutable { data.reset(); };
    Initialize(fourcc,                            //
               std::move(deallocation_function),  //
               data0, stride0,                    //
               data1, stride1,                    //
               data2, stride2,                    //
               width, height, bit_depth);
  }

  // Convenience constructor to construct the YUVImage with data stored
  // in three unique_ptrs.
  ABSL_DEPRECATED("Use the span-based Initialize instead.")
  YUVImage(libyuv::FourCC fourcc,                          //
           std::unique_ptr<uint8_t[]> data0, int stride0,  //
           std::unique_ptr<uint8_t[]> data1, int stride1,  //
           std::unique_ptr<uint8_t[]> data2, int stride2,  //
           int width, int height, int bit_depth = 8) {
    uint8_t* ptr0 = data0.get();
    uint8_t* ptr1 = data1.get();
    uint8_t* ptr2 = data2.get();
    absl::AnyInvocable<void()> deallocation_function =
        [d0 = std::move(data0), d1 = std::move(data1),
         d2 = std::move(data2)]() mutable {
          d0.reset();
          d1.reset();
          d2.reset();
        };
    Initialize(fourcc,                            //
               std::move(deallocation_function),  //
               ptr0, stride0,                     //
               ptr1, stride1,                     //
               ptr2, stride2,                     //
               width, height,                     //
               bit_depth);
  }

  // Convenience constructor to resolve ambiguity when nullptr is passed for all
  // planes.
  ABSL_DEPRECATED("Use the span-based Initialize instead.")
  YUVImage(libyuv::FourCC fourcc,        //
           std::nullptr_t, int stride0,  //
           std::nullptr_t, int stride1,  //
           std::nullptr_t, int stride2,  //
           int width, int height, int bit_depth = 8)
      : YUVImage(fourcc, std::unique_ptr<uint8_t[]>(), stride0,
                 std::unique_ptr<uint8_t[]>(), stride1,
                 std::unique_ptr<uint8_t[]>(), stride2, width, height,
                 bit_depth) {}

  ABSL_DEPRECATED("Use the span-based Initialize instead.")
  void Initialize(libyuv::FourCC fourcc,                             //
                  absl::AnyInvocable<void()> deallocation_function,  //
                  uint8_t* data0, int stride0,                       //
                  uint8_t* data1, int stride1,                       //
                  uint8_t* data2, int stride2,                       //
                  int width, int height, int bit_depth = 8) {
    // SAFETY: The span size is a best-effort guess based on stride, height,
    // fourcc and bit depth.
    Initialize(fourcc,                            //
               std::move(deallocation_function),  //
               SpanFromPtr(data0, stride0, fourcc, 0, height, bit_depth),
               stride0,  //
               SpanFromPtr(data1, stride1, fourcc, 1, height, bit_depth),
               stride1,  //
               SpanFromPtr(data2, stride2, fourcc, 2, height, bit_depth),
               stride2,  //
               width, height, bit_depth);
  }

  // Clear and initialize member variables.
  //
  // First argument is an enum of FourCC (see http://www.fourcc.org/yuv.php)
  // defined in libyuv/video_common.h
  //
  // A deallocation function is provided which will be called on the next
  // Clear() or on destruction.
  //
  // The next three argument pairs are spans to pixel data buffers for each
  // plane and its image stride (http://en.wikipedia.org/wiki/Stride).
  // Spans that are empty are treated as null planes.
  //
  // The class is very generic and it is up to the user how they want
  // to use this data holder class.  For example, if one intends to
  // use this for NV21, one can ignore data2 and stride2 by giving
  // an empty span and 0, respectively, and call the right libyuv functions
  // for actual processing.  This class is agnostic of the data and the
  // pixel format it holds.
  void Initialize(libyuv::FourCC fourcc,                             //
                  absl::AnyInvocable<void()> deallocation_function,  //
                  absl::Span<uint8_t> data0, int stride0,            //
                  absl::Span<uint8_t> data1, int stride1,            //
                  absl::Span<uint8_t> data2, int stride2,            //
                  int width, int height, int bit_depth = 8) {
    Clear();
    deallocation_function_ = std::move(deallocation_function);
    fourcc_ = fourcc;
    data_[0] = data0;
    stride_[0] = stride0;
    data_[1] = data1;
    stride_[1] = stride1;
    data_[2] = data2;
    stride_[2] = stride2;
    width_ = width;
    height_ = height;
    bit_depth_ = bit_depth;
  }

  void Clear() {
    if (deallocation_function_) {
      deallocation_function_();
      deallocation_function_ = nullptr;
    }
    fourcc_ = libyuv::FOURCC_ANY;
    data_[0] = {};
    data_[1] = {};
    data_[2] = {};
    stride_[0] = 0;
    stride_[1] = 0;
    stride_[2] = 0;
    width_ = 0;
    height_ = 0;
    bit_depth_ = 0;
  }

  // Getters.
  libyuv::FourCC fourcc() const { return fourcc_; }
  const uint8_t* data(int index) const {
    return data_[index].empty() ? nullptr : data_[index].data();
  }
  absl::Span<const uint8_t> data_span(int index) const { return data_[index]; }
  int stride(int index) const { return stride_[index]; }
  int width() const { return width_; }
  int height() const { return height_; }
  int bit_depth() const { return bit_depth_; }
  ColorMatrixCoefficients matrix_coefficients() const {
    return matrix_coefficients_;
  }
  bool full_range() const { return full_range_; }

  // Setters.
  void set_fourcc(libyuv::FourCC fourcc) { fourcc_ = fourcc; }
  uint8_t* mutable_data(int index) {
    return data_[index].empty() ? nullptr : data_[index].data();
  }
  absl::Span<uint8_t> mutable_data_span(int index) { return data_[index]; }
  void set_stride(int index, int stride) { stride_[index] = stride; }
  void set_width(int width) { width_ = width; }
  void set_height(int height) { height_ = height; }
  void set_matrix_coefficients(ColorMatrixCoefficients coeffs) {
    matrix_coefficients_ = coeffs;
  }
  void set_full_range(bool full_range) { full_range_ = full_range; }

 private:
  static constexpr int kMaxNumPlanes = 3;

  // Constructs a span from a raw pointer by guessing the buffer size.
  //
  // This is a terrible thing to do in general, but exists as a fallback to
  // support callers that use YUVImage's legacy constructors with raw pointers.
  static absl::Span<uint8_t> SpanFromPtr(uint8_t* data, int stride,
                                         libyuv::FourCC fourcc, int plane_index,
                                         int height, int bit_depth) {
    if (data == nullptr || stride <= 0 || height <= 0) {
      return {};
    }
    const uint32_t canonical_fourcc =
        libyuv::CanonicalFourCC(static_cast<uint32_t>(fourcc));

    // Formats with only 1 plane (luma-only or packed): planes 1 and 2 are
    // empty.
    if (plane_index > 0) {
      switch (canonical_fourcc) {
        case libyuv::FOURCC_I400:
        case libyuv::FOURCC_J400:
        case libyuv::FOURCC_YUY2:
        case libyuv::FOURCC_UYVY:
        case libyuv::FOURCC_MJPG:
        case libyuv::FOURCC_ARGB:
        case libyuv::FOURCC_BGRA:
        case libyuv::FOURCC_ABGR:
        case libyuv::FOURCC_RGBA:
        case libyuv::FOURCC_RAW:
        case libyuv::FOURCC_24BG:
        case libyuv::FOURCC_RGBP:
        case libyuv::FOURCC_RGBO:
        case libyuv::FOURCC_R444:
        case libyuv::FOURCC_AR30:
        case libyuv::FOURCC_AB30:
        case libyuv::FOURCC_AR64:
        case libyuv::FOURCC_AB64:
          return {};
        default:
          break;
      }
    }

    // Semi-planar formats: plane 2 is empty (UV are interleaved in plane 1).
    if (plane_index == 2) {
      switch (canonical_fourcc) {
        case libyuv::FOURCC_NV12:
        case libyuv::FOURCC_NV21:
        case libyuv::FOURCC_P010:
        case libyuv::FOURCC_P210:
        case libyuv::FOURCC_M420:
          return {};
        default:
          break;
      }
    }

    int plane_height = height;
    if (plane_index > 0) {
      if (bit_depth > 8) {
        plane_height = (height + 1) / 2;
      } else {
        switch (canonical_fourcc) {
          case libyuv::FOURCC_I420:
          case libyuv::FOURCC_YV12:
          case libyuv::FOURCC_NV12:
          case libyuv::FOURCC_NV21:
          case libyuv::FOURCC_J420:
          case libyuv::FOURCC_H420:
          case libyuv::FOURCC_F420:
          case libyuv::FOURCC_U420:
          case libyuv::FOURCC_M420:
          case libyuv::FOURCC_Q420:
            plane_height = (height + 1) / 2;
            break;
          default:
            break;
        }
      }
    }
    // SAFETY: This is a best-effort guess. For unknown formats, stride * height
    // is used as an upper bound for all standard YUV plane sizes.
    return absl::MakeSpan(data, static_cast<size_t>(stride) * plane_height);
  }

  absl::AnyInvocable<void()> deallocation_function_;

  libyuv::FourCC fourcc_ = libyuv::FOURCC_ANY;
  absl::Span<uint8_t> data_[kMaxNumPlanes] = {};
  int stride_[kMaxNumPlanes] = {};
  int width_ = 0;
  int height_ = 0;
  int bit_depth_ = 0;
  ColorMatrixCoefficients matrix_coefficients_ =
      ColorMatrixCoefficients::COLOR_MATRIX_COEFFICIENTS_UNSPECIFIED;
  bool full_range_ = false;
};

}  // namespace mediapipe

#endif  // MEDIAPIPE_FRAMEWORK_FORMATS_YUV_IMAGE_H_
