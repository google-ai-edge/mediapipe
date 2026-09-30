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

#include "mediapipe/util/image_frame_util.h"

#include <cmath>
#include <cstdint>
#include <functional>

#include "absl/log/absl_check.h"
#include "absl/log/absl_log.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/types/span.h"
#include "libyuv/scale.h"
#include "libyuv/video_common.h"
#include "mediapipe/framework/deps/mathutil.h"
#include "mediapipe/framework/formats/image_frame.h"
#include "mediapipe/framework/formats/image_frame_opencv.h"
#include "mediapipe/framework/formats/yuv_image.h"
#include "mediapipe/framework/port/aligned_malloc_and_free.h"
#include "mediapipe/framework/port/libyuv_port.h"
#include "mediapipe/framework/port/port.h"
#include "mediapipe/framework/port/ret_check.h"
#include "mediapipe/framework/port/status_macros.h"

namespace mediapipe {

namespace image_frame_util {

void RescaleImageFrame(const ImageFrame& source_frame, const int width,
                       const int height, const int alignment_boundary,
                       const int open_cv_interpolation_algorithm,
                       ImageFrame* destination_frame) {
  ABSL_CHECK(destination_frame);
  ABSL_CHECK_EQ(ImageFormat::SRGB, source_frame.Format());

  cv::Mat source_mat = ::mediapipe::formats::MatView(&source_frame);
  destination_frame->Reset(source_frame.Format(), width, height,
                           alignment_boundary);
  cv::Mat destination_mat = ::mediapipe::formats::MatView(destination_frame);
  image_frame_util::RescaleSrgbImage(source_mat, width, height,
                                     open_cv_interpolation_algorithm,
                                     &destination_mat);
}

void RescaleSrgbImage(const cv::Mat& source, const int width, const int height,
                      const int open_cv_interpolation_algorithm,
                      cv::Mat* destination) {
  ABSL_CHECK(destination);

  // Convert input_mat into 16 bit per channel linear RGB space.
  cv::Mat input_mat16;
  image_frame_util::SrgbToLinearRgb16(source, &input_mat16);

  // Resize in 16 bit linear RGB space.
  cv::Mat output_mat16;
  // Notice that OpenCV assumes the image is in BGR pixel ordering.
  // However, in resizing, the channel ordering is irrelevant so there
  // is no need to convert the channel order.
  cv::resize(input_mat16, output_mat16, cv::Size(width, height), 0.0, 0.0,
             open_cv_interpolation_algorithm);

  // Convert back to SRGB colorspace.
  image_frame_util::LinearRgb16ToSrgb(output_mat16, destination);
}

void ImageFrameToYUVImage(const ImageFrame& image_frame, YUVImage* yuv_image) {
  const int width = image_frame.Width();
  const int height = image_frame.Height();
  const int uv_width = (width + 1) / 2;
  const int uv_height = (height + 1) / 2;
  // Align y_stride and uv_stride on 16-byte boundaries.
  const int y_stride = (width + 15) & ~15;
  const int uv_stride = (uv_width + 15) & ~15;
  const int y_size = y_stride * height;
  const int uv_size = uv_stride * uv_height;
  uint8_t* data =
      reinterpret_cast<uint8_t*>(aligned_malloc(y_size + uv_size * 2, 16));
  std::function<void()> deallocate = [data]() { aligned_free(data); };
  auto y = absl::MakeSpan(data, y_size);
  auto u = absl::MakeSpan(data + y_size, uv_size);
  auto v = absl::MakeSpan(data + y_size + uv_size, uv_size);
  yuv_image->Initialize(libyuv::FOURCC_I420, deallocate,  //
                        y, y_stride,                      //
                        u, uv_stride,                     //
                        v, uv_stride,                     //
                        width, height);
  switch (image_frame.Format()) {
    case ImageFormat::SRGBA:
      // ABGR little endian (RGBA in memory).
      ABSL_CHECK_OK(libyuv_port::ABGRToI420(
          {image_frame.PixelDataSpan(), image_frame.WidthStep()},  //
          {y, y_stride},                                           //
          {u, uv_stride},                                          //
          {v, uv_stride},                                          //
          width, height));
      break;
    case ImageFormat::SRGB:
      // RAW in libyuv is byte order R, G, B, see libyuv/convert.h.
      ABSL_CHECK_OK(libyuv_port::RAWToI420(
          {image_frame.PixelDataSpan(), image_frame.WidthStep()},  //
          {y, y_stride},                                           //
          {u, uv_stride},                                          //
          {v, uv_stride},                                          //
          width, height));
      break;
    default:
      ABSL_LOG(ERROR)
          << "Using RGB conversion for unexpected image frame format";
      // RAW in libyuv is byte order R, G, B, see libyuv/convert.h.
      ABSL_CHECK_OK(libyuv_port::RAWToI420(
          {image_frame.PixelDataSpan(), image_frame.WidthStep()},  //
          {y, y_stride},                                           //
          {u, uv_stride},                                          //
          {v, uv_stride},                                          //
          width, height));
  }
}

void ImageFrameToYUVNV12Image(const ImageFrame& image_frame,
                              YUVImage* yuv_nv12_image) {
  // Create a YUV I420 image that will hold the converted RGBA image.
  YUVImage yuv_i420_image;
  ImageFrameToYUVImage(image_frame, &yuv_i420_image);

  // Now create a YUV NV12 image and convert the I420 to NV12.
  const int width = yuv_i420_image.width();
  const int height = yuv_i420_image.height();
  const int y_stride = yuv_i420_image.stride(0);
  const int y_size = y_stride * height;
  const int uv_stride = y_stride;
  const int uv_height = (height + 1) / 2;
  const int uv_size = uv_stride * uv_height;
  uint8_t* data =
      reinterpret_cast<uint8_t*>(aligned_malloc(y_size + uv_size, 16));
  std::function<void()> deallocate = [data] { aligned_free(data); };
  auto y = absl::MakeSpan(data, y_size);
  auto uv = absl::MakeSpan(data + y_size, uv_size);
  yuv_nv12_image->Initialize(libyuv::FOURCC_NV12, deallocate,  //
                             y, y_stride,                      //
                             uv, uv_stride,                    //
                             {}, 0,                            //
                             width, height);
  ABSL_CHECK_OK(libyuv_port::I420ToNV12(
      {yuv_i420_image.data_span(0), yuv_i420_image.stride(0)},  //
      {yuv_i420_image.data_span(1), yuv_i420_image.stride(1)},  //
      {yuv_i420_image.data_span(2), yuv_i420_image.stride(2)},  //
      {y, y_stride},                                            //
      {uv, uv_stride},                                          //
      width, height));
}

void YUVImageToImageFrame(const YUVImage& yuv_image, ImageFrame* image_frame,
                          bool use_bt709) {
  ABSL_CHECK(image_frame);
  int width = yuv_image.width();
  int height = yuv_image.height();
  image_frame->Reset(ImageFormat::SRGB, width, height, 16);

  if (use_bt709) {
    ABSL_CHECK_OK(libyuv_port::H420ToRAW(                                 //
        {yuv_image.data_span(0), yuv_image.stride(0)},                    //
        {yuv_image.data_span(1), yuv_image.stride(1)},                    //
        {yuv_image.data_span(2), yuv_image.stride(2)},                    //
        {image_frame->MutablePixelDataSpan(), image_frame->WidthStep()},  //
        width, height));
  } else {
    ABSL_CHECK_OK(libyuv_port::I420ToRAW(                                 //
        {yuv_image.data_span(0), yuv_image.stride(0)},                    //
        {yuv_image.data_span(1), yuv_image.stride(1)},                    //
        {yuv_image.data_span(2), yuv_image.stride(2)},                    //
        {image_frame->MutablePixelDataSpan(), image_frame->WidthStep()},  //
        width, height));
  }
}

void YUVImageToImageFrameFromFormat(const YUVImage& yuv_image,
                                    ImageFrame* image_frame) {
  ABSL_CHECK(image_frame);
  int width = yuv_image.width();
  int height = yuv_image.height();
  image_frame->Reset(ImageFormat::SRGB, width, height, 16);

  const auto& format = yuv_image.fourcc();
  switch (format) {
    case libyuv::FOURCC_NV12:
      // 8-bit Y plane followed by an interleaved 8-bit U/V plane with 2×2
      // subsampling.
      ABSL_CHECK_OK(libyuv_port::NV12ToRAW(
          {yuv_image.data_span(0), yuv_image.stride(0)},  //
          {yuv_image.data_span(1), yuv_image.stride(1)},  //
          {image_frame->MutablePixelDataSpan(), image_frame->WidthStep()},
          width, height));
      break;
    case libyuv::FOURCC_NV21:
      // 8-bit Y plane followed by an interleaved 8-bit V/U plane with 2×2
      // subsampling.
      ABSL_CHECK_OK(libyuv_port::NV21ToRAW(
          {yuv_image.data_span(0), yuv_image.stride(0)},                    //
          {yuv_image.data_span(1), yuv_image.stride(1)},                    //
          {image_frame->MutablePixelDataSpan(), image_frame->WidthStep()},  //
          width, height));
      break;
    case libyuv::FOURCC_I420:
      // Also known as YV21.
      // 8-bit Y plane followed by 8-bit 2×2 subsampled U and V planes.
      ABSL_CHECK_OK(libyuv_port::I420ToRAW(                                 //
          {yuv_image.data_span(0), yuv_image.stride(0)},                    //
          {yuv_image.data_span(1), yuv_image.stride(1)},                    //
          {yuv_image.data_span(2), yuv_image.stride(2)},                    //
          {image_frame->MutablePixelDataSpan(), image_frame->WidthStep()},  //
          width, height));
      break;
    case libyuv::FOURCC_YV12:
      // 8-bit Y plane followed by 8-bit 2×2 subsampled V and U planes.
      ABSL_CHECK_OK(libyuv_port::I420ToRAW(                                 //
          {yuv_image.data_span(0), yuv_image.stride(0)},                    //
          {yuv_image.data_span(2), yuv_image.stride(2)},                    //
          {yuv_image.data_span(1), yuv_image.stride(1)},                    //
          {image_frame->MutablePixelDataSpan(), image_frame->WidthStep()},  //
          width, height));
      break;
    default:
      ABSL_LOG(FATAL) << "Unsupported YUVImage format.";
  }
}

absl::Status ScaleYUVImage(const YUVImage& source, int output_width,
                           int output_height, YUVImage* destination) {
  RET_CHECK(destination != nullptr);
  RET_CHECK_LE(source.bit_depth(), 8);
  RET_CHECK_GT(output_width, 0);
  RET_CHECK_GT(output_height, 0);

  const int y_size = output_width * output_height;
  const int uv_width = (output_width + 1) / 2;
  const int uv_height = (output_height + 1) / 2;
  const int uv_size = uv_width * uv_height;
  const libyuv::FourCC format = source.fourcc();
  if (format == libyuv::FOURCC_I420 || format == libyuv::FOURCC_YV12 ||
      format == libyuv::FOURCC_ANY) {
    const int u_plane = (format == libyuv::FOURCC_YV12) ? 2 : 1;
    const int v_plane = (format == libyuv::FOURCC_YV12) ? 1 : 2;
    uint8_t* data =
        reinterpret_cast<uint8_t*>(aligned_malloc(y_size + uv_size * 2, 16));
    absl::Span<uint8_t> y(data, y_size);
    absl::Span<uint8_t> u(data + y_size, uv_size);
    absl::Span<uint8_t> v(data + y_size + uv_size, uv_size);
    const int rv = libyuv::I420Scale(
        source.data(0), source.stride(0), source.data(u_plane),
        source.stride(u_plane), source.data(v_plane), source.stride(v_plane),
        source.width(), source.height(), y.data(), output_width, u.data(),
        uv_width, v.data(), uv_width, output_width, output_height,
        libyuv::kFilterBox);
    if (rv != 0) {
      aligned_free(data);
    }
    RET_CHECK_EQ(rv, 0);
    destination->Initialize(
        libyuv::FOURCC_I420, [data]() { aligned_free(data); }, y, output_width,
        u, uv_width, v, uv_width, output_width, output_height);
    return absl::OkStatus();
  }
  if (format == libyuv::FOURCC_NV12 || format == libyuv::FOURCC_NV21) {
    const int uv_stride = uv_width * 2;
    uint8_t* data =
        reinterpret_cast<uint8_t*>(aligned_malloc(y_size + uv_size * 2, 16));
    absl::Span<uint8_t> y(data, y_size);
    absl::Span<uint8_t> uv(data + y_size, uv_size * 2);
    const int rv = libyuv::NV12Scale(
        source.data(0), source.stride(0), source.data(1), source.stride(1),
        source.width(), source.height(), y.data(), output_width, uv.data(),
        uv_stride, output_width, output_height, libyuv::kFilterBox);
    if (rv != 0) {
      aligned_free(data);
    }
    RET_CHECK_EQ(rv, 0);
    destination->Initialize(
        format, [data]() { aligned_free(data); }, y, output_width, uv,
        uv_stride, {}, 0, output_width, output_height);
    return absl::OkStatus();
  }
  return absl::InvalidArgumentError(
      absl::StrCat("Unsupported YUVImage format: ", format));
}

void SrgbToMpegYCbCr(const uint8_t r, const uint8_t g, const uint8_t b,  //
                     uint8_t* y, uint8_t* cb, uint8_t* cr) {
  // ITU-R BT.601 conversion from sRGB to YCbCr.
  // FastIntRound is used rather than SafeRound since the possible
  // range of values is [16,235] for Y and [16,240] for Cb and Cr and we
  // don't care about the rounding direction for values exactly between
  // two integers.
  *y = static_cast<uint8_t>(
      mediapipe::MathUtil::FastIntRound(16.0 +                 //
                                        65.481 * r / 255.0 +   //
                                        128.553 * g / 255.0 +  //
                                        24.966 * b / 255.0));
  *cb = static_cast<uint8_t>(
      mediapipe::MathUtil::FastIntRound(128.0 +                //
                                        -37.797 * r / 255.0 +  //
                                        -74.203 * g / 255.0 +  //
                                        112.0 * b / 255.0));
  *cr = static_cast<uint8_t>(
      mediapipe::MathUtil::FastIntRound(128.0 +                //
                                        112.0 * r / 255.0 +    //
                                        -93.786 * g / 255.0 +  //
                                        -18.214 * b / 255.0));
}

void MpegYCbCrToSrgb(const uint8_t y, const uint8_t cb, const uint8_t cr,  //
                     uint8_t* r, uint8_t* g, uint8_t* b) {
  // ITU-R BT.601 conversion from YCbCr to sRGB
  // Use SafeRound since many MPEG YCbCr values do not correspond directly
  // to an sRGB value.
  *r = mediapipe::MathUtil::SafeRound<uint8_t, double>(  //
      255.0 / 219.0 * (y - 16.0) +                       //
      255.0 / 112.0 * 0.701 * (cr - 128.0));
  *g = mediapipe::MathUtil::SafeRound<uint8_t, double>(
      255.0 / 219.0 * (y - 16.0) -                            //
      255.0 / 112.0 * 0.886 * 0.114 / 0.587 * (cb - 128.0) -  //
      255.0 / 112.0 * 0.701 * 0.299 / 0.587 * (cr - 128.0));
  *b = mediapipe::MathUtil::SafeRound<uint8_t, double>(  //
      255.0 / 219.0 * (y - 16.0) +                       //
      255.0 / 112.0 * 0.886 * (cb - 128.0));
}

// SrgbToLinearRgb16() and LinearRgb16ToSrgb() internally use LUTs (lookup
// tables) to avoid repeated floating point computation.  These helper functions
// create and initialize the LUTs respectively.
//
// The conversion constants and formulae were taken from
// http://en.wikipedia.org/wiki/SRGB and double-checked with other sources.

cv::Mat GetSrgbToLinearRgb16Lut() {
  cv::Mat lut(1, 256, CV_16UC1);
  uint16_t* ptr = lut.ptr<uint16_t>();
  constexpr double kUint8Max = 255.0;
  constexpr double kUint16Max = 65535.0;
  for (int i = 0; i < 256; ++i) {
    if (i < 0.04045 * kUint8Max) {
      ptr[i] = static_cast<uint16_t>(
          (static_cast<double>(i) / kUint8Max / 12.92) * kUint16Max + .5);
    } else {
      ptr[i] = static_cast<uint16_t>(
          pow((static_cast<double>(i) / kUint8Max + 0.055) / 1.055, 2.4) *
              kUint16Max +
          .5);
    }
  }
  return lut;
}

cv::Mat GetLinearRgb16ToSrgbLut() {
  cv::Mat lut(1, 65536, CV_8UC1);
  uint8_t* ptr = lut.ptr<uint8_t>();
  constexpr double kUint8Max = 255.0;
  constexpr double kUint16Max = 65535.0;
  for (int i = 0; i < 65536; ++i) {
    if (i < 0.0031308 * kUint16Max) {
      ptr[i] = static_cast<uint8_t>(
          (static_cast<double>(i) / kUint16Max * 12.92) * kUint8Max + .5);
    } else {
      ptr[i] = static_cast<uint8_t>(
          (1.055 * pow(static_cast<double>(i) / kUint16Max, 1.0 / 2.4) - .055) *
              kUint8Max +
          .5);
    }
  }
  return lut;
}

void SrgbToLinearRgb16(const cv::Mat& source, cv::Mat* destination) {
  static const cv::Mat kLut = GetSrgbToLinearRgb16Lut();
  cv::LUT(source, kLut, *destination);
}

void LinearRgb16ToSrgb(const cv::Mat& source, cv::Mat* destination) {
  // Ensure the destination is in the proper format (OpenCV style).
  destination->create(source.size(), CV_8UC(source.channels()));

  static const cv::Mat kLut = GetLinearRgb16ToSrgbLut();
  const uint8_t* lookup_table_ptr = kLut.ptr<uint8_t>();
  const int num_channels = source.channels();
  for (int row = 0; row < source.rows; ++row) {
    uint8_t* ptr = destination->ptr<uint8_t>(row);
    const uint16_t* ptr16 = source.ptr<uint16_t>(row);
    for (int col = 0; col < source.cols; ++col) {
      for (int channel = 0; channel < num_channels; ++channel) {
        *ptr = lookup_table_ptr[*ptr16];
        ++ptr;
        ++ptr16;
      }
    }
  }
}

}  // namespace image_frame_util
}  // namespace mediapipe
