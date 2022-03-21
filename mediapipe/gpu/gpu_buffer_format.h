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

#ifndef MEDIAPIPE_GPU_GPU_BUFFER_FORMAT_H_
#define MEDIAPIPE_GPU_GPU_BUFFER_FORMAT_H_

#ifdef __APPLE__
#include <CoreVideo/CoreVideo.h>
#if !TARGET_OS_OSX
#define MEDIAPIPE_GPU_BUFFER_USE_CV_PIXEL_BUFFER 1
#endif  // TARGET_OS_OSX
#endif  // defined(__APPLE__)

#include "mediapipe/framework/formats/image_format.pb.h"
#if !MEDIAPIPE_DISABLE_GPU
#include "mediapipe/gpu/gl_base.h"
#endif  // !MEDIAPIPE_DISABLE_GPU

// The behavior of multi-char constants is implementation-defined, so out of an
// excess of caution we define them in this portable way.
#define MEDIAPIPE_FOURCC(a, b, c, d) \
  (((a) << 24) + ((b) << 16) + ((c) << 8) + (d))

namespace mediapipe {

using mediapipe::ImageFormat;

enum class GpuBufferFormat : uint32_t {
  kUnknown = 0,
  kBGRA32 = MEDIAPIPE_FOURCC('B', 'G', 'R', 'A'),
  kRGBA32 = MEDIAPIPE_FOURCC('R', 'G', 'B', 'A'),
  kGrayFloat32 = MEDIAPIPE_FOURCC('L', '0', '0', 'f'),
  kGrayHalf16 = MEDIAPIPE_FOURCC('L', '0', '0', 'h'),
  kOneComponent8 = MEDIAPIPE_FOURCC('L', '0', '0', '8'),
  kOneComponent8Red = MEDIAPIPE_FOURCC('R', '0', '0', '8'),
  kTwoComponent8 = MEDIAPIPE_FOURCC('2', 'C', '0', '8'),
  kTwoComponentHalf16 = MEDIAPIPE_FOURCC('2', 'C', '0', 'h'),
  kTwoComponentFloat32 = MEDIAPIPE_FOURCC('2', 'C', '0', 'f'),
  kBiPlanar420YpCbCr8VideoRange = MEDIAPIPE_FOURCC('4', '2', '0', 'v'),
  kBiPlanar420YpCbCr8FullRange = MEDIAPIPE_FOURCC('4', '2', '0', 'f'),
  kRGB24 = 0x00000018,  // Note: prefer BGRA32 whenever possible.
  kRGBAHalf64 = MEDIAPIPE_FOURCC('R', 'G', 'h', 'A'),
  kRGBAFloat128 = MEDIAPIPE_FOURCC('R', 'G', 'f', 'A'),
};

#if !MEDIAPIPE_DISABLE_GPU
// TODO: make this more generally applicable.
enum class GlVersion {
  kGL = 1,
  kGLES2 = 2,
  kGLES3 = 3,
};

struct GlTextureInfo {
  GLint gl_internal_format;
  GLenum gl_format;
  GLenum gl_type;
  // For multiplane buffers, this represents how many times smaller than
  // the nominal image size a plane is.
  int downscale;
};

const GlTextureInfo& GlTextureInfoForGpuBufferFormat(GpuBufferFormat format,
                                                     int plane,
                                                     GlVersion gl_version);
#endif  // !MEDIAPIPE_DISABLE_GPU

ImageFormat::Format ImageFormatForGpuBufferFormat(GpuBufferFormat format);
GpuBufferFormat GpuBufferFormatForImageFormat(ImageFormat::Format format);

#ifdef __APPLE__

inline OSType CVPixelFormatForGpuBufferFormat(GpuBufferFormat format) {
  switch (format) {
    case GpuBufferFormat::kBGRA32:
      return kCVPixelFormatType_32BGRA;
    case GpuBufferFormat::kRGBA32:
      return kCVPixelFormatType_32RGBA;
    case GpuBufferFormat::kGrayHalf16:
      return kCVPixelFormatType_OneComponent16Half;
    case GpuBufferFormat::kGrayFloat32:
      return kCVPixelFormatType_OneComponent32Float;
    case GpuBufferFormat::kOneComponent8:
      return kCVPixelFormatType_OneComponent8;
    case GpuBufferFormat::kOneComponent8Red:
      return -1;
    case GpuBufferFormat::kTwoComponent8:
      return kCVPixelFormatType_TwoComponent8;
    case GpuBufferFormat::kTwoComponentHalf16:
      return kCVPixelFormatType_TwoComponent16Half;
    case GpuBufferFormat::kTwoComponentFloat32:
      return kCVPixelFormatType_TwoComponent32Float;
    case GpuBufferFormat::kBiPlanar420YpCbCr8VideoRange:
      return kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange;
    case GpuBufferFormat::kBiPlanar420YpCbCr8FullRange:
      return kCVPixelFormatType_420YpCbCr8BiPlanarFullRange;
    case GpuBufferFormat::kRGB24:
      return kCVPixelFormatType_24RGB;
    case GpuBufferFormat::kRGBAHalf64:
      return kCVPixelFormatType_64RGBAHalf;
    case GpuBufferFormat::kRGBAFloat128:
      return kCVPixelFormatType_128RGBAFloat;
    case GpuBufferFormat::kUnknown:
      return -1;
  }
  return -1;
}

inline GpuBufferFormat GpuBufferFormatForCVPixelFormat(OSType format) {
  switch (format) {
    case kCVPixelFormatType_32BGRA:
      return GpuBufferFormat::kBGRA32;
    case kCVPixelFormatType_32RGBA:
      return GpuBufferFormat::kRGBA32;
    case kCVPixelFormatType_DepthFloat32:
      return GpuBufferFormat::kGrayFloat32;
    case kCVPixelFormatType_OneComponent16Half:
      return GpuBufferFormat::kGrayHalf16;
    case kCVPixelFormatType_OneComponent32Float:
      return GpuBufferFormat::kGrayFloat32;
    case kCVPixelFormatType_OneComponent8:
      return GpuBufferFormat::kOneComponent8;
    case kCVPixelFormatType_TwoComponent8:
      return GpuBufferFormat::kTwoComponent8;
    case kCVPixelFormatType_TwoComponent16Half:
      return GpuBufferFormat::kTwoComponentHalf16;
    case kCVPixelFormatType_TwoComponent32Float:
      return GpuBufferFormat::kTwoComponentFloat32;
    case kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange:
      return GpuBufferFormat::kBiPlanar420YpCbCr8VideoRange;
    case kCVPixelFormatType_420YpCbCr8BiPlanarFullRange:
      return GpuBufferFormat::kBiPlanar420YpCbCr8FullRange;
    case kCVPixelFormatType_24RGB:
      return GpuBufferFormat::kRGB24;
    case kCVPixelFormatType_64RGBAHalf:
      return GpuBufferFormat::kRGBAHalf64;
    case kCVPixelFormatType_128RGBAFloat:
      return GpuBufferFormat::kRGBAFloat128;
  }
  return GpuBufferFormat::kUnknown;
}

#endif  // __APPLE__

}  // namespace mediapipe

#endif  // MEDIAPIPE_GPU_GPU_BUFFER_FORMAT_H_
