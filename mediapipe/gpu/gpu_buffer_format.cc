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

#include "mediapipe/gpu/gpu_buffer_format.h"

#include "absl/container/flat_hash_map.h"
#include "absl/log/absl_check.h"
#include "mediapipe/framework/deps/no_destructor.h"
#include "mediapipe/framework/port/logging.h"
#if !MEDIAPIPE_DISABLE_GPU
#include "mediapipe/gpu/gl_base.h"
#endif

namespace mediapipe {

#ifndef GL_RGBA16F
#define GL_RGBA16F 34842
#endif  // GL_RGBA16F

#ifndef GL_HALF_FLOAT
#define GL_HALF_FLOAT 0x140B
#endif  // GL_HALF_FLOAT

#ifdef __EMSCRIPTEN__
#ifndef GL_HALF_FLOAT_OES
#define GL_HALF_FLOAT_OES 0x8D61
#endif  // GL_HALF_FLOAT_OES
#endif  // __EMSCRIPTEN__

#ifndef GL_RGBA8
#define GL_RGBA8 0x8058
#endif  // GL_RGBA8

#if !MEDIAPIPE_DISABLE_GPU
#ifdef GL_ES_VERSION_2_0
static void AdaptGlTextureInfoForGLES2(GlTextureInfo* info) {
  switch (info->gl_internal_format) {
    case GL_R16F:
    case GL_R32F:
      // Should this be GL_RED_EXT instead?
      info->gl_internal_format = info->gl_format = GL_LUMINANCE;
      return;
    case GL_RG16F:
    case GL_RG32F:
      // Should this be GL_RG_EXT instead?
      info->gl_internal_format = info->gl_format = GL_LUMINANCE_ALPHA;
      return;
    case GL_R8:
      info->gl_internal_format = info->gl_format = GL_RED_EXT;
      return;
    case GL_RG8:
      info->gl_internal_format = info->gl_format = GL_RG_EXT;
      return;
#ifdef __EMSCRIPTEN__
    case GL_RGBA16F:
      info->gl_internal_format = GL_RGBA;
      info->gl_type = GL_HALF_FLOAT_OES;
      return;
#endif  // __EMSCRIPTEN__
    default:
      return;
  }
}
#endif  // GL_ES_VERSION_2_0

const GlTextureInfo& GlTextureInfoForGpuBufferFormat(GpuBufferFormat format,
                                                     int plane,
                                                     GlVersion gl_version) {
  // TODO: check/add more cases using info from
  // CVPixelFormatDescriptionCreateWithPixelFormatType.
  static const mediapipe::NoDestructor<
      absl::flat_hash_map<GpuBufferFormat, std::vector<GlTextureInfo>>>
      gles3_format_info{{
          {GpuBufferFormat::kRGBA32,
           {
               {GL_RGBA, GL_RGBA, GL_UNSIGNED_BYTE, 1},
           }},
          {GpuBufferFormat::kBGRA32,
           {
  // internal_format, format, type, downscale
#if MEDIAPIPE_GPU_BUFFER_USE_CV_PIXEL_BUFFER
               // On Apple platforms, we have different code paths for iOS
               // (using CVPixelBuffer) and on macOS (using GlTextureBuffer).
               // When using CVPixelBuffer, the preferred transfer format is
               // BGRA.
               // TODO: Check iOS simulator.
               {GL_RGBA, GL_BGRA, GL_UNSIGNED_BYTE, 1},
#else
               {GL_RGBA, GL_RGBA, GL_UNSIGNED_BYTE, 1},
#endif  // MEDIAPIPE_GPU_BUFFER_USE_CV_PIXEL_BUFFER
           }},
          {GpuBufferFormat::kOneComponent8,
           {
  // This format is like RGBA grayscale: GL_LUMINANCE replicates
  // the single channel texel values to RGB channels, and set alpha
  // to 1.0. If it is desired to see only the texel values in the R
  // channel, use kOneComponent8Red instead.
#if !TARGET_OS_OSX
               {GL_LUMINANCE, GL_LUMINANCE, GL_UNSIGNED_BYTE, 1},
#else
               {GL_R8, GL_RED, GL_UNSIGNED_BYTE, 1},
#endif  // TARGET_OS_OSX
           }},
          {GpuBufferFormat::kOneComponent8Alpha,
           {
               {GL_ALPHA, GL_ALPHA, GL_UNSIGNED_BYTE, 1},
           }},
          {GpuBufferFormat::kOneComponent8Red,
           {
               {GL_R8, GL_RED, GL_UNSIGNED_BYTE, 1},
           }},
          {GpuBufferFormat::kTwoComponent8,
           {
               {GL_RG8, GL_RG, GL_UNSIGNED_BYTE, 1},
           }},
#ifdef __APPLE__
          // TODO: figure out GL_RED_EXT etc. on Android.
          {GpuBufferFormat::kBiPlanar420YpCbCr8VideoRange,
           {
               // Apple's documentation suggests GL_LUMINANCE and
               // GL_LUMINANCE_ALPHA,
               // but since they are deprecated in later versions of OpenGL, we
               // use
               // GL_RED and GL_RG. On GLES2 we can use GL_RED_EXT and GL_RG_EXT
               // instead, though we are not sure if it may cause compatibility
               // problems
               // with very old devices.
               {GL_R8, GL_RED, GL_UNSIGNED_BYTE, 1},
               {GL_RG8, GL_RG, GL_UNSIGNED_BYTE, 2},
           }},
          {GpuBufferFormat::kBiPlanar420YpCbCr8FullRange,
           {
               {GL_R8, GL_RED, GL_UNSIGNED_BYTE, 1},
               {GL_RG8, GL_RG, GL_UNSIGNED_BYTE, 2},
           }},
#endif  // __APPLE__
          {GpuBufferFormat::kTwoComponentHalf16,
           {
               // TODO: use GL_HALF_FLOAT_OES on GLES2?
               {GL_RG16F, GL_RG, GL_HALF_FLOAT, 1},
           }},
          {GpuBufferFormat::kTwoComponentFloat32,
           {
               {GL_RG32F, GL_RG, GL_FLOAT, 1},
           }},
          {GpuBufferFormat::kGrayHalf16,
           {
               {GL_R16F, GL_RED, GL_HALF_FLOAT, 1},
           }},
          {GpuBufferFormat::kGrayFloat32,
           {
               {GL_R32F, GL_RED, GL_FLOAT, 1},
           }},
          {GpuBufferFormat::kRGB24,
           {
               {GL_RGB, GL_RGB, GL_UNSIGNED_BYTE, 1},
           }},
          {GpuBufferFormat::kRGBAHalf64,
           {
               {GL_RGBA16F, GL_RGBA, GL_HALF_FLOAT, 1},
           }},
          {GpuBufferFormat::kRGBAFloat128,
           {
               {GL_RGBA32F, GL_RGBA, GL_FLOAT, 1},
           }},
          {GpuBufferFormat::kImmutableRGBAFloat128,
           {
               {GL_RGBA32F, GL_RGBA, GL_FLOAT, 1, true /* immutable */},
           }},
          {GpuBufferFormat::kImmutableRGBA32,
           {
               {GL_RGBA8, GL_RGBA, GL_UNSIGNED_BYTE, 1, true /* immutable */},
           }},
      }};

  static const auto* gles2_format_info = ([] {
    auto formats =
        new absl::flat_hash_map<GpuBufferFormat, std::vector<GlTextureInfo>>(
            *gles3_format_info);
#ifdef GL_ES_VERSION_2_0
    for (auto& format_planes : *formats) {
      for (auto& info : format_planes.second) {
        AdaptGlTextureInfoForGLES2(&info);
      }
    }
#endif  // GL_ES_VERSION_2_0
    return formats;
  })();

  auto* format_info = gles3_format_info.get();
  switch (gl_version) {
    case GlVersion::kGLES2:
      format_info = gles2_format_info;
      break;
    case GlVersion::kGLES3:
    case GlVersion::kGL:
      break;
  }

  auto iter = format_info->find(format);
  ABSL_CHECK(iter != format_info->end())
      << "unsupported format: "
      << static_cast<std::underlying_type_t<decltype(format)>>(format);
  const auto& planes = iter->second;
#ifndef __APPLE__
  ABSL_CHECK_EQ(planes.size(), 1)
      << "multiplanar formats are not supported on this platform";
#endif
  ABSL_CHECK_GE(plane, 0) << "invalid plane number";
  ABSL_CHECK_LT(plane, planes.size()) << "invalid plane number";
  return planes[plane];
}
#endif  // MEDIAPIPE_DISABLE_GPU

ImageFormat::Format ImageFormatForGpuBufferFormat(GpuBufferFormat format) {
  switch (format) {
    case GpuBufferFormat::kImmutableRGBA32:
    case GpuBufferFormat::kBGRA32:
      // TODO: verify we are handling order of channels correctly.
      return ImageFormat::SRGBA;
    case GpuBufferFormat::kGrayFloat32:
      return ImageFormat::VEC32F1;
    case GpuBufferFormat::kOneComponent8:
      return ImageFormat::GRAY8;
    case GpuBufferFormat::kBiPlanar420YpCbCr8VideoRange:
    case GpuBufferFormat::kBiPlanar420YpCbCr8FullRange:
      // TODO: should either of these be YCBCR420P10?
      return ImageFormat::YCBCR420P;
    case GpuBufferFormat::kRGB24:
      return ImageFormat::SRGB;
    case GpuBufferFormat::kTwoComponentFloat32:
      return ImageFormat::VEC32F2;
    case GpuBufferFormat::kImmutableRGBAFloat128:
    case GpuBufferFormat::kRGBAFloat128:
      return ImageFormat::VEC32F4;
    case GpuBufferFormat::kRGBA32:
      return ImageFormat::SRGBA;
    case GpuBufferFormat::kGrayHalf16:
    case GpuBufferFormat::kOneComponent8Alpha:
    case GpuBufferFormat::kOneComponent8Red:
    case GpuBufferFormat::kTwoComponent8:
    case GpuBufferFormat::kTwoComponentHalf16:
    case GpuBufferFormat::kRGBAHalf64:
    case GpuBufferFormat::kNV12:
    case GpuBufferFormat::kNV21:
    case GpuBufferFormat::kI420:
    case GpuBufferFormat::kYV12:
    case GpuBufferFormat::kUnknown:
      return ImageFormat::UNKNOWN;
  }
}

GpuBufferFormat GpuBufferFormatForImageFormat(ImageFormat::Format format) {
  switch (format) {
    case ImageFormat::SRGB:
      return GpuBufferFormat::kRGB24;
    case ImageFormat::SRGBA:
      // TODO: verify we are handling order of channels correctly.
      return GpuBufferFormat::kBGRA32;
    case ImageFormat::VEC32F1:
      return GpuBufferFormat::kGrayFloat32;
    case ImageFormat::VEC32F2:
      return GpuBufferFormat::kTwoComponentFloat32;
    case ImageFormat::VEC32F4:
      return GpuBufferFormat::kRGBAFloat128;
    case ImageFormat::GRAY8:
      return GpuBufferFormat::kOneComponent8;
    case ImageFormat::YCBCR420P:
      // TODO: or video range?
      return GpuBufferFormat::kBiPlanar420YpCbCr8FullRange;
    case ImageFormat::UNKNOWN:
    default:
      return GpuBufferFormat::kUnknown;
  }
}

}  // namespace mediapipe
