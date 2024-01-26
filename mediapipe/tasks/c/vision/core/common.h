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

#ifndef MEDIAPIPE_TASKS_C_VISION_CORE_COMMON_H_
#define MEDIAPIPE_TASKS_C_VISION_CORE_COMMON_H_

#include <cstdint>

#ifdef __cplusplus
extern "C" {
#endif

// Supported image formats.
enum ImageFormat {
  UNKNOWN = 0,
  SRGB = 1,
  SRGBA = 2,
  GRAY8 = 3,
  SBGRA = 11  // compatible with Flutter `bgra8888` format.
};

// Supported processing modes.
enum RunningMode {
  IMAGE = 1,
  VIDEO = 2,
  LIVE_STREAM = 3,
};

// Structure to hold image frame.
struct ImageFrame {
  enum ImageFormat format;
  const uint8_t* image_buffer;
  int width;
  int height;
};

// TODO: Add GPU buffer declaration and processing logic for it.
struct GpuBuffer {
  int width;
  int height;
};

// The object to contain an image, realizes `OneOf` concept.
struct MpImage {
  enum { IMAGE_FRAME, GPU_BUFFER } type;
  union {
    struct ImageFrame image_frame;
    struct GpuBuffer gpu_buffer;
  };
};

enum MaskFormat { UINT8, FLOAT };

struct ImageFrameMask {
  enum MaskFormat mask_format;
  const uint8_t* image_buffer;
  int width;
  int height;
};

// TODO: Add GPU buffer declaration and processing logic for it.
struct GpuBufferMask {
  int width;
  int height;
};

struct MpMask {
  enum { IMAGE_FRAME, GPU_BUFFER } type;
  union {
    struct ImageFrameMask image_frame;
    struct GpuBufferMask gpu_buffer;
  };
};

#ifdef __cplusplus
}  // extern C
#endif

#endif  // MEDIAPIPE_TASKS_C_VISION_CORE_COMMON_H_
