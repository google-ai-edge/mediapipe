// Copyright 2025 The MediaPipe Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef MEDIAPIPE_TASKS_C_VISION_CORE_IMAGE_H_
#define MEDIAPIPE_TASKS_C_VISION_CORE_IMAGE_H_

#include <stdint.h>

#include "mediapipe/tasks/c/core/mp_status.h"

#ifndef MP_EXPORT
#define MP_EXPORT __attribute__((visibility("default")))
#endif  // MP_EXPORT

#ifdef __cplusplus
extern "C" {
#endif

// An enum describing supported raw image formats.
// Corresponds to mediapipe::ImageFormat::Format.
typedef enum MpImageFormat {
  kMpImageFormatUnknown = 0,
  kMpImageFormatSrgb = 1,
  kMpImageFormatSrgba = 2,
  kMpImageFormatGray8 = 3,
  kMpImageFormatGray16 = 4,
  kMpImageFormatSrgb48 = 7,
  kMpImageFormatSrgba64 = 8,
  kMpImageFormatVec32F1 = 9,
  kMpImageFormatVec32F2 = 12,
  kMpImageFormatVec32F4 = 13,
} MpImageFormat;

typedef struct MpImageInternal* MpImagePtr;

// Creates an MpImage from a buffer of pixel data. The buffer is copied
// into the new MpImage. The caller retains ownership of the buffer.
//
// If successful, returns MP_OK and sets `out` to the new MpImage. You must free
// the image with mp_image_free().
MP_EXPORT MpStatus MpImageCreateFromUint8Data(MpImageFormat format, int width,
                                              int height,
                                              const uint8_t* pixel_data,
                                              int pixel_data_size,
                                              MpImagePtr* out);

// Creates an MpImage from a buffer of pixel data. The buffer is copied
// into the new MpImage. The caller retains ownership of the buffer.
//
// If successful, returns MP_OK and sets `out` to the new MpImage. You must free
// the image with mp_image_free().
MP_EXPORT MpStatus MpImageCreateFromUint16Data(MpImageFormat format, int width,
                                               int height,
                                               const uint16_t* pixel_data,
                                               int pixel_data_size,
                                               MpImagePtr* out);

// Creates an MpImage from a buffer of pixel data. The buffer is copied
// into the new MpImage. The caller retains ownership of the buffer.
//
// If successful, returns MP_OK and sets `out` to the new MpImage. You must free
// the image with mp_image_free().
MP_EXPORT MpStatus MpImageCreateFromFloatData(MpImageFormat format, int width,
                                              int height,
                                              const float* pixel_data,
                                              int pixel_data_size,
                                              MpImagePtr* out);

// Creates an MpImage from an ImageFrame.
//
// The new MpImage will point to the same data as pixel data, extending the
// lifetime of the underlying ImageFrame. If the original image is on the GPU,
// the data will be transferred to the CPU first.
//
// If successful, returns MP_OK and sets `out` to the new MpImage. You must free
// the image must with mp_image_free().
MP_EXPORT MpStatus MpImageCreateFromImageFrame(MpImagePtr image,
                                               MpImagePtr* out);

// Creates an MpImage from a file.
//
// If successful, returns MP_OK and sets `out` to a new Image. You must free
// the image with mp_image_free().
MP_EXPORT MpStatus MpImageCreateFromFile(const char* file_name,
                                         MpImagePtr* out);

// Returns true if the pixel data is stored contiguously.
MP_EXPORT bool MpImageIsContiguous(MpImagePtr image);

// Returns true if the image is backed by GPU memory.
MP_EXPORT bool MpImageUsesGpu(MpImagePtr image);

// Returns true if the image is empty.
MP_EXPORT bool MpImageIsEmpty(MpImagePtr image);

// Returns true if each row of the data is aligned to alignment_boundary.
MP_EXPORT bool MpImageIsAligned(MpImagePtr image, uint32_t alignment_boundary);

// Returns the width of the image.
MP_EXPORT int MpImageGetWidth(MpImagePtr image);

// Returns the height of the image.
MP_EXPORT int MpImageGetHeight(MpImagePtr image);

// Returns the number of channels in the image.
MP_EXPORT int MpImageGetChannels(MpImagePtr image);

// Returns the byte depth of the image format. (e.g. 1 for SRGB, 2 for
// SRGB48).
MP_EXPORT int MpImageGetByteDepth(MpImagePtr image);

// Returns the width step of the image.
MP_EXPORT int MpImageGetWidthStep(MpImagePtr image);

// Returns the image format.
MP_EXPORT MpImageFormat MpImageGetFormat(MpImagePtr image);

// Sets `out` to point to the pixel data. The data is owned by the MpImage and
// the pointer is valid until mp_image_free() is called.
//
// If the image is not contiguous, the data is first copied into a contiguous
// buffer and cached internally for further access.
MP_EXPORT MpStatus MpImageDataUint8(MpImagePtr image, const uint8_t** out);

// Sets `out` to point to the pixel data. The data is owned by the MpImage and
// the pointer is valid until mp_image_free() is called.
//
// If the image is not contiguous, the data is first copied into a contiguous
// buffer and cached internally for further access.
MP_EXPORT MpStatus MpImageDataUint16(MpImagePtr image, const uint16_t** out);

// Sets `out` to point to the pixel data. The data is owned by the MpImage and
// the pointer is valid until mp_image_free() is called.
//
// If the image is not contiguous, the data is first copied into a contiguous
// buffer and cached internally for further access.
MP_EXPORT MpStatus MpImageDataFloat32(MpImagePtr image, const float** out);

// Sets `out` to the value at the given coordinate for uint8 images.
MP_EXPORT MpStatus MpImageGetValueUint8(MpImagePtr image, int* pos,
                                        int pos_size, uint8_t* out);

// Sets `out` to the value at the given coordinate for uint16 images.
MP_EXPORT MpStatus MpImageGetValueUint16(MpImagePtr image, int* pos,
                                         int pos_size, uint16_t* out);

// Sets `out` to the value at the given coordinate for float32 images.
MP_EXPORT MpStatus MpImageGetValueFloat32(MpImagePtr image, int* pos,
                                          int pos_size, float* out);

// Frees an MpImage.
MP_EXPORT void MpImageFree(MpImagePtr image);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // MEDIAPIPE_TASKS_C_VISION_CORE_IMAGE_H_
