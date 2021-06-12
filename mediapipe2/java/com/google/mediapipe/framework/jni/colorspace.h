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

#ifndef JAVA_COM_GOOGLE_MEDIAPIPE_FRAMEWORK_JNI_COLORSPACE_H_
#define JAVA_COM_GOOGLE_MEDIAPIPE_FRAMEWORK_JNI_COLORSPACE_H_

#include <cstdint>

namespace mediapipe {
namespace android {
// TODO: switch to more efficient implementation, like halide later.

// Converts an RGBA image to RGB
inline void RgbaToRgb(const uint8_t* rgba_img, int rgba_width_step, int width,
                      int height, uint8_t* rgb_img, int rgb_width_step) {
  for (int y = 0; y < height; ++y) {
    const auto* rgba = rgba_img + y * rgba_width_step;
    auto* rgb = rgb_img + y * rgb_width_step;
    for (int x = 0; x < width; ++x) {
      *rgb = *rgba;
      *(rgb + 1) = *(rgba + 1);
      *(rgb + 2) = *(rgba + 2);
      rgb += 3;
      rgba += 4;
    }
  }
}

// Converts a RGB image to RGBA
inline void RgbToRgba(const uint8_t* rgb_img, int rgb_width_step, int width,
                      int height, uint8_t* rgba_img, int rgba_width_step,
                      uint8_t alpha) {
  for (int y = 0; y < height; ++y) {
    const auto* rgb = rgb_img + y * rgb_width_step;
    auto* rgba = rgba_img + y * rgba_width_step;
    for (int x = 0; x < width; ++x) {
      *rgba = *rgb;
      *(rgba + 1) = *(rgb + 1);
      *(rgba + 2) = *(rgb + 2);
      *(rgba + 3) = alpha;
      rgb += 3;
      rgba += 4;
    }
  }
}

}  // namespace android
}  // namespace mediapipe
#endif  // JAVA_COM_GOOGLE_MEDIAPIPE_FRAMEWORK_JNI_COLORSPACE_H_
