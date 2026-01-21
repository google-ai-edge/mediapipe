// Copyright 2025 The MediaPipe Authors.
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

#ifndef MEDIAPIPE_TASKS_C_VISION_CORE_IMAGE_TEST_UTIL_H_
#define MEDIAPIPE_TASKS_C_VISION_CORE_IMAGE_TEST_UTIL_H_

#include <memory>
#include <string>

#include "mediapipe/tasks/c/vision/core/image.h"

namespace mediapipe::tasks::vision::core {

struct MpImageDeleter {
  void operator()(MpImagePtr image) const {
    if (image) {
      MpImageFree(image);
    }
  }
};

using ScopedMpImage = std::unique_ptr<MpImageInternal, MpImageDeleter>;

ScopedMpImage GetImage(const std::string& file_name);

ScopedMpImage CreateEmptyGpuMpImage();

}  // namespace mediapipe::tasks::vision::core

#endif  // MEDIAPIPE_TASKS_C_VISION_CORE_IMAGE_TEST_UTIL_H_
