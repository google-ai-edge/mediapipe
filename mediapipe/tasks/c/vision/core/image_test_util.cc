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

#include "mediapipe/tasks/c/vision/core/image_test_util.h"

#include <string>

#include "absl/log/absl_check.h"
#include "mediapipe/framework/formats/image.h"
#include "mediapipe/gpu/gpu_buffer.h"
#include "mediapipe/tasks/c/core/mp_status.h"
#include "mediapipe/tasks/c/vision/core/image.h"
#include "mediapipe/tasks/c/vision/core/image_frame_util.h"

namespace mediapipe::tasks::vision::core {

ScopedMpImage GetImage(const std::string& file_name) {
  MpImagePtr image_ptr = nullptr;
  MpStatus status = MpImageCreateFromFile(file_name.c_str(), &image_ptr);
  ABSL_CHECK_EQ(status, kMpOk);
  ABSL_CHECK_NE(image_ptr, nullptr);
  return ScopedMpImage(image_ptr);
}

ScopedMpImage CreateEmptyGpuMpImage() {
  return ScopedMpImage(new MpImageInternal{.image = Image(GpuBuffer()),
                                           .cached_contiguous_data = {}});
}

}  // namespace mediapipe::tasks::vision::core
