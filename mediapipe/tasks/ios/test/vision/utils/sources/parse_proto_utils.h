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

#ifndef MEDIAPIPE_TASKS_IOS_TEST_VISION_UTILS_H_
#define MEDIAPIPE_TASKS_IOS_TEST_VISION_UTILS_H_

#include <string>

#include "absl/status/status.h"
#include "google/protobuf/message.h"

namespace mediapipe::tasks::ios::test::vision::utils {
absl::Status get_proto_from_pbtxt(std::string file_path,
                                  ::google::protobuf::Message& proto);
}  // namespace mediapipe::tasks::ios::test::vision::utils

#endif  // MEDIAPIPE_TASKS_IOS_TEST_VISION_UTILS_H_
