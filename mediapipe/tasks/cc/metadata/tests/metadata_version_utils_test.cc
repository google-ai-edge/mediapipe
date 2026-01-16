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
#include "mediapipe/tasks/cc/metadata/metadata_version_utils.h"

#include "mediapipe/framework/port/gmock.h"
#include "mediapipe/framework/port/gtest.h"

namespace mediapipe {
namespace tasks {
namespace metadata {
namespace {

TEST(MetadataVersionTest, CompareVersions) {
  ASSERT_EQ(0, CompareVersions("1.0", "1.0"));
  ASSERT_EQ(0, CompareVersions("1", "1.0.0"));

  ASSERT_EQ(-1, CompareVersions("2", "3"));
  ASSERT_EQ(-1, CompareVersions("1.2", "1.3"));
  ASSERT_EQ(-1, CompareVersions("3.2.9", "3.2.10"));
  ASSERT_EQ(-1, CompareVersions("10.1.9", "10.2"));

  ASSERT_EQ(1, CompareVersions("3", "2"));
  ASSERT_EQ(1, CompareVersions("1.3", "1.2"));
  ASSERT_EQ(1, CompareVersions("0.95", "0.94.3124"));
  ASSERT_EQ(1, CompareVersions("1.1.1.12", "1.1.1.9"));
  ASSERT_EQ(1, CompareVersions("1.1.1.12", "1.1.0.13"));
}
}  // namespace
}  // namespace metadata
}  // namespace tasks
}  // namespace mediapipe
