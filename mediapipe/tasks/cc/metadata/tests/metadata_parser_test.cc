/* Copyright 2022 The MediaPipe Authors.

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
#include "mediapipe/tasks/cc/metadata/metadata_parser.h"

#include "mediapipe/framework/port/gmock.h"
#include "mediapipe/framework/port/gtest.h"

namespace mediapipe {
namespace tasks {
namespace metadata {
namespace {

using ::testing::MatchesRegex;

TEST(MetadataParserTest, MatadataParserVersionIsWellFormed) {
  // Validates that the version is well-formed (x.y.z).
#ifdef _WIN32
  EXPECT_THAT(kMetadataParserVersion, MatchesRegex("\\d+\\.\\d+\\.\\d+"));
#else
  EXPECT_THAT(kMetadataParserVersion, MatchesRegex("[0-9]+\\.[0-9]+\\.[0-9]+"));
#endif  // _WIN32
}

}  // namespace
}  // namespace metadata
}  // namespace tasks
}  // namespace mediapipe
