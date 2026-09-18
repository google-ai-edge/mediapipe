// Copyright 2026 The MediaPipe Authors.
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

#include "mediapipe/tasks/cc/audio/audio_embedder/audio_embedder.h"

#include <memory>
#include <string>
#include <utility>

#include "absl/flags/flag.h"
#include "absl/status/status.h"
#include "mediapipe/framework/deps/file_path.h"
#include "mediapipe/framework/formats/matrix.h"
#include "mediapipe/framework/port/gmock.h"
#include "mediapipe/framework/port/gtest.h"

namespace mediapipe {
namespace tasks {
namespace audio {
namespace audio_embedder {

namespace {

using ::mediapipe::file::JoinPath;

TEST(AudioEmbedderTest, FailsWithMissingModel) {
  auto options = std::make_unique<AudioEmbedderOptions>();
  auto audio_embedder = AudioEmbedder::Create(std::move(options));
  EXPECT_EQ(audio_embedder.status().code(),
            ::absl::StatusCode::kFailedPrecondition);
}

}  // namespace
}  // namespace audio_embedder
}  // namespace audio
}  // namespace tasks
}  // namespace mediapipe
