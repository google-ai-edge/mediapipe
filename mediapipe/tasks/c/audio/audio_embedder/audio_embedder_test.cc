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

#include "mediapipe/tasks/c/audio/audio_embedder/audio_embedder.h"

#include <string>
#include <vector>

#include "absl/flags/flag.h"
#include "absl/strings/string_view.h"
#include "mediapipe/framework/deps/file_path.h"
#include "mediapipe/framework/port/gtest.h"
#include "mediapipe/tasks/c/audio/core/common.h"
#include "mediapipe/tasks/c/core/mp_status.h"

namespace {

using ::mediapipe::file::JoinPath;

std::string GetFullPath(absl::string_view file_name) {
  return JoinPath("./", kTestDataDirectory, file_name);
}

MpAudioEmbedderOptions CreateAudioEmbedderOptions(const char* model_path) {
  return {
      .base_options = {.model_asset_buffer = nullptr,
                       .model_asset_buffer_count = 0,
                       .model_asset_path = model_path},
      .embedder_options = {.l2_normalize = false, .quantize = false},
  };
}

TEST(AudioEmbedderTest, CreateSucceedsWithLiteRtLmModel) {
  const std::string model_path = GetFullPath(kTestModelPath);
  MpAudioEmbedderOptions options =
      CreateAudioEmbedderOptions(model_path.c_str());

  MpAudioEmbedderPtr embedder;
  ASSERT_EQ(MpAudioEmbedderCreate(&options, &embedder, /*error_msg=*/nullptr),
            kMpOk);
  EXPECT_EQ(MpAudioEmbedderClose(embedder, /*error_msg=*/nullptr), kMpOk);
}

}  // namespace
