/* Copyright 2026 The MediaPipe Authors. All Rights Reserved.

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

#include "mediapipe/tasks/cc/retrieval/semantic_retriever/image_decoder.h"

#include <fstream>
#include <ios>
#include <string>

#include "absl/status/status.h"
#include "mediapipe/framework/deps/file_path.h"
#include "mediapipe/framework/port/gtest.h"
#include "mediapipe/framework/port/status_matchers.h"

namespace mediapipe::tasks::retrieval {
namespace {

using ::mediapipe::file::JoinPath;

TEST(ImageDecoderTest, TestDecodeImageBytesSucceeds) {
  std::string file_path = JoinPath(testing::TempDir(), "test_image.png");

  // Write a dummy image file
  std::ofstream out(file_path, std::ios::binary);
  out << "test_bytes";
  out.close();

  MP_ASSERT_OK_AND_ASSIGN(auto bytes,
                          ImageDecoder::DecodeImageBytes(file_path));
  EXPECT_EQ(bytes, "test_bytes");
}

TEST(ImageDecoderTest, TestDecodeImageBytesFileNotFound) {
  std::string file_path = JoinPath(testing::TempDir(), "missing_file_xyz.png");
  auto bytes_or = ImageDecoder::DecodeImageBytes(file_path);
  EXPECT_EQ(bytes_or.status().code(), absl::StatusCode::kNotFound);
}

}  // namespace
}  // namespace mediapipe::tasks::retrieval
