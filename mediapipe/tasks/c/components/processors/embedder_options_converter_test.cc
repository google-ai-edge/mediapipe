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

#include "mediapipe/tasks/c/components/processors/embedder_options_converter.h"

#include "mediapipe/framework/port/gtest.h"
#include "mediapipe/tasks/c/components/processors/embedder_options.h"
#include "mediapipe/tasks/cc/components/processors/embedder_options.h"

namespace mediapipe::tasks::c::components::processors {

TEST(EmbedderOptionsConverterTest, ConvertsEmbedderOptionsCustomValues) {
  EmbedderOptions c_embedder_options = {/* l2_normalize= */ true,
                                        /* quantize= */ false};

  mediapipe::tasks::components::processors::EmbedderOptions
      cpp_embedder_options = {};

  CppConvertToEmbedderOptions(c_embedder_options, &cpp_embedder_options);
  EXPECT_EQ(cpp_embedder_options.l2_normalize, true);
  EXPECT_EQ(cpp_embedder_options.quantize, false);
}

}  // namespace mediapipe::tasks::c::components::processors
