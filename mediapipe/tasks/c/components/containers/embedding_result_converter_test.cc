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

#include "mediapipe/tasks/c/components/containers/embedding_result_converter.h"

#include <optional>
#include <string>

#include "mediapipe/framework/port/gtest.h"
#include "mediapipe/tasks/c/components/containers/embedding_result.h"
#include "mediapipe/tasks/cc/components/containers/embedding_result.h"

namespace mediapipe::tasks::c::components::containers {

TEST(EmbeddingResultConverterTest, ConvertsEmbeddingResultCustomEmbedding) {
  mediapipe::tasks::components::containers::EmbeddingResult
      cpp_embedding_result = {
          // Initializing embeddings vector
          {// First embedding
           {
               {0.1f, 0.2f, 0.3f, 0.4f, 0.5f},  // float embedding
               {},                              // quantized embedding (empty)
               0,                               // head index
               "foo"                            // head name
           },
           // Second embedding
           {
               {},                         // float embedding (empty)
               {127, 127, 127, 127, 127},  // quantized embedding
               1,                          // head index
               std::nullopt                // no head name
           }},
          // Initializing timestamp_ms
          42  // timestamp in ms
      };

  EmbeddingResult c_embedding_result;
  CppConvertToEmbeddingResult(cpp_embedding_result, &c_embedding_result);
  EXPECT_NE(c_embedding_result.embeddings, nullptr);
  EXPECT_EQ(c_embedding_result.embeddings_count, 2);
  EXPECT_NE(c_embedding_result.embeddings[0].float_embedding, nullptr);
  EXPECT_EQ(c_embedding_result.embeddings[0].values_count, 5);
  EXPECT_EQ(c_embedding_result.embeddings[0].head_index, 0);
  EXPECT_NE(c_embedding_result.embeddings[1].quantized_embedding, nullptr);
  EXPECT_EQ(c_embedding_result.embeddings[1].values_count, 5);
  EXPECT_EQ(c_embedding_result.embeddings[1].head_index, 1);
  EXPECT_EQ(std::string(c_embedding_result.embeddings[0].head_name), "foo");
  EXPECT_EQ(c_embedding_result.timestamp_ms, 42);
  EXPECT_EQ(c_embedding_result.has_timestamp_ms, true);

  CppCloseEmbeddingResult(&c_embedding_result);
}

}  // namespace mediapipe::tasks::c::components::containers
