// Copyright 2023 The MediaPipe Authors.
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

#include "mediapipe/util/graph_builder_utils.h"

#include "mediapipe/framework/calculator.pb.h"
#include "mediapipe/framework/port/gmock.h"
#include "mediapipe/framework/port/gtest.h"

namespace mediapipe {
namespace {

TEST(GraphUtils, HasInput) {
  CalculatorGraphConfig::Node node;
  node.add_input_stream("SOME_TAG:some_name");

  EXPECT_TRUE(HasInput(node, "SOME_TAG"));
  EXPECT_FALSE(HasInput(node, "SOME"));
}

TEST(GraphUtils, HasSideInput) {
  CalculatorGraphConfig::Node node;
  node.add_input_side_packet("SOME_TAG:some_name");

  EXPECT_TRUE(HasSideInput(node, "SOME_TAG"));
  EXPECT_FALSE(HasSideInput(node, "SOME"));
}

TEST(GraphUtils, HasOutput) {
  CalculatorGraphConfig::Node node;
  node.add_output_stream("SOME_TAG:some_name");

  EXPECT_TRUE(HasOutput(node, "SOME_TAG"));
  EXPECT_FALSE(HasOutput(node, "SOME"));
}

}  // namespace
}  // namespace mediapipe
