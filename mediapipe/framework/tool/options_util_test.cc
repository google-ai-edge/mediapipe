// Copyright 2019 The MediaPipe Authors.
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

#include <memory>
#include <vector>

#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/port/gtest.h"
#include "mediapipe/framework/port/parse_text_proto.h"
#include "mediapipe/framework/port/status_matchers.h"
#include "mediapipe/framework/testdata/night_light_calculator.pb.h"
#include "mediapipe/framework/tool/options_registry.h"

namespace mediapipe {
namespace {

// Tests for calculator and graph options.
//
class OptionsUtilTest : public ::testing::Test {
 protected:
  void SetUp() override {}
  void TearDown() override {}
};

// Retrieves the description of a protobuf.
TEST_F(OptionsUtilTest, GetProtobufDescriptor) {
  const proto_ns::Descriptor* descriptor =
      tool::GetProtobufDescriptor("mediapipe.CalculatorGraphConfig");
#ifndef MEDIAPIPE_MOBILE
  EXPECT_NE(nullptr, descriptor);
#else
  EXPECT_EQ(nullptr, descriptor);
#endif
}

// Retrieves the description of a protobuf from the OptionsRegistry.
TEST_F(OptionsUtilTest, GetProtobufDescriptorRegistered) {
  const proto_ns::Descriptor* descriptor =
      tool::OptionsRegistry::GetProtobufDescriptor(
          "mediapipe.NightLightCalculatorOptions");
  EXPECT_NE(nullptr, descriptor);
}

}  // namespace
}  // namespace mediapipe
