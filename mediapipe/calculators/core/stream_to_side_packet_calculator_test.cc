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
#include <string>

#include "absl/memory/memory.h"
#include "mediapipe/framework/calculator_runner.h"
#include "mediapipe/framework/packet.h"
#include "mediapipe/framework/port/gmock.h"
#include "mediapipe/framework/port/gtest.h"
#include "mediapipe/framework/port/status.h"
#include "mediapipe/framework/port/status_matchers.h"
#include "mediapipe/framework/timestamp.h"

namespace mediapipe {

using ::testing::Test;

class StreamToSidePacketCalculatorTest : public Test {
 protected:
  StreamToSidePacketCalculatorTest() {
    const char kConfig[] = R"(
      calculator: "StreamToSidePacketCalculator"
      input_stream: "stream"
      output_side_packet: "side_packet"
    )";
    runner_ = absl::make_unique<CalculatorRunner>(kConfig);
  }

  std::unique_ptr<CalculatorRunner> runner_;
};

TEST_F(StreamToSidePacketCalculatorTest,
       StreamToSidePacketCalculatorWithEmptyStreamFails) {
  EXPECT_EQ(runner_->Run().code(), mediapipe::StatusCode::kUnavailable);
}

TEST_F(StreamToSidePacketCalculatorTest,
       StreamToSidePacketCalculatorWithSinglePacketCreatesSidePacket) {
  runner_->MutableInputs()->Index(0).packets.push_back(
      Adopt(new std::string("test")).At(Timestamp(1)));
  MP_ASSERT_OK(runner_->Run());
  EXPECT_EQ(runner_->OutputSidePackets().Index(0).Get<std::string>(), "test");
}

TEST_F(StreamToSidePacketCalculatorTest,
       StreamToSidePacketCalculatorWithMultiplePacketsFails) {
  runner_->MutableInputs()->Index(0).packets.push_back(
      Adopt(new std::string("test1")).At(Timestamp(1)));
  runner_->MutableInputs()->Index(0).packets.push_back(
      Adopt(new std::string("test2")).At(Timestamp(2)));
  EXPECT_EQ(runner_->Run().code(), mediapipe::StatusCode::kAlreadyExists);
}

}  // namespace mediapipe
