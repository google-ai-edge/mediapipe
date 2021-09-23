// Copyright 2021 The MediaPipe Authors.
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

#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/calculator_runner.h"
#include "mediapipe/framework/port/canonical_errors.h"
#include "mediapipe/framework/port/gmock.h"
#include "mediapipe/framework/port/gtest.h"
#include "mediapipe/framework/port/status.h"
#include "mediapipe/framework/port/status_matchers.h"
#include "mediapipe/framework/timestamp.h"
#include "mediapipe/framework/tool/validate_type.h"
#include "mediapipe/util/packet_test_util.h"
#include "mediapipe/util/time_series_test_util.h"

namespace mediapipe {

class MakePairCalculatorTest
    : public mediapipe::TimeSeriesCalculatorTest<mediapipe::NoOptions> {
 protected:
  void SetUp() override {
    calculator_name_ = "MakePairCalculator";
    num_input_streams_ = 2;
  }
};

TEST_F(MakePairCalculatorTest, ProducesExpectedPairs) {
  InitializeGraph();
  AppendInputPacket(new std::string("first packet"), Timestamp(1),
                    /* input_index= */ 0);
  AppendInputPacket(new std::string("second packet"), Timestamp(5),
                    /* input_index= */ 0);
  AppendInputPacket(new int(10), Timestamp(1), /* input_index= */ 1);
  AppendInputPacket(new int(20), Timestamp(5), /* input_index= */ 1);

  MP_ASSERT_OK(RunGraph());

  EXPECT_THAT(
      output().packets,
      ::testing::ElementsAre(
          mediapipe::PacketContainsTimestampAndPayload<
              std::pair<Packet, Packet>>(
              Timestamp(1),
              ::testing::Pair(
                  mediapipe::PacketContainsTimestampAndPayload<std::string>(
                      Timestamp(1), std::string("first packet")),
                  mediapipe::PacketContainsTimestampAndPayload<int>(
                      Timestamp(1), 10))),
          mediapipe::PacketContainsTimestampAndPayload<
              std::pair<Packet, Packet>>(
              Timestamp(5),
              ::testing::Pair(
                  mediapipe::PacketContainsTimestampAndPayload<std::string>(
                      Timestamp(5), std::string("second packet")),
                  mediapipe::PacketContainsTimestampAndPayload<int>(
                      Timestamp(5), 20)))));
}

}  // namespace mediapipe
