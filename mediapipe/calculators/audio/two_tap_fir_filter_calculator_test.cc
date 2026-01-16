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
#include <utility>

#include "Eigen/Core"
#include "absl/strings/substitute.h"
#include "mediapipe/calculators/audio/two_tap_fir_filter_calculator.pb.h"
#include "mediapipe/framework/calculator.pb.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/calculator_runner.h"
#include "mediapipe/framework/formats/matrix.h"
#include "mediapipe/framework/formats/time_series_header.pb.h"
#include "mediapipe/framework/output_stream_poller.h"
#include "mediapipe/framework/port/gmock.h"
#include "mediapipe/framework/port/gtest.h"
#include "mediapipe/framework/port/parse_text_proto.h"
#include "mediapipe/framework/port/status_matchers.h"

namespace mediapipe {
namespace {

using ::mediapipe::Matrix;
using ::mediapipe::ParseTextProtoOrDie;

// Create test graph config
CalculatorGraphConfig CreateTestGraphConfig(float gain_now, float gain_prev) {
  const std::string graph_pbtxt = absl::Substitute(
      R"pb(
        input_stream: "input"
        output_stream: "output"
        node {
          calculator: "TwoTapFirFilterCalculator"
          input_stream: "INPUT:input"
          output_stream: "OUTPUT:output"
          node_options {
            [type.googleapis.com/mediapipe.TwoTapFirFilterCalculatorOptions] {
              gain_now: $0
              gain_prev: $1
            }
          }
        }
      )pb",
      gain_now, gain_prev);
  return ParseTextProtoOrDie<CalculatorGraphConfig>(graph_pbtxt);
  ;
}

// Generates a multi-channel input packet with an impulse at the first sample.
Packet GenerateImpulseInputPacket(int packet_size_samples, int num_channels) {
  auto impulse =
      std::make_unique<Matrix>(Matrix::Zero(num_channels, packet_size_samples));
  for (int i = 0; i < num_channels; ++i) {
    (*impulse)(i, 0) = 1.0f;
  }
  return mediapipe::MakePacket<Matrix>(*std::move(impulse));
}

TEST(TwoTapFirFilterCalculatorTest, ShoudldKeepImpulse) {
  const CalculatorGraphConfig config = CreateTestGraphConfig(
      /*gain_now=*/1.0f, /*gain_prev=*/0.0f);

  CalculatorGraph graph;
  MP_EXPECT_OK(graph.Initialize(config));

  auto statusOrPoller = graph.AddOutputStreamPoller("output");
  MP_EXPECT_OK(statusOrPoller.status());
  OutputStreamPoller& poller = statusOrPoller.value();

  std::unique_ptr<mediapipe::TimeSeriesHeader> header(
      new mediapipe::TimeSeriesHeader());
  header->set_sample_rate(48000.0f);
  header->set_num_channels(2);
  MP_ASSERT_OK(graph.StartRun({}, {{"input", Adopt(header.release())}}));
  MP_EXPECT_OK(graph.AddPacketToInputStream(
      "input", GenerateImpulseInputPacket(/*packet_size_samples=*/128,
                                          /*num_channels=*/2)
                   .At(Timestamp(0))));

  MP_EXPECT_OK(graph.CloseAllInputStreams());
  Packet output_packet;
  ASSERT_TRUE(poller.Next(&output_packet));
  MP_EXPECT_OK(graph.WaitUntilDone());

  Matrix output = output_packet.Get<Matrix>();
  EXPECT_EQ(output.rows(), 2);
  EXPECT_EQ(output.cols(), 128);
  EXPECT_FLOAT_EQ(output(0, 0), 1.0f);
  EXPECT_FLOAT_EQ(output(1, 0), 1.0f);
}

TEST(TwoTapFirFilterCalculatorTest, ShoudldDelayImpulseByOneSample) {
  const CalculatorGraphConfig config = CreateTestGraphConfig(
      /*gain_now=*/0.0f, /*gain_prev=*/1.0f);

  CalculatorGraph graph;
  MP_EXPECT_OK(graph.Initialize(config));

  auto statusOrPoller = graph.AddOutputStreamPoller("output");
  MP_EXPECT_OK(statusOrPoller.status());
  OutputStreamPoller& poller = statusOrPoller.value();

  std::unique_ptr<mediapipe::TimeSeriesHeader> header(
      new mediapipe::TimeSeriesHeader());
  header->set_sample_rate(48000.0f);
  header->set_num_channels(2);
  MP_ASSERT_OK(graph.StartRun({}, {{"input", Adopt(header.release())}}));
  MP_EXPECT_OK(graph.AddPacketToInputStream(
      "input", GenerateImpulseInputPacket(/*packet_size_samples=*/128,
                                          /*num_channels=*/2)
                   .At(Timestamp(0))));

  MP_EXPECT_OK(graph.CloseAllInputStreams());
  Packet output_packet;
  ASSERT_TRUE(poller.Next(&output_packet));
  MP_EXPECT_OK(graph.WaitUntilDone());

  Matrix output = output_packet.Get<Matrix>();
  EXPECT_EQ(output.rows(), 2);
  EXPECT_EQ(output.cols(), 128);
  EXPECT_FLOAT_EQ(output(0, 1), 1.0f);
  EXPECT_FLOAT_EQ(output(1, 1), 1.0f);
}

}  // anonymous namespace
}  // namespace mediapipe
