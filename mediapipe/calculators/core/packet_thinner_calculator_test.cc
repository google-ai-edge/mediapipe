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

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "absl/log/absl_check.h"
#include "absl/strings/str_cat.h"
#include "mediapipe/calculators/core/packet_thinner_calculator.pb.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/calculator_runner.h"
#include "mediapipe/framework/formats/video_stream_header.h"
#include "mediapipe/framework/port/gmock.h"
#include "mediapipe/framework/port/gtest.h"
#include "mediapipe/framework/port/status_matchers.h"

namespace mediapipe {
namespace {

constexpr char kPeriodTag[] = "PERIOD";

// A simple version of CalculatorRunner with built-in convenience methods for
// setting inputs from a vector and checking outputs against a vector of
// expected outputs.
class SimpleRunner : public CalculatorRunner {
 public:
  explicit SimpleRunner(const CalculatorOptions& options)
      : CalculatorRunner("PacketThinnerCalculator", options) {
    SetNumInputs(1);
    SetNumOutputs(1);
    SetNumInputSidePackets(0);
  }

  explicit SimpleRunner(const CalculatorGraphConfig::Node& node)
      : CalculatorRunner(node) {}

  void SetInput(const std::vector<int>& timestamp_list) {
    MutableInputs()->Index(0).packets.clear();
    for (const int ts : timestamp_list) {
      MutableInputs()->Index(0).packets.push_back(
          MakePacket<std::string>(absl::StrCat("Frame #", ts))
              .At(Timestamp(ts)));
    }
  }

  void SetFrameRate(const double frame_rate) {
    auto video_header = absl::make_unique<VideoHeader>();
    video_header->frame_rate = frame_rate;
    MutableInputs()->Index(0).header = Adopt(video_header.release());
  }

  std::vector<int64_t> GetOutputTimestamps() const {
    std::vector<int64_t> timestamps;
    for (const Packet& packet : Outputs().Index(0).packets) {
      timestamps.emplace_back(packet.Timestamp().Value());
    }
    return timestamps;
  }

  double GetFrameRate() const {
    ABSL_CHECK(!Outputs().Index(0).header.IsEmpty());
    return Outputs().Index(0).header.Get<VideoHeader>().frame_rate;
  }
};

// Check that thinner respects start_time and end_time options.
// We only test with one thinner because the logic for start & end time
// handling is shared across both types of thinner in Process().
TEST(PacketThinnerCalculatorTest, StartAndEndTimeTest) {
  CalculatorOptions options;
  auto* extension =
      options.MutableExtension(PacketThinnerCalculatorOptions::ext);
  extension->set_thinner_type(PacketThinnerCalculatorOptions::ASYNC);
  extension->set_period(5);
  extension->set_start_time(4);
  extension->set_end_time(12);
  SimpleRunner runner(options);
  runner.SetInput({2, 3, 5, 7, 11, 13, 17, 19, 23, 29});
  MP_ASSERT_OK(runner.Run());

  const std::vector<int64_t> expected_timestamps = {5, 11};
  EXPECT_EQ(expected_timestamps, runner.GetOutputTimestamps());
}

TEST(PacketThinnerCalculatorTest, AsyncUniformStreamThinningTest) {
  CalculatorOptions options;
  auto* extension =
      options.MutableExtension(PacketThinnerCalculatorOptions::ext);
  extension->set_thinner_type(PacketThinnerCalculatorOptions::ASYNC);
  extension->set_period(5);
  SimpleRunner runner(options);
  runner.SetInput({2, 4, 6, 8, 10, 12, 14});
  MP_ASSERT_OK(runner.Run());

  const std::vector<int64_t> expected_timestamps = {2, 8, 14};
  EXPECT_EQ(expected_timestamps, runner.GetOutputTimestamps());
}

TEST(PacketThinnerCalculatorTest, ASyncUniformStreamThinningTestBySidePacket) {
  // Note: sync runner but outputting *original* timestamps.
  CalculatorGraphConfig::Node node;
  node.set_calculator("PacketThinnerCalculator");
  node.add_input_side_packet("PERIOD:period");
  node.add_input_stream("input_stream");
  node.add_output_stream("output_stream");
  auto* extension = node.mutable_options()->MutableExtension(
      PacketThinnerCalculatorOptions::ext);
  extension->set_thinner_type(PacketThinnerCalculatorOptions::ASYNC);
  extension->set_start_time(0);
  extension->set_sync_output_timestamps(false);

  SimpleRunner runner(node);
  runner.SetInput({2, 4, 6, 8, 10, 12, 14});
  runner.MutableSidePackets()->Tag(kPeriodTag) = MakePacket<int64_t>(5);
  MP_ASSERT_OK(runner.Run());

  const std::vector<int64_t> expected_timestamps = {2, 8, 14};
  EXPECT_EQ(expected_timestamps, runner.GetOutputTimestamps());
}

TEST(PacketThinnerCalculatorTest, SyncUniformStreamThinningTest1) {
  // Note: sync runner but outputting *original* timestamps.
  CalculatorOptions options;
  auto* extension =
      options.MutableExtension(PacketThinnerCalculatorOptions::ext);
  extension->set_thinner_type(PacketThinnerCalculatorOptions::SYNC);
  extension->set_start_time(0);
  extension->set_period(5);
  extension->set_sync_output_timestamps(false);
  SimpleRunner runner(options);
  runner.SetInput({2, 4, 6, 8, 10, 12, 14});
  MP_ASSERT_OK(runner.Run());

  const std::vector<int64_t> expected_timestamps = {2, 6, 10, 14};
  EXPECT_EQ(expected_timestamps, runner.GetOutputTimestamps());
}

TEST(PacketThinnerCalculatorTest, SyncUniformStreamThinningTestBySidePacket1) {
  // Note: sync runner but outputting *original* timestamps.
  CalculatorGraphConfig::Node node;
  node.set_calculator("PacketThinnerCalculator");
  node.add_input_side_packet("PERIOD:period");
  node.add_input_stream("input_stream");
  node.add_output_stream("output_stream");
  auto* extension = node.mutable_options()->MutableExtension(
      PacketThinnerCalculatorOptions::ext);
  extension->set_thinner_type(PacketThinnerCalculatorOptions::SYNC);
  extension->set_start_time(0);
  extension->set_sync_output_timestamps(false);

  SimpleRunner runner(node);
  runner.SetInput({2, 4, 6, 8, 10, 12, 14});
  runner.MutableSidePackets()->Tag(kPeriodTag) = MakePacket<int64_t>(5);
  MP_ASSERT_OK(runner.Run());

  const std::vector<int64_t> expected_timestamps = {2, 6, 10, 14};
  EXPECT_EQ(expected_timestamps, runner.GetOutputTimestamps());
}

TEST(PacketThinnerCalculatorTest, SyncUniformStreamThinningTest2) {
  // Same test but now with synced timestamps.
  CalculatorOptions options;
  auto* extension =
      options.MutableExtension(PacketThinnerCalculatorOptions::ext);
  extension->set_thinner_type(PacketThinnerCalculatorOptions::SYNC);
  extension->set_start_time(0);
  extension->set_period(5);
  extension->set_sync_output_timestamps(true);
  SimpleRunner runner(options);
  runner.SetInput({2, 4, 6, 8, 10, 12, 14});
  MP_ASSERT_OK(runner.Run());

  const std::vector<int64_t> expected_timestamps = {0, 5, 10, 15};
  EXPECT_EQ(expected_timestamps, runner.GetOutputTimestamps());
}

// Test: Given a stream with timestamps corresponding to first ten prime numbers
// and period of 5, confirm whether timestamps of thinner stream matches
// expectations.
TEST(PacketThinnerCalculatorTest, PrimeStreamThinningTest1) {
  // ASYNC thinner.
  CalculatorOptions options;
  auto* extension =
      options.MutableExtension(PacketThinnerCalculatorOptions::ext);
  extension->set_thinner_type(PacketThinnerCalculatorOptions::ASYNC);
  extension->set_period(5);
  SimpleRunner runner(options);
  runner.SetInput({2, 3, 5, 7, 11, 13, 17, 19, 23, 29});
  MP_ASSERT_OK(runner.Run());

  const std::vector<int64_t> expected_timestamps = {2, 7, 13, 19, 29};
  EXPECT_EQ(expected_timestamps, runner.GetOutputTimestamps());
}

TEST(PacketThinnerCalculatorTest, PrimeStreamThinningTest2) {
  // SYNC with original timestamps.
  CalculatorOptions options;
  auto* extension =
      options.MutableExtension(PacketThinnerCalculatorOptions::ext);
  extension->set_thinner_type(PacketThinnerCalculatorOptions::SYNC);
  extension->set_start_time(0);
  extension->set_period(5);
  extension->set_sync_output_timestamps(false);
  SimpleRunner runner(options);
  runner.SetInput({2, 3, 5, 7, 11, 13, 17, 19, 23, 29});
  MP_ASSERT_OK(runner.Run());

  const std::vector<int64_t> expected_timestamps = {2, 5, 11, 17, 19, 23, 29};
  EXPECT_EQ(expected_timestamps, runner.GetOutputTimestamps());
}

// Confirm that Calculator correctly handles boundary cases.
TEST(PacketThinnerCalculatorTest, BoundaryTimestampTest1) {
  // Odd period, negative start_time
  CalculatorOptions options;
  auto* extension =
      options.MutableExtension(PacketThinnerCalculatorOptions::ext);
  extension->set_thinner_type(PacketThinnerCalculatorOptions::SYNC);
  extension->set_start_time(-10);
  extension->set_period(5);
  extension->set_sync_output_timestamps(true);
  SimpleRunner runner(options);
  // Two timestamps falling on either side of a period boundary.
  runner.SetInput({2, 3});
  MP_ASSERT_OK(runner.Run());

  const std::vector<int64_t> expected_timestamps = {0, 5};
  EXPECT_EQ(expected_timestamps, runner.GetOutputTimestamps());
}

TEST(PacketThinnerCalculatorTest, BoundaryTimestampTest2) {
  // Even period, negative start_time, negative packet timestamps.
  CalculatorOptions options;
  auto* extension =
      options.MutableExtension(PacketThinnerCalculatorOptions::ext);
  extension->set_thinner_type(PacketThinnerCalculatorOptions::SYNC);
  extension->set_start_time(-144);
  extension->set_period(6);
  extension->set_sync_output_timestamps(true);
  SimpleRunner runner(options);
  // Two timestamps falling on either side of a period boundary.
  runner.SetInput({-4, -3, 8, 9});
  MP_ASSERT_OK(runner.Run());

  const std::vector<int64_t> expected_timestamps = {-6, 0, 6, 12};
  EXPECT_EQ(expected_timestamps, runner.GetOutputTimestamps());
}

TEST(PacketThinnerCalculatorTest, FrameRateTest1) {
  CalculatorOptions options;
  auto* extension =
      options.MutableExtension(PacketThinnerCalculatorOptions::ext);
  extension->set_thinner_type(PacketThinnerCalculatorOptions::ASYNC);
  extension->set_period(5);
  extension->set_update_frame_rate(true);
  SimpleRunner runner(options);
  runner.SetInput({2, 4, 6, 8, 10, 12, 14});
  runner.SetFrameRate(1000000.0 / 2);
  MP_ASSERT_OK(runner.Run());

  const std::vector<int64_t> expected_timestamps = {2, 8, 14};
  EXPECT_EQ(expected_timestamps, runner.GetOutputTimestamps());
  // The true sampling period is 6.
  EXPECT_DOUBLE_EQ(1000000.0 / 6, runner.GetFrameRate());
}

TEST(PacketThinnerCalculatorTest, FrameRateTest2) {
  CalculatorOptions options;
  auto* extension =
      options.MutableExtension(PacketThinnerCalculatorOptions::ext);
  extension->set_thinner_type(PacketThinnerCalculatorOptions::ASYNC);
  extension->set_period(5);
  extension->set_update_frame_rate(true);
  SimpleRunner runner(options);
  runner.SetInput({8, 16, 24, 32, 40, 48, 56});
  runner.SetFrameRate(1000000.0 / 8);
  MP_ASSERT_OK(runner.Run());
  const std::vector<int64_t> expected_timestamps = {8, 16, 24, 32, 40, 48, 56};
  EXPECT_EQ(expected_timestamps, runner.GetOutputTimestamps());
  // The true sampling period is still 8.
  EXPECT_DOUBLE_EQ(1000000.0 / 8, runner.GetFrameRate());
}

TEST(PacketThinnerCalculatorTest, FrameRateTest3) {
  // Note: sync runner but outputting *original* timestamps.
  CalculatorOptions options;
  auto* extension =
      options.MutableExtension(PacketThinnerCalculatorOptions::ext);
  extension->set_thinner_type(PacketThinnerCalculatorOptions::SYNC);
  extension->set_start_time(0);
  extension->set_period(5);
  extension->set_sync_output_timestamps(false);
  extension->set_update_frame_rate(true);
  SimpleRunner runner(options);
  runner.SetInput({2, 4, 6, 8, 10, 12, 14});
  runner.SetFrameRate(1000000.0 / 2);
  MP_ASSERT_OK(runner.Run());

  const std::vector<int64_t> expected_timestamps = {2, 6, 10, 14};
  EXPECT_EQ(expected_timestamps, runner.GetOutputTimestamps());
  // The true (long-run) sampling period is 5.
  EXPECT_DOUBLE_EQ(1000000.0 / 5, runner.GetFrameRate());
}

TEST(PacketThinnerCalculatorTest, FrameRateTest4) {
  // Same test but now with synced timestamps.
  CalculatorOptions options;
  auto* extension =
      options.MutableExtension(PacketThinnerCalculatorOptions::ext);
  extension->set_thinner_type(PacketThinnerCalculatorOptions::SYNC);
  extension->set_start_time(0);
  extension->set_period(5);
  extension->set_sync_output_timestamps(true);
  extension->set_update_frame_rate(true);
  SimpleRunner runner(options);
  runner.SetInput({2, 4, 6, 8, 10, 12, 14});
  runner.SetFrameRate(1000000.0 / 2);
  MP_ASSERT_OK(runner.Run());

  const std::vector<int64_t> expected_timestamps = {0, 5, 10, 15};
  EXPECT_EQ(expected_timestamps, runner.GetOutputTimestamps());
  // The true (long-run) sampling period is 5.
  EXPECT_DOUBLE_EQ(1000000.0 / 5, runner.GetFrameRate());
}

TEST(PacketThinnerCalculatorTest, FrameRateTest5) {
  CalculatorOptions options;
  auto* extension =
      options.MutableExtension(PacketThinnerCalculatorOptions::ext);
  extension->set_thinner_type(PacketThinnerCalculatorOptions::SYNC);
  extension->set_start_time(0);
  extension->set_period(5);
  extension->set_sync_output_timestamps(true);
  extension->set_update_frame_rate(true);
  SimpleRunner runner(options);
  runner.SetInput({8, 16, 24, 32, 40, 48, 56});
  runner.SetFrameRate(1000000.0 / 8);
  MP_ASSERT_OK(runner.Run());

  const std::vector<int64_t> expected_timestamps = {10, 15, 25, 30, 40, 50, 55};
  EXPECT_EQ(expected_timestamps, runner.GetOutputTimestamps());
  // The true (long-run) sampling period is 8.
  EXPECT_DOUBLE_EQ(1000000.0 / 8, runner.GetFrameRate());
}

}  // namespace
}  // namespace mediapipe
