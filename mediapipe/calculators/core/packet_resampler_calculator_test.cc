// Copyright 2018 The MediaPipe Authors.
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

#include "mediapipe/calculators/core/packet_resampler_calculator.h"

#include <cmath>
#include <memory>
#include <string>
#include <vector>

#include "absl/strings/str_cat.h"
#include "mediapipe/calculators/core/packet_resampler_calculator.pb.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/calculator_runner.h"
#include "mediapipe/framework/formats/video_stream_header.h"
#include "mediapipe/framework/port/gmock.h"
#include "mediapipe/framework/port/gtest.h"
#include "mediapipe/framework/port/parse_text_proto.h"
#include "mediapipe/framework/port/status_matchers.h"

namespace mediapipe {

using ::testing::ElementsAre;
namespace {

constexpr char kOptionsTag[] = "OPTIONS";
constexpr char kSeedTag[] = "SEED";
constexpr char kVideoHeaderTag[] = "VIDEO_HEADER";
constexpr char kDataTag[] = "DATA";

// A simple version of CalculatorRunner with built-in convenience
// methods for setting inputs from a vector and checking outputs
// against expected outputs (both timestamps and contents).
class SimpleRunner : public CalculatorRunner {
 public:
  explicit SimpleRunner(const std::string& options_string)
      : CalculatorRunner("PacketResamplerCalculator", options_string, 1, 1, 0) {
  }
  explicit SimpleRunner(const CalculatorGraphConfig::Node& node_config)
      : CalculatorRunner(node_config) {}

  virtual ~SimpleRunner() {}

  void SetInput(const std::vector<int64_t>& timestamp_list) {
    MutableInputs()->Index(0).packets.clear();
    for (const int64_t ts : timestamp_list) {
      MutableInputs()->Index(0).packets.push_back(
          Adopt(new std::string(absl::StrCat("Frame #", ts)))
              .At(Timestamp(ts)));
    }
  }

  void SetVideoHeader(const double frame_rate) {
    video_header_.width = static_count_;
    video_header_.height = static_count_ * 10;
    video_header_.frame_rate = frame_rate;
    video_header_.duration = static_count_ * 100.0;
    video_header_.format = static_cast<ImageFormat::Format>(
        static_count_ % ImageFormat::Format_ARRAYSIZE);
    MutableInputs()->Index(0).header = Adopt(new VideoHeader(video_header_));
    ++static_count_;
  }

  void CheckOutputTimestamps(
      const std::vector<int64_t>& expected_frames,
      const std::vector<int64_t>& expected_timestamps) const {
    EXPECT_EQ(expected_frames.size(), Outputs().Index(0).packets.size());
    EXPECT_EQ(expected_timestamps.size(), Outputs().Index(0).packets.size());
    int count = 0;
    for (const Packet& packet : Outputs().Index(0).packets) {
      EXPECT_EQ(Timestamp(expected_timestamps[count]), packet.Timestamp());
      const std::string& packet_contents = packet.Get<std::string>();
      EXPECT_EQ(std::string(absl::StrCat("Frame #", expected_frames[count])),
                packet_contents);
      ++count;
    }
  }

  void CheckVideoHeader(const double expected_frame_rate) const {
    ASSERT_FALSE(Outputs().Index(0).header.IsEmpty());
    const VideoHeader& header = Outputs().Index(0).header.Get<VideoHeader>();
    const double frame_rate = header.frame_rate;

    EXPECT_EQ(video_header_.width, header.width);
    EXPECT_EQ(video_header_.height, header.height);
    EXPECT_DOUBLE_EQ(expected_frame_rate, frame_rate);
    EXPECT_FLOAT_EQ(video_header_.duration, header.duration);
    EXPECT_EQ(video_header_.format, header.format);
  }

 private:
  VideoHeader video_header_;
  static int static_count_;
};

// Matcher for Packets with uint64 payload, comparing arg packet's
// timestamp and uint64 payload.
MATCHER_P2(PacketAtTimestamp, payload, timestamp,
           absl::StrCat(negation ? "isn't" : "is", " a packet with payload ",
                        payload, " @ time ", timestamp)) {
  if (timestamp != arg.Timestamp().Value()) {
    *result_listener << "at incorrect timestamp = " << arg.Timestamp().Value();
    return false;
  }
  int64_t actual_payload = arg.template Get<int64_t>();
  if (actual_payload != payload) {
    *result_listener << "with incorrect payload = " << actual_payload;
    return false;
  }
  return true;
}

// JitterWithReflectionStrategy child class which injects a specified stream
// of "random" numbers.
//
// Calculators are created through factory methods, making testing and injection
// tricky.  This class utilizes a static variable, random_sequence, to pass
// the desired random sequence into the calculator.
class ReproducibleJitterWithReflectionStrategyForTesting
    : public ReproducibleJitterWithReflectionStrategy {
 public:
  ReproducibleJitterWithReflectionStrategyForTesting(
      PacketResamplerCalculator* calculator)
      : ReproducibleJitterWithReflectionStrategy(calculator) {}

  // Statically accessed random sequence to use for jitter with reflection.
  //
  // An EXPECT will fail if sequence is less than the number requested during
  // processing.
  static std::vector<uint64_t> random_sequence;

 protected:
  virtual uint64_t GetNextRandom(uint64_t n) {
    EXPECT_LT(sequence_index_, random_sequence.size());
    return random_sequence[sequence_index_++] % n;
  }

 private:
  int32_t sequence_index_ = 0;
};
std::vector<uint64_t>
    ReproducibleJitterWithReflectionStrategyForTesting::random_sequence;

// PacketResamplerCalculator child class which injects a specified stream
// of "random" numbers.
//
// Calculators are created through factory methods, making testing and injection
// tricky.  This class utilizes a static variable, random_sequence, to pass
// the desired random sequence into the calculator.
class ReproducibleResamplerCalculatorForTesting
    : public PacketResamplerCalculator {
 public:
  static absl::Status GetContract(CalculatorContract* cc) {
    return PacketResamplerCalculator::GetContract(cc);
  }

 protected:
  std::unique_ptr<class PacketResamplerStrategy> GetSamplingStrategy(
      const mediapipe::PacketResamplerCalculatorOptions& Options) {
    return absl::make_unique<
        ReproducibleJitterWithReflectionStrategyForTesting>(this);
  }
};

REGISTER_CALCULATOR(ReproducibleResamplerCalculatorForTesting);

int SimpleRunner::static_count_ = 0;

TEST(PacketResamplerCalculatorTest, NoPacketsInStream) {
  {
    SimpleRunner runner(
        "[mediapipe.PacketResamplerCalculatorOptions.ext]: "
        "{frame_rate:30}");
    runner.SetInput({});
    MP_ASSERT_OK(runner.Run());
  }
}

TEST(PacketResamplerCalculatorTest, SinglePacketInStream) {
  // Stream with 1 packet / 1 period.
  {
    SimpleRunner runner(
        "[mediapipe.PacketResamplerCalculatorOptions.ext]: "
        "{frame_rate:30}");
    runner.SetInput({0});
    MP_ASSERT_OK(runner.Run());
    runner.CheckOutputTimestamps({0}, {0});
  }

  // Stream with 1 packet / 1 period (0 < packet timestamp < first limit).
  {
    SimpleRunner runner(
        "[mediapipe.PacketResamplerCalculatorOptions.ext]: "
        "{frame_rate:30}");
    runner.SetInput({1000});
    MP_ASSERT_OK(runner.Run());
    runner.CheckOutputTimestamps({1000}, {1000});
  }

  // Stream with 1 packet / 1 period (packet timestamp > first limit).
  {
    SimpleRunner runner(
        "[mediapipe.PacketResamplerCalculatorOptions.ext]: "
        "{frame_rate:30}");
    runner.SetInput({16668});
    MP_ASSERT_OK(runner.Run());
    runner.CheckOutputTimestamps({16668}, {16668});
  }
}

TEST(PacketResamplerCalculatorTest, TwoPacketsInStream) {
  // Stream with 2 packets / 1 period.
  {
    SimpleRunner runner(
        "[mediapipe.PacketResamplerCalculatorOptions.ext]: "
        "{frame_rate:30}");
    runner.SetInput({0, 16666});
    MP_ASSERT_OK(runner.Run());
    runner.CheckOutputTimestamps({0}, {0});
  }

  // Stream with 2 packets / 2 periods (left extreme for second period).
  {
    SimpleRunner runner(
        "[mediapipe.PacketResamplerCalculatorOptions.ext]: "
        "{frame_rate:30}");
    runner.SetInput({0, 16667});
    MP_ASSERT_OK(runner.Run());
    runner.CheckOutputTimestamps({0, 16667}, {0, 33333});
  }

  // Stream with 2 packets / 2 periods (right extreme for second period).
  {
    SimpleRunner runner(
        "[mediapipe.PacketResamplerCalculatorOptions.ext]: "
        "{frame_rate:30}");
    runner.SetInput({0, 49999});
    MP_ASSERT_OK(runner.Run());
    runner.CheckOutputTimestamps({0, 49999}, {0, 33333});
  }

  // Stream with 2 packets / 3 periods (filling 1 in the middle).
  {
    SimpleRunner runner(
        "[mediapipe.PacketResamplerCalculatorOptions.ext]: "
        "{frame_rate:30}");
    runner.SetInput({0, 50000});
    MP_ASSERT_OK(runner.Run());
    runner.CheckOutputTimestamps({0, 0, 50000}, {0, 33333, 66667});
  }

  // Stream with 2 packets / 4 periods (filling 2 in the middle).
  {
    SimpleRunner runner(
        "[mediapipe.PacketResamplerCalculatorOptions.ext]: "
        "{frame_rate:30}");
    runner.SetInput({2000, 118666});
    MP_ASSERT_OK(runner.Run());
    runner.CheckOutputTimestamps({2000, 2000, 2000, 118666},
                                 {2000, 35333, 68667, 102000});
  }
}

TEST(PacketResamplerCalculatorTest, UseInputFrameRate_HeaderHasSameFramerate) {
  CalculatorRunner runner(ParseTextProtoOrDie<CalculatorGraphConfig::Node>(R"pb(
    calculator: "PacketResamplerCalculator"
    input_stream: "DATA:in_data"
    input_stream: "VIDEO_HEADER:in_video_header"
    output_stream: "DATA:out_data"
    options {
      [mediapipe.PacketResamplerCalculatorOptions.ext] {
        use_input_frame_rate: true
        frame_rate: 1000.0
      }
    }
  )pb"));

  for (const int64_t ts : {0, 5000, 10010, 15001, 19990}) {
    runner.MutableInputs()->Tag(kDataTag).packets.push_back(
        Adopt(new std::string(absl::StrCat("Frame #", ts))).At(Timestamp(ts)));
  }
  VideoHeader video_header_in;
  video_header_in.width = 10;
  video_header_in.height = 100;
  video_header_in.frame_rate = 200.0;
  video_header_in.duration = 1.0;
  video_header_in.format = ImageFormat::SRGB;
  runner.MutableInputs()
      ->Tag(kVideoHeaderTag)
      .packets.push_back(
          Adopt(new VideoHeader(video_header_in)).At(Timestamp::PreStream()));
  MP_ASSERT_OK(runner.Run());

  std::vector<int64_t> expected_frames = {0, 5000, 10010, 15001, 19990};
  std::vector<int64_t> expected_timestamps = {0, 5000, 10000, 15000, 20000};
  EXPECT_EQ(expected_frames.size(),
            runner.Outputs().Tag(kDataTag).packets.size());
  EXPECT_EQ(expected_timestamps.size(),
            runner.Outputs().Tag(kDataTag).packets.size());

  int count = 0;
  for (const Packet& packet : runner.Outputs().Tag(kDataTag).packets) {
    EXPECT_EQ(Timestamp(expected_timestamps[count]), packet.Timestamp());
    const std::string& packet_contents = packet.Get<std::string>();
    EXPECT_EQ(std::string(absl::StrCat("Frame #", expected_frames[count])),
              packet_contents);
    ++count;
  }
}

TEST(PacketResamplerCalculatorTest,
     UseInputFrameRate_HeaderHasSmallerFramerate) {
  CalculatorRunner runner(ParseTextProtoOrDie<CalculatorGraphConfig::Node>(R"pb(
    calculator: "PacketResamplerCalculator"
    input_stream: "DATA:in_data"
    input_stream: "VIDEO_HEADER:in_video_header"
    output_stream: "DATA:out_data"
    options {
      [mediapipe.PacketResamplerCalculatorOptions.ext] {
        use_input_frame_rate: true
        frame_rate: 1000.0
      }
    }
  )pb"));

  for (const int64_t ts : {0, 5000, 10010, 15001}) {
    runner.MutableInputs()->Tag(kDataTag).packets.push_back(
        Adopt(new std::string(absl::StrCat("Frame #", ts))).At(Timestamp(ts)));
  }
  VideoHeader video_header_in;
  video_header_in.width = 10;
  video_header_in.height = 100;
  video_header_in.frame_rate = 100.0;
  video_header_in.duration = 1.0;
  video_header_in.format = ImageFormat::SRGB;
  runner.MutableInputs()
      ->Tag(kVideoHeaderTag)
      .packets.push_back(
          Adopt(new VideoHeader(video_header_in)).At(Timestamp::PreStream()));
  MP_ASSERT_OK(runner.Run());

  std::vector<int64_t> expected_frames = {0, 10010, 15001};
  std::vector<int64_t> expected_timestamps = {0, 10000, 20000};
  EXPECT_EQ(expected_frames.size(),
            runner.Outputs().Tag(kDataTag).packets.size());
  EXPECT_EQ(expected_timestamps.size(),
            runner.Outputs().Tag(kDataTag).packets.size());

  int count = 0;
  for (const Packet& packet : runner.Outputs().Tag(kDataTag).packets) {
    EXPECT_EQ(Timestamp(expected_timestamps[count]), packet.Timestamp());
    const std::string& packet_contents = packet.Get<std::string>();
    EXPECT_EQ(std::string(absl::StrCat("Frame #", expected_frames[count])),
              packet_contents);
    ++count;
  }
}

TEST(PacketResamplerCalculatorTest,
     UseInputFrameRate_MaxFrameRateSmallerThanInput) {
  CalculatorRunner runner(ParseTextProtoOrDie<CalculatorGraphConfig::Node>(R"pb(
    calculator: "PacketResamplerCalculator"
    input_stream: "DATA:in_data"
    input_stream: "VIDEO_HEADER:in_video_header"
    output_stream: "DATA:out_data"
    options {
      [mediapipe.PacketResamplerCalculatorOptions.ext] {
        use_input_frame_rate: true
        frame_rate: 1000.0
        max_frame_rate: 50.0
      }
    }
  )pb"));

  for (const int64_t ts : {0, 5000, 10010, 15001, 20010}) {
    runner.MutableInputs()->Tag(kDataTag).packets.push_back(
        Adopt(new std::string(absl::StrCat("Frame #", ts))).At(Timestamp(ts)));
  }
  VideoHeader video_header_in;
  video_header_in.width = 10;
  video_header_in.height = 200;
  video_header_in.frame_rate = 100.0;
  video_header_in.duration = 1.0;
  video_header_in.format = ImageFormat::SRGB;
  runner.MutableInputs()
      ->Tag(kVideoHeaderTag)
      .packets.push_back(
          Adopt(new VideoHeader(video_header_in)).At(Timestamp::PreStream()));
  MP_ASSERT_OK(runner.Run());

  std::vector<int64_t> expected_frames = {0, 20010};
  std::vector<int64_t> expected_timestamps = {0, 20000};
  EXPECT_EQ(expected_frames.size(),
            runner.Outputs().Tag(kDataTag).packets.size());
  EXPECT_EQ(expected_timestamps.size(),
            runner.Outputs().Tag(kDataTag).packets.size());

  int count = 0;
  for (const Packet& packet : runner.Outputs().Tag(kDataTag).packets) {
    EXPECT_EQ(Timestamp(expected_timestamps[count]), packet.Timestamp());
    const std::string& packet_contents = packet.Get<std::string>();
    EXPECT_EQ(std::string(absl::StrCat("Frame #", expected_frames[count])),
              packet_contents);
    ++count;
  }
}

TEST(PacketResamplerCalculatorTest, InputAtExactFrequencyMiddlepoints) {
  SimpleRunner runner(
      "[mediapipe.PacketResamplerCalculatorOptions.ext]: "
      "{frame_rate:30}");
  runner.SetInput({0, 33333, 66667, 100000, 133333, 166667, 200000});
  MP_ASSERT_OK(runner.Run());
  runner.CheckOutputTimestamps(
      {0, 33333, 66667, 100000, 133333, 166667, 200000},
      {0, 33333, 66667, 100000, 133333, 166667, 200000});
}

// When there are several candidates for a period, the one closer to the center
// should be sent to the output.
TEST(PacketResamplerCalculatorTest, MultiplePacketsForPeriods) {
  SimpleRunner runner(
      "[mediapipe.PacketResamplerCalculatorOptions.ext]: "
      "{frame_rate:30}");
  runner.SetInput({0, 16666, 16667, 20000, 33300, 49999, 50000, 66600});
  MP_ASSERT_OK(runner.Run());
  runner.CheckOutputTimestamps({0, 33300, 66600}, {0, 33333, 66667});
}

// When a period must be filled, we use the latest packet received (not
// necessarily the same as the one stored for the best in the previous period).
TEST(PacketResamplerCalculatorTest, FillPeriodsWithLatestPacket) {
  {
    SimpleRunner runner(
        "[mediapipe.PacketResamplerCalculatorOptions.ext]: "
        "{frame_rate:30}");
    runner.SetInput({0, 5000, 16666, 83334});
    MP_ASSERT_OK(runner.Run());
    runner.CheckOutputTimestamps({0, 16666, 16666, 83334},
                                 {0, 33333, 66667, 100000});
  }

  {
    SimpleRunner runner(
        "[mediapipe.PacketResamplerCalculatorOptions.ext]: "
        "{frame_rate:30}");
    runner.SetInput({0, 16666, 16667, 25000, 33000, 35000, 135000});
    MP_ASSERT_OK(runner.Run());
    runner.CheckOutputTimestamps({0, 33000, 35000, 35000, 135000},
                                 {0, 33333, 66667, 100000, 133333});
  }

  {
    SimpleRunner runner(
        "[mediapipe.PacketResamplerCalculatorOptions.ext]: "
        "{frame_rate:30}");
    runner.SetInput({0, 15000, 32000, 49999, 150000});
    MP_ASSERT_OK(runner.Run());
    runner.CheckOutputTimestamps({0, 32000, 49999, 49999, 49999, 150000},
                                 {0, 33333, 66667, 100000, 133333, 166667});
  }
}

TEST(PacketResamplerCalculatorTest, SuperHighFrameRate) {
  // frame rate == 500000 (a packet will have to be sent every 2 ticks).
  {
    SimpleRunner runner(
        "[mediapipe.PacketResamplerCalculatorOptions.ext]: "
        "{frame_rate:500000}");
    runner.SetInput({0, 10, 13});
    MP_ASSERT_OK(runner.Run());
    runner.CheckOutputTimestamps({0, 0, 0, 0, 0, 10, 10, 13},
                                 {0, 2, 4, 6, 8, 10, 12, 14});
  }

  // frame rate == 1000000 (a packet will have to be sent in each tick).
  {
    SimpleRunner runner(
        "[mediapipe.PacketResamplerCalculatorOptions.ext]: "
        "{frame_rate:1000000}");
    runner.SetInput({0, 10, 13});
    MP_ASSERT_OK(runner.Run());
    runner.CheckOutputTimestamps(
        {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 10, 10, 10, 13},
        {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13});
  }
}

TEST(PacketResamplerCalculatorTest, NegativeTimestampTest) {
  // Stream with negative timestamps / 1 period.
  {
    SimpleRunner runner(
        "[mediapipe.PacketResamplerCalculatorOptions.ext]: "
        "{frame_rate:30}");
    runner.SetInput({-200, -20, 16466});
    MP_ASSERT_OK(runner.Run());
    runner.CheckOutputTimestamps({-200}, {-200});
  }

  // Stream with negative timestamps / 2 periods.
  {
    SimpleRunner runner(
        "[mediapipe.PacketResamplerCalculatorOptions.ext]: "
        "{frame_rate:30}");
    runner.SetInput({-200, -20, 16467});
    MP_ASSERT_OK(runner.Run());
    runner.CheckOutputTimestamps({-200, 16467}, {-200, 33133});
  }

  // Stream with negative timestamps and filling an empty period.
  {
    SimpleRunner runner(
        "[mediapipe.PacketResamplerCalculatorOptions.ext]: "
        "{frame_rate:30}");
    runner.SetInput({-500, 66667});
    MP_ASSERT_OK(runner.Run());
    runner.CheckOutputTimestamps({-500, -500, 66667}, {-500, 32833, 66167});
  }

  // Stream with negative timestamps and initial packet < -period.
  {
    SimpleRunner runner(
        "[mediapipe.PacketResamplerCalculatorOptions.ext]: "
        "{frame_rate:30}");
    runner.SetInput({-50000, -33334, 33334});
    MP_ASSERT_OK(runner.Run());
    runner.CheckOutputTimestamps({-50000, -33334, -33334, 33334},
                                 {-50000, -16667, 16667, 50000});
  }
}

TEST(PacketResamplerCalculatorTest, ExactFramesPerSecond) {
  // Using frame_rate=50, that makes a period of 20000 microsends (exact).
  {
    SimpleRunner runner(
        "[mediapipe.PacketResamplerCalculatorOptions.ext]: "
        "{frame_rate:50}");
    runner.SetInput({0, 9999, 29999});
    MP_ASSERT_OK(runner.Run());
    runner.CheckOutputTimestamps({0, 29999}, {0, 20000});
  }

  // Test filling empty periods.
  {
    SimpleRunner runner(
        "[mediapipe.PacketResamplerCalculatorOptions.ext]: "
        "{frame_rate:50}");
    runner.SetInput({0, 10000, 50000});
    MP_ASSERT_OK(runner.Run());
    runner.CheckOutputTimestamps({0, 10000, 10000, 50000},
                                 {0, 20000, 40000, 60000});
  }
}

TEST(PacketResamplerCalculatorTest, FrameRateTest) {
  // Test changing Frame Rate to the same initial value.
  {
    SimpleRunner runner(
        "[mediapipe.PacketResamplerCalculatorOptions.ext]: "
        "{frame_rate:50, output_header:UPDATE_VIDEO_HEADER}");
    runner.SetInput({0, 10000, 30000, 50000, 60000});
    runner.SetVideoHeader(50.0);
    MP_ASSERT_OK(runner.Run());
    runner.CheckOutputTimestamps({0, 10000, 30000, 60000},
                                 {0, 20000, 40000, 60000});
    runner.CheckVideoHeader(50.0);
  }

  // Test changing Frame Rate to new value.
  {
    SimpleRunner runner(
        "[mediapipe.PacketResamplerCalculatorOptions.ext]: "
        "{frame_rate:50, output_header:UPDATE_VIDEO_HEADER}");
    runner.SetInput({0, 5000, 10010, 15001, 19990});
    runner.SetVideoHeader(200.0);
    MP_ASSERT_OK(runner.Run());
    runner.CheckOutputTimestamps({0, 19990}, {0, 20000});
    runner.CheckVideoHeader(50.0);
  }

  // Test that the frame rate is not changing if update_video_header = false.
  {
    SimpleRunner runner(
        "[mediapipe.PacketResamplerCalculatorOptions.ext]: "
        "{frame_rate:50, output_header:PASS_HEADER}");
    runner.SetInput({0, 5000, 10010, 15001, 19990});
    runner.SetVideoHeader(200.0);
    MP_ASSERT_OK(runner.Run());
    runner.CheckOutputTimestamps({0, 19990}, {0, 20000});
    runner.CheckVideoHeader(200.0);
  }
}

TEST(PacketResamplerCalculatorTest, SetVideoHeader) {
  CalculatorRunner runner(ParseTextProtoOrDie<CalculatorGraphConfig::Node>(R"pb(
    calculator: "PacketResamplerCalculator"
    input_stream: "DATA:in_data"
    input_stream: "VIDEO_HEADER:in_video_header"
    output_stream: "DATA:out_data"
    output_stream: "VIDEO_HEADER:out_video_header"
    options {
      [mediapipe.PacketResamplerCalculatorOptions.ext] { frame_rate: 50.0 }
    }
  )pb"));

  for (const int64_t ts : {0, 5000, 10010, 15001, 19990}) {
    runner.MutableInputs()->Tag(kDataTag).packets.push_back(
        Adopt(new std::string(absl::StrCat("Frame #", ts))).At(Timestamp(ts)));
  }
  VideoHeader video_header_in;
  video_header_in.width = 10;
  video_header_in.height = 100;
  video_header_in.frame_rate = 1.0;
  video_header_in.duration = 1.0;
  video_header_in.format = ImageFormat::SRGB;
  runner.MutableInputs()
      ->Tag(kVideoHeaderTag)
      .packets.push_back(
          Adopt(new VideoHeader(video_header_in)).At(Timestamp::PreStream()));
  MP_ASSERT_OK(runner.Run());

  ASSERT_EQ(1, runner.Outputs().Tag(kVideoHeaderTag).packets.size());
  EXPECT_EQ(Timestamp::PreStream(),
            runner.Outputs().Tag(kVideoHeaderTag).packets[0].Timestamp());
  const VideoHeader& video_header_out =
      runner.Outputs().Tag(kVideoHeaderTag).packets[0].Get<VideoHeader>();
  EXPECT_EQ(video_header_in.width, video_header_out.width);
  EXPECT_EQ(video_header_in.height, video_header_out.height);
  EXPECT_DOUBLE_EQ(50.0, video_header_out.frame_rate);
  EXPECT_FLOAT_EQ(video_header_in.duration, video_header_out.duration);
  EXPECT_EQ(video_header_in.format, video_header_out.format);
}

TEST(PacketResamplerCalculatorTest, FlushLastPacketWithoutRound) {
  SimpleRunner runner(R"(
      [mediapipe.PacketResamplerCalculatorOptions.ext] {
        frame_rate: 1
      })");
  runner.SetInput({0, 333333, 666667, 1000000, 1333333});
  MP_ASSERT_OK(runner.Run());
  // 1333333 is not emitted as 2000000, because it does not round to 2000000.
  runner.CheckOutputTimestamps({0, 1000000}, {0, 1000000});
}

TEST(PacketResamplerCalculatorTest, FlushLastPacketWithRound) {
  SimpleRunner runner(R"(
      [mediapipe.PacketResamplerCalculatorOptions.ext] {
        frame_rate: 1
      })");
  runner.SetInput({0, 333333, 666667, 1000000, 1333333, 1666667});
  MP_ASSERT_OK(runner.Run());
  // 1666667 is emitted as 2000000, because it rounds to 2000000.
  runner.CheckOutputTimestamps({0, 1000000, 1666667}, {0, 1000000, 2000000});
}

TEST(PacketResamplerCalculatorTest, DoNotFlushLastPacketWithoutRound) {
  SimpleRunner runner(R"(
      [mediapipe.PacketResamplerCalculatorOptions.ext] {
        frame_rate: 1
        flush_last_packet: false
      })");
  runner.SetInput({0, 333333, 666667, 1000000, 1333333});
  MP_ASSERT_OK(runner.Run());
  // 1333333 is not emitted no matter what; see FlushLastPacketWithoutRound.
  runner.CheckOutputTimestamps({0, 1000000}, {0, 1000000});
}

TEST(PacketResamplerCalculatorTest, DoNotFlushLastPacketWithRound) {
  SimpleRunner runner(R"(
      [mediapipe.PacketResamplerCalculatorOptions.ext] {
        frame_rate: 1
        flush_last_packet: false
      })");
  runner.SetInput({0, 333333, 666667, 1000000, 1333333, 1666667});
  MP_ASSERT_OK(runner.Run());
  // 1666667 is not emitted due to flush_last_packet: false.
  runner.CheckOutputTimestamps({0, 1000000}, {0, 1000000});
}

// When base_timestamp is specified, output timestamps are aligned with it.
TEST(PacketResamplerCalculatorTest, InputAtExactFrequencyMiddlepointsAligned) {
  {
    // Without base_timestamp, outputs are aligned with the first input
    // timestamp, (33333 - 222).
    SimpleRunner runner(
        "[mediapipe.PacketResamplerCalculatorOptions.ext]: "
        "{frame_rate:30}");
    runner.SetInput({33111, 66667, 100000, 133333, 166667, 200000});
    MP_ASSERT_OK(runner.Run());
    runner.CheckOutputTimestamps({33111, 66667, 100000, 133333, 166667, 200000},
                                 {33111, 66444, 99778, 133111, 166444, 199778});
  }
  {
    // With base_timestamp, outputs are aligned with base_timestamp, 0.
    SimpleRunner runner(
        "[mediapipe.PacketResamplerCalculatorOptions.ext]: "
        "{frame_rate:30 "
        "base_timestamp:0}");
    runner.SetInput({33111, 66667, 100000, 133333, 166667, 200000});
    MP_ASSERT_OK(runner.Run());
    runner.CheckOutputTimestamps(
        {33111, 66667, 100000, 133333, 166667, 200000},
        {33333, 66666, 100000, 133333, 166666, 200000});
  }
}

// When base_timestamp is specified, output timestamps are aligned with it.
TEST(PacketResamplerCalculatorTest, MultiplePacketsForPeriodsAligned) {
  {
    // Without base_timestamp, outputs are aligned with the first input, -222.
    SimpleRunner runner(
        "[mediapipe.PacketResamplerCalculatorOptions.ext]: "
        "{frame_rate:30}");
    runner.SetInput({-222, 16666, 16667, 20000, 33300, 49999, 50000, 66600});
    MP_ASSERT_OK(runner.Run());
    runner.CheckOutputTimestamps({-222, 33300, 66600}, {-222, 33111, 66445});
  }
  {
    // With base_timestamp, outputs are aligned with base_timestamp, 900011.
    SimpleRunner runner(
        "[mediapipe.PacketResamplerCalculatorOptions.ext]: "
        "{frame_rate:30 "
        "base_timestamp:900011}");
    runner.SetInput({-222, 16666, 16667, 20000, 33300, 49999, 50000, 66600});
    MP_ASSERT_OK(runner.Run());
    runner.CheckOutputTimestamps({-222, 33300, 66600}, {11, 33344, 66678});
  }
  {
    // With base_timestamp, outputs still approximate input timestamps,
    // while aligned to base_timestamp, 11.
    SimpleRunner runner(
        "[mediapipe.PacketResamplerCalculatorOptions.ext]: "
        "{frame_rate:30 "
        "base_timestamp:11}");
    runner.SetInput(
        {899888, 916666, 916667, 920000, 933300, 949999, 950000, 966600});
    MP_ASSERT_OK(runner.Run());
    runner.CheckOutputTimestamps({899888, 933300, 966600},
                                 {900011, 933344, 966678});
  }
}

// When a period must be filled, we use the latest packet received.
// When base_timestamp is specified, output timestamps are aligned with it.
TEST(PacketResamplerCalculatorTest, FillPeriodsWithLatestPacketAligned) {
  {
    // Without base_timestamp, outputs are aligned with the first input, -222.
    SimpleRunner runner(
        "[mediapipe.PacketResamplerCalculatorOptions.ext]: "
        "{frame_rate:30}");
    runner.SetInput({-222, 15000, 32000, 49999, 150000});
    MP_ASSERT_OK(runner.Run());
    runner.CheckOutputTimestamps({-222, 32000, 49999, 49999, 49999, 150000},
                                 {-222, 33111, 66445, 99778, 133111, 166445});
  }
  {
    // With base_timestamp, outputs are aligned with base_timestamp, 0.
    SimpleRunner runner(
        "[mediapipe.PacketResamplerCalculatorOptions.ext]: "
        "{frame_rate:30 "
        "base_timestamp:0}");
    runner.SetInput({-222, 15000, 32000, 49999, 150000});
    MP_ASSERT_OK(runner.Run());
    runner.CheckOutputTimestamps({-222, 32000, 49999, 49999, 49999, 150000},
                                 {0, 33333, 66667, 100000, 133333, 166667});
  }
}

// When base_timestamp is specified, output timestamps are aligned with it.
// The first packet is included, because we assume that the input includes the
// whole first sampling interval.
TEST(PacketResamplerCalculatorTest, FirstInputAfterMiddlepointAligned) {
  {
    // Packet 100020 is omitted from the output sequence because
    // packet 99990 is closer to the period midpoint.
    SimpleRunner runner(
        "[mediapipe.PacketResamplerCalculatorOptions.ext]: "
        "{frame_rate:30 "
        "base_timestamp:0}");
    runner.SetInput({66667, 100020, 133333, 166667});
    MP_ASSERT_OK(runner.Run());
    runner.CheckOutputTimestamps({66667, 100020, 133333, 166667},
                                 {66667, 100000, 133334, 166667});
  }
  {
    // If we seek to packet 100020, packet 100020 is included in
    // the output sequence, because we assume that the input includes the
    // whole first sampling interval.
    //
    // We assume that the input includes whole sampling intervals
    // in order to produce "reproducible timestamps", which are timestamps
    // from the series of timestamps starting at 0.
    SimpleRunner runner(
        "[mediapipe.PacketResamplerCalculatorOptions.ext]: "
        "{frame_rate:30 "
        "base_timestamp:0}");
    runner.SetInput({100020, 133333, 166667});
    MP_ASSERT_OK(runner.Run());
    runner.CheckOutputTimestamps({100020, 133333, 166667},
                                 {100000, 133333, 166667});
  }
}

TEST(PacketResamplerCalculatorTest, OutputTimestampRangeAligned) {
  {
    // With base_timestamp, outputs are aligned with base_timestamp, 0.
    SimpleRunner runner(
        "[mediapipe.PacketResamplerCalculatorOptions.ext]: "
        "{frame_rate:30 "
        "base_timestamp:0}");
    runner.SetInput({-222, 15000, 32000, 49999, 150000});
    MP_ASSERT_OK(runner.Run());
    runner.CheckOutputTimestamps({-222, 32000, 49999, 49999, 49999, 150000},
                                 {0, 33333, 66667, 100000, 133333, 166667});
  }
  {
    // With start_time, end_time, outputs are filtered.
    SimpleRunner runner(
        "[mediapipe.PacketResamplerCalculatorOptions.ext]: "
        "{frame_rate:30 "
        "base_timestamp:0 "
        "start_time:40000 "
        "end_time:160000}");
    runner.SetInput({-222, 15000, 32000, 49999, 150000});
    MP_ASSERT_OK(runner.Run());
    runner.CheckOutputTimestamps({49999, 49999, 49999},
                                 {66667, 100000, 133333});
  }
  {
    // With start_time, end_time, round_limits, outputs are filtered,
    // rounding to the nearest limit.
    SimpleRunner runner(
        "[mediapipe.PacketResamplerCalculatorOptions.ext]: "
        "{frame_rate:30 "
        "base_timestamp:0 "
        "start_time:40000 "
        "end_time:160000 "
        "round_limits:true}");
    runner.SetInput({-222, 15000, 32000, 49999, 150000});
    MP_ASSERT_OK(runner.Run());
    runner.CheckOutputTimestamps({32000, 49999, 49999, 49999, 150000},
                                 {33333, 66667, 100000, 133333, 166667});
  }
}

TEST(PacketResamplerCalculatorTest, OptionsSidePacket) {
  CalculatorGraphConfig::Node node_config =
      ParseTextProtoOrDie<CalculatorGraphConfig::Node>(R"pb(
        calculator: "PacketResamplerCalculator"
        input_side_packet: "OPTIONS:options"
        input_stream: "input"
        output_stream: "output"
        options {
          [mediapipe.PacketResamplerCalculatorOptions.ext] {
            frame_rate: 60
            base_timestamp: 0
          }
        })pb");

  {
    SimpleRunner runner(node_config);
    auto options =
        new CalculatorOptions(ParseTextProtoOrDie<CalculatorOptions>(
            R"pb(
              [mediapipe.PacketResamplerCalculatorOptions.ext] {
                frame_rate: 30
              })pb"));
    runner.MutableSidePackets()->Tag(kOptionsTag) = Adopt(options);
    runner.SetInput({-222, 15000, 32000, 49999, 150000});
    MP_ASSERT_OK(runner.Run());
    EXPECT_EQ(6, runner.Outputs().Index(0).packets.size());
  }
  {
    SimpleRunner runner(node_config);

    auto options =
        new CalculatorOptions(ParseTextProtoOrDie<CalculatorOptions>(R"pb(
          merge_fields: false
          [mediapipe.PacketResamplerCalculatorOptions.ext] {
            frame_rate: 30
            base_timestamp: 0
          })pb"));
    runner.MutableSidePackets()->Tag(kOptionsTag) = Adopt(options);

    runner.SetInput({-222, 15000, 32000, 49999, 150000});
    MP_ASSERT_OK(runner.Run());
    EXPECT_EQ(6, runner.Outputs().Index(0).packets.size());
  }
}

}  // namespace
}  // namespace mediapipe
