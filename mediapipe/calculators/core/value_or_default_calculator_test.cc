#include <algorithm>
#include <cstdint>
#include <vector>

#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/calculator_runner.h"
#include "mediapipe/framework/packet.h"
#include "mediapipe/framework/port/gmock.h"
#include "mediapipe/framework/port/gtest.h"
#include "mediapipe/framework/port/status_matchers.h"

namespace mediapipe {
namespace {

using ::testing::AllOf;
using ::testing::ContainerEq;
using ::testing::Each;
using ::testing::ElementsAre;
using ::testing::IsEmpty;
using ::testing::SizeIs;
using ::testing::Test;

const int kDefaultValue = 0;

// Utility to a create a mediapipe graph runner with the tested calculator and a
// default value, for all the tests.
class ValueOrDefaultRunner : public mediapipe::CalculatorRunner {
 public:
  ValueOrDefaultRunner()
      : mediapipe::CalculatorRunner(R"pb(
          calculator: "ValueOrDefaultCalculator"
          input_stream: "IN:in"
          input_stream: "TICK:tick"
          input_side_packet: "default"
          output_stream: "OUT:out"
          output_stream: "FLAG:used_default"
        )pb") {
    MutableSidePackets()->Index(0) = mediapipe::MakePacket<int>(kDefaultValue);
  }

  // Utility to push inputs to the runner to the TICK stream, so we could easily
  // tick.
  void TickAt(int64_t time) {
    // The type or value of the stream isn't relevant, we use just a bool.
    MutableInputs()->Tag("TICK").packets.push_back(
        mediapipe::Adopt(new bool(false)).At(mediapipe::Timestamp(time)));
  }

  // Utility to push the real inputs to the runner (IN stream).
  void ProvideInput(int64_t time, int value) {
    MutableInputs()->Tag("IN").packets.push_back(
        mediapipe::Adopt(new int(value)).At(mediapipe::Timestamp(time)));
  }

  // Extracts the timestamps (as int64) of the output stream of the calculator.
  std::vector<int64_t> GetOutputTimestamps() const {
    std::vector<int64_t> timestamps;
    for (const mediapipe::Packet& packet : Outputs().Tag("OUT").packets) {
      timestamps.emplace_back(packet.Timestamp().Value());
    }
    return timestamps;
  }

  // Extracts the values from the output stream of the calculator.
  std::vector<int> GetOutputValues() const {
    std::vector<int> values;
    for (const mediapipe::Packet& packet : Outputs().Tag("OUT").packets) {
      values.emplace_back(packet.Get<int>());
    }
    return values;
  }

  // Extracts the timestamps (as int64) of the flag stream, which indicates on
  // times without an input value (i.e. using the default value).
  std::vector<int64_t> GetFlagTimestamps() const {
    std::vector<int64_t> timestamps;
    for (const mediapipe::Packet& packet : Outputs().Tag("FLAG").packets) {
      timestamps.emplace_back(packet.Timestamp().Value());
    }
    return timestamps;
  }

  // Extracts the output from the flags stream (which should always be true).
  std::vector<bool> GetFlagValues() const {
    std::vector<bool> flags;
    for (const mediapipe::Packet& packet : Outputs().Tag("FLAG").packets) {
      flags.emplace_back(packet.Get<bool>());
    }
    return flags;
  }
};

// To be used as input values:
std::vector<int> GetIntegersRange(int size) {
  std::vector<int> result;
  for (int i = 0; i < size; ++i) {
    // We start with default-value+1 so it won't contain the default value.
    result.push_back(kDefaultValue + 1 + i);
  }
  return result;
}

TEST(ValueOrDefaultCalculatorTest, NoInputs) {
  // Check that when no real inputs are provided - we get the default value over
  // and over, with the correct timestamps.
  ValueOrDefaultRunner runner;
  const std::vector<int64_t> ticks = {0, 1, 2, 5, 8, 12, 33, 231};

  for (int tick : ticks) {
    runner.TickAt(tick);
  }

  MP_EXPECT_OK(runner.Run());

  // Make sure we get the right timestamps:
  EXPECT_THAT(runner.GetOutputTimestamps(), ContainerEq(ticks));
  // All should be default value:
  EXPECT_THAT(runner.GetOutputValues(),
              AllOf(Each(kDefaultValue), SizeIs(ticks.size())));
  // We should get the default indication all the time:
  EXPECT_THAT(runner.GetFlagTimestamps(), ContainerEq(ticks));
}

TEST(ValueOrDefaultCalculatorTest, NeverDefault) {
  // Check that when we provide the inputs on time - we get them as outputs.
  ValueOrDefaultRunner runner;
  const std::vector<int64_t> ticks = {0, 1, 2, 5, 8, 12, 33, 231};
  const std::vector<int> values = GetIntegersRange(ticks.size());

  for (int i = 0; i < ticks.size(); ++i) {
    runner.TickAt(ticks[i]);
    runner.ProvideInput(ticks[i], values[i]);
  }

  MP_EXPECT_OK(runner.Run());

  // Make sure we get the right timestamps:
  EXPECT_THAT(runner.GetOutputTimestamps(), ContainerEq(ticks));
  // Should get the inputs values:
  EXPECT_THAT(runner.GetOutputValues(), ContainerEq(values));
  // We should never get the default indication:
  EXPECT_THAT(runner.GetFlagTimestamps(), IsEmpty());
}

TEST(ValueOrDefaultCalculatorTest, DefaultAndValues) {
  // Check that when we provide inputs only part of the time - we get them, but
  // defaults at the missing times.
  // That's the usual use case for this calculator.
  ValueOrDefaultRunner runner;
  const std::vector<int64_t> ticks = {0, 1, 5, 8, 12, 231};
  // Provide inputs only part of the ticks.
  // Chosen so there will be defaults before the first input, between the
  // inputs and after the last input.
  const std::vector<int64_t> in_ticks = {/*0,*/ 1, 5, /*8,*/ 12, /*, 231*/};
  const std::vector<int> in_values = GetIntegersRange(in_ticks.size());

  for (int tick : ticks) {
    runner.TickAt(tick);
  }
  for (int i = 0; i < in_ticks.size(); ++i) {
    runner.ProvideInput(in_ticks[i], in_values[i]);
  }

  MP_EXPECT_OK(runner.Run());

  // Make sure we get all the timestamps:
  EXPECT_THAT(runner.GetOutputTimestamps(), ContainerEq(ticks));
  // The timestamps of the flag should be exactly the ones not in in_ticks.
  EXPECT_THAT(runner.GetFlagTimestamps(), ElementsAre(0, 8, 231));
  // And the values are default in these times, and the input values for
  // in_ticks.
  EXPECT_THAT(
      runner.GetOutputValues(),
      ElementsAre(kDefaultValue, 1, 2, kDefaultValue, 3, kDefaultValue));
}

TEST(ValueOrDefaultCalculatorTest, TimestampsMismatch) {
  // Check that when we provide the inputs not on time - we don't get them.
  ValueOrDefaultRunner runner;
  const std::vector<int64_t> ticks = {1, 2, 5, 8, 12, 33, 231};
  // The timestamps chosen so it will be before the first tick, in between ticks
  // and after the last one. Also - more inputs than ticks.
  const std::vector<int64_t> in_ticks = {0,  3,  4,  6,  7,  9,  10,
                                         11, 13, 14, 15, 16, 232};
  const std::vector<int> in_values = GetIntegersRange(in_ticks.size());
  for (int tick : ticks) {
    runner.TickAt(tick);
  }
  for (int i = 0; i < in_ticks.size(); ++i) {
    runner.ProvideInput(in_ticks[i], in_values[i]);
  }

  MP_EXPECT_OK(runner.Run());

  // Non of the in_ticks should be inserted:
  EXPECT_THAT(runner.GetOutputTimestamps(), ContainerEq(ticks));
  EXPECT_THAT(runner.GetOutputValues(),
              AllOf(Each(kDefaultValue), SizeIs(ticks.size())));
  // All (and only) ticks should get the default.
  EXPECT_THAT(runner.GetFlagTimestamps(), ContainerEq(ticks));
}

TEST(ValueOrDefaultCalculatorTest, FlagValue) {
  // Since we anyway suppose that the Flag is a bool - there is nothing
  // interesting to check, but we should check once that the value is the right
  // (true) one.
  ValueOrDefaultRunner runner;
  runner.TickAt(0);
  MP_EXPECT_OK(runner.Run());
  EXPECT_THAT(runner.GetFlagValues(), ElementsAre(true));
}

TEST(ValueOrDefaultCalculatorTest, FullTest) {
  // Make sure that nothing gets wrong with an input that have both right and
  // wrong timestamps, some defaults etc.
  ValueOrDefaultRunner runner;
  const std::vector<int64_t> ticks = {1, 2, 5, 8, 12, 33, 231};
  const std::vector<int64_t> in_ticks = {0, 2, 4, 6, 8, 9, 12, 33, 54, 232};
  const std::vector<int> in_values = GetIntegersRange(in_ticks.size());

  for (int tick : ticks) {
    runner.TickAt(tick);
  }
  for (int i = 0; i < in_ticks.size(); ++i) {
    runner.ProvideInput(in_ticks[i], in_values[i]);
  }

  MP_EXPECT_OK(runner.Run());

  EXPECT_THAT(runner.GetOutputTimestamps(), ContainerEq(ticks));
  // Calculated by hand:
  EXPECT_THAT(
      runner.GetOutputValues(),
      ElementsAre(kDefaultValue, 2, kDefaultValue, 5, 7, 8, kDefaultValue));
  EXPECT_THAT(runner.GetFlagTimestamps(), ElementsAre(1, 5, 231));
  EXPECT_THAT(runner.GetFlagValues(), AllOf(Each(true), SizeIs(3)));
}

}  // namespace
}  // namespace mediapipe
