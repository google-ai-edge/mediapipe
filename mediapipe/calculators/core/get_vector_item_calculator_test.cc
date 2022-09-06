#include "mediapipe/calculators/core/get_vector_item_calculator.h"

#include <memory>
#include <string>
#include <vector>

#include "absl/strings/str_format.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/calculator_runner.h"
#include "mediapipe/framework/port/gmock.h"
#include "mediapipe/framework/port/status_matchers.h"

namespace mediapipe {

MATCHER_P(IntPacket, value, "") {
  return testing::Value(arg.template Get<int>(), testing::Eq(value));
}

MATCHER_P(TimestampValue, value, "") {
  return testing::Value(arg.Timestamp(), testing::Eq(Timestamp(value)));
}

using TestGetIntVectorItemCalculator = api2::GetVectorItemCalculator<int>;
MEDIAPIPE_REGISTER_NODE(TestGetIntVectorItemCalculator);

CalculatorRunner MakeRunnerWithStream() {
  return CalculatorRunner(R"(
    calculator: "TestGetIntVectorItemCalculator"
    input_stream: "VECTOR:vector_stream"
    input_stream: "INDEX:index_stream"
    output_stream: "ITEM:item_stream"
  )");
}

CalculatorRunner MakeRunnerWithOptions(int set_index) {
  return CalculatorRunner(absl::StrFormat(R"(
    calculator: "TestGetIntVectorItemCalculator"
    input_stream: "VECTOR:vector_stream"
    output_stream: "ITEM:item_stream"
    options {
      [mediapipe.GetVectorItemCalculatorOptions.ext] {
        item_index: %d
      }
    }
  )",
                                          set_index));
}

void AddInputVector(CalculatorRunner& runner, const std::vector<int>& inputs,
                    int timestamp) {
  runner.MutableInputs()->Tag("VECTOR").packets.push_back(
      MakePacket<std::vector<int>>(inputs).At(Timestamp(timestamp)));
}

void AddInputIndex(CalculatorRunner& runner, int index, int timestamp) {
  runner.MutableInputs()->Tag("INDEX").packets.push_back(
      MakePacket<int>(index).At(Timestamp(timestamp)));
}

TEST(TestGetIntVectorItemCalculatorTest, EmptyIndexStreamNoOutput) {
  CalculatorRunner runner = MakeRunnerWithStream();
  const std::vector<int> inputs = {1, 2, 3};

  AddInputVector(runner, inputs, 1);
  MP_ASSERT_OK(runner.Run());

  const std::vector<Packet>& outputs = runner.Outputs().Tag("ITEM").packets;
  EXPECT_EQ(0, outputs.size());
}

TEST(TestGetIntVectorItemCalculatorTest, SuccessfulExtractionIndexStream) {
  CalculatorRunner runner = MakeRunnerWithStream();
  const std::vector<int> inputs = {1, 2, 3};
  const int index = 1;

  AddInputVector(runner, inputs, 1);
  AddInputIndex(runner, index, 1);
  MP_ASSERT_OK(runner.Run());

  const std::vector<Packet>& outputs = runner.Outputs().Tag("ITEM").packets;
  EXPECT_THAT(outputs, testing::ElementsAre(IntPacket(inputs[index])));
}

TEST(TestGetIntVectorItemCalculatorTest, SuccessfulExtractionIndexProto) {
  const int index = 2;
  CalculatorRunner runner = MakeRunnerWithOptions(index);
  const std::vector<int> inputs = {1, 2, 3};

  AddInputVector(runner, inputs, 1);
  MP_ASSERT_OK(runner.Run());

  const std::vector<Packet>& outputs = runner.Outputs().Tag("ITEM").packets;
  EXPECT_THAT(outputs, testing::ElementsAre(IntPacket(inputs[index])));
}

TEST(TestGetIntVectorItemCalculatorTest, StreamIsPreferred) {
  CalculatorRunner runner(R"(
    calculator: "TestGetIntVectorItemCalculator"
    input_stream: "VECTOR:vector_stream"
    input_stream: "INDEX:index_stream"
    output_stream: "ITEM:item_stream"
    options {
      [mediapipe.GetVectorItemCalculatorOptions.ext] {
        item_index: 2
      }
    }
  )");
  const std::vector<int> inputs = {1, 2, 3};
  const int stream_index = 0;

  AddInputVector(runner, inputs, 1);
  AddInputIndex(runner, stream_index, 1);
  MP_ASSERT_OK(runner.Run());

  const std::vector<Packet>& outputs = runner.Outputs().Tag("ITEM").packets;
  EXPECT_THAT(outputs, testing::ElementsAre(IntPacket(inputs[stream_index])));
}

TEST(TestGetIntVectorItemCalculatorTest, NoStreamNorOptionsExpectFail) {
  CalculatorRunner runner(R"(
    calculator: "TestGetIntVectorItemCalculator"
    input_stream: "VECTOR:vector_stream"
    output_stream: "ITEM:item_stream"
  )");

  absl::Status status = runner.Run();
  ASSERT_FALSE(status.ok());
  EXPECT_THAT(
      status.message(),
      testing::HasSubstr("kIdx(cc).IsConnected() || options.has_item_index()"));
}

TEST(TestGetIntVectorItemCalculatorTest, StreamIndexBoundsCheckFail1) {
  CalculatorRunner runner = MakeRunnerWithStream();
  const std::vector<int> inputs = {1, 2, 3};
  const int try_index = -1;

  AddInputVector(runner, inputs, 1);
  AddInputIndex(runner, try_index, 1);

  absl::Status status = runner.Run();
  ASSERT_FALSE(status.ok());
  EXPECT_THAT(status.message(),
              testing::HasSubstr("idx >= 0 && idx < items.size()"));
}

TEST(TestGetIntVectorItemCalculatorTest, StreamIndexBoundsCheckFail2) {
  CalculatorRunner runner = MakeRunnerWithStream();
  const std::vector<int> inputs = {1, 2, 3};
  const int try_index = 3;

  AddInputVector(runner, inputs, 1);
  AddInputIndex(runner, try_index, 1);

  absl::Status status = runner.Run();
  ASSERT_FALSE(status.ok());
  EXPECT_THAT(status.message(),
              testing::HasSubstr("idx >= 0 && idx < items.size()"));
}

TEST(TestGetIntVectorItemCalculatorTest, OptionsIndexBoundsCheckFail1) {
  const int try_index = -1;
  CalculatorRunner runner = MakeRunnerWithOptions(try_index);
  const std::vector<int> inputs = {1, 2, 3};

  AddInputVector(runner, inputs, 1);

  absl::Status status = runner.Run();
  ASSERT_FALSE(status.ok());
  EXPECT_THAT(status.message(),
              testing::HasSubstr("idx >= 0 && idx < items.size()"));
}

TEST(TestGetIntVectorItemCalculatorTest, OptionsIndexBoundsCheckFail2) {
  const int try_index = 3;
  CalculatorRunner runner = MakeRunnerWithOptions(try_index);
  const std::vector<int> inputs = {1, 2, 3};

  AddInputVector(runner, inputs, 1);

  absl::Status status = runner.Run();
  ASSERT_FALSE(status.ok());
  EXPECT_THAT(status.message(),
              testing::HasSubstr("idx >= 0 && idx < items.size()"));
}

TEST(TestGetIntVectorItemCalculatorTest, IndexStreamTwoTimestamps) {
  CalculatorRunner runner = MakeRunnerWithStream();

  {
    const std::vector<int> inputs = {1, 2, 3};
    const int index = 1;
    AddInputVector(runner, inputs, 1);
    AddInputIndex(runner, index, 1);
  }
  {
    const std::vector<int> inputs = {5, 6, 7, 8};
    const int index = 3;
    AddInputVector(runner, inputs, 2);
    AddInputIndex(runner, index, 2);
  }
  MP_ASSERT_OK(runner.Run());

  const std::vector<Packet>& outputs = runner.Outputs().Tag("ITEM").packets;
  EXPECT_THAT(outputs, testing::ElementsAre(IntPacket(2), IntPacket(8)));
  EXPECT_THAT(outputs,
              testing::ElementsAre(TimestampValue(1), TimestampValue(2)));
}

TEST(TestGetIntVectorItemCalculatorTest, IndexOptionsTwoTimestamps) {
  const int static_index = 2;
  CalculatorRunner runner = MakeRunnerWithOptions(static_index);

  {
    const std::vector<int> inputs = {1, 2, 3};
    AddInputVector(runner, inputs, 1);
  }
  {
    const std::vector<int> inputs = {5, 6, 7, 8};
    AddInputVector(runner, inputs, 2);
  }
  MP_ASSERT_OK(runner.Run());

  const std::vector<Packet>& outputs = runner.Outputs().Tag("ITEM").packets;
  EXPECT_THAT(outputs, testing::ElementsAre(IntPacket(3), IntPacket(7)));
  EXPECT_THAT(outputs,
              testing::ElementsAre(TimestampValue(1), TimestampValue(2)));
}

}  // namespace mediapipe
