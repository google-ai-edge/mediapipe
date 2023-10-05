#include "mediapipe/framework/api2/stream/merge.h"

#include "mediapipe/framework/api2/builder.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/port/gmock.h"
#include "mediapipe/framework/port/gtest.h"
#include "mediapipe/framework/port/parse_text_proto.h"

namespace mediapipe::api2::builder {
namespace {

TEST(Merge, VerifyConfig) {
  mediapipe::api2::builder::Graph graph;

  Stream<int> input_a = graph.In("INPUT_A").Cast<int>();
  Stream<int> input_b = graph.In("INPUT_B").Cast<int>();
  Stream<int> input = Merge(input_a, input_b, graph);
  input.SetName("input");

  EXPECT_THAT(
      graph.GetConfig(),
      EqualsProto(mediapipe::ParseTextProtoOrDie<CalculatorGraphConfig>(R"pb(
        node {
          calculator: "MergeCalculator"
          input_stream: "__stream_0"
          input_stream: "__stream_1"
          output_stream: "input"
        }
        input_stream: "INPUT_A:__stream_0"
        input_stream: "INPUT_B:__stream_1"
      )pb")));
}

}  // namespace
}  // namespace mediapipe::api2::builder
