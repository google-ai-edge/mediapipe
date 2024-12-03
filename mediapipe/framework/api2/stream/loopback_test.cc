#include "mediapipe/framework/api2/stream/loopback.h"

#include "mediapipe/framework/api2/builder.h"
#include "mediapipe/framework/api2/node.h"
#include "mediapipe/framework/api2/port.h"
#include "mediapipe/framework/port/gmock.h"
#include "mediapipe/framework/port/gtest.h"
#include "mediapipe/framework/port/parse_text_proto.h"
#include "mediapipe/framework/port/status_matchers.h"

namespace mediapipe::api2::builder {
namespace {

class TestDataProducer : public NodeIntf {
 public:
  static constexpr Input<float> kLoopbackData{"LOOPBACK_DATA"};
  static constexpr Output<float> kProducedData{"PRODUCED_DATA"};
  MEDIAPIPE_NODE_INTERFACE(TestDataProducer, kLoopbackData, kProducedData);
};

TEST(LoopbackTest, GetLoopbackData) {
  Graph graph;

  Stream<int> tick = graph.In("TICK").Cast<int>();

  auto [data, set_loopback_data_fn] = GetLoopbackData<float>(tick, graph);

  auto& producer = graph.AddNode<TestDataProducer>();
  data.ConnectTo(producer[TestDataProducer::kLoopbackData]);
  Stream<float> data_to_loopback(producer[TestDataProducer::kProducedData]);

  set_loopback_data_fn(data_to_loopback);

  // PreviousLoopbackCalculator configuration is incorrect here and should be
  // updated when corresponding b/175887687 is fixed.
  // Use mediapipe::aimatter::GraphBuilder to fix back edges in the graph.
  EXPECT_THAT(
      graph.GetConfig(),
      EqualsProto(mediapipe::ParseTextProtoOrDie<CalculatorGraphConfig>(R"pb(
        node {
          calculator: "PreviousLoopbackCalculator"
          input_stream: "LOOP:__stream_2"
          input_stream: "MAIN:__stream_0"
          output_stream: "PREV_LOOP:__stream_1"
        }
        node {
          calculator: "TestDataProducer"
          input_stream: "LOOPBACK_DATA:__stream_1"
          output_stream: "PRODUCED_DATA:__stream_2"
        }
        input_stream: "TICK:__stream_0"
      )pb")));
}

}  // namespace
}  // namespace mediapipe::api2::builder
