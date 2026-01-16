/* Copyright 2022 The MediaPipe Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include <memory>
#include <optional>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "mediapipe/framework/api2/builder.h"
#include "mediapipe/framework/api2/port.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/output_stream_poller.h"
#include "mediapipe/framework/packet.h"
#include "mediapipe/framework/port/gmock.h"
#include "mediapipe/framework/port/gtest.h"
#include "mediapipe/framework/port/parse_text_proto.h"
#include "mediapipe/framework/port/status_macros.h"
#include "mediapipe/framework/port/status_matchers.h"
#include "mediapipe/framework/timestamp.h"
#include "mediapipe/tasks/cc/components/containers/proto/embeddings.pb.h"
#include "tensorflow/lite/test_util.h"

namespace mediapipe {
namespace {

using ::mediapipe::ParseTextProtoOrDie;
using ::mediapipe::api2::Input;
using ::mediapipe::api2::Output;
using ::mediapipe::api2::builder::Graph;
using ::mediapipe::api2::builder::Source;
using ::mediapipe::tasks::components::containers::proto::EmbeddingResult;
using ::testing::Pointwise;

constexpr char kEmbeddingsTag[] = "EMBEDDINGS";
constexpr char kEmbeddingsInName[] = "embeddings_in";
constexpr char kEmbeddingsOutName[] = "embeddings_out";
constexpr char kTimestampsTag[] = "TIMESTAMPS";
constexpr char kTimestampsName[] = "timestamps_in";
constexpr char kTimestampedEmbeddingsTag[] = "TIMESTAMPED_EMBEDDINGS";
constexpr char kTimestampedEmbeddingsName[] = "timestamped_embeddings_out";

class EmbeddingAggregationCalculatorTest : public tflite::testing::Test {
 protected:
  absl::StatusOr<OutputStreamPoller> BuildGraph(bool connect_timestamps) {
    Graph graph;
    auto& calculator = graph.AddNode("EmbeddingAggregationCalculator");
    graph[Input<EmbeddingResult>(kEmbeddingsTag)].SetName(kEmbeddingsInName) >>
        calculator.In(kEmbeddingsTag);
    if (connect_timestamps) {
      graph[Input<std::vector<Timestamp>>(kTimestampsTag)].SetName(
          kTimestampsName) >>
          calculator.In(kTimestampsTag);
      calculator.Out(kTimestampedEmbeddingsTag)
              .SetName(kTimestampedEmbeddingsName) >>
          graph[Output<std::vector<EmbeddingResult>>(
              kTimestampedEmbeddingsTag)];
    } else {
      calculator.Out(kEmbeddingsTag).SetName(kEmbeddingsOutName) >>
          graph[Output<EmbeddingResult>(kEmbeddingsTag)];
    }

    MP_RETURN_IF_ERROR(calculator_graph_.Initialize(graph.GetConfig()));
    if (connect_timestamps) {
      MP_ASSIGN_OR_RETURN(auto poller, calculator_graph_.AddOutputStreamPoller(
                                           kTimestampedEmbeddingsName));
      MP_RETURN_IF_ERROR(calculator_graph_.StartRun(/*extra_side_packets=*/{}));
      return poller;
    }
    MP_ASSIGN_OR_RETURN(auto poller, calculator_graph_.AddOutputStreamPoller(
                                         kEmbeddingsOutName));
    MP_RETURN_IF_ERROR(calculator_graph_.StartRun(/*extra_side_packets=*/{}));
    return poller;
  }

  absl::Status Send(
      const EmbeddingResult& embeddings, int timestamp = 0,
      std::optional<std::vector<int>> aggregation_timestamps = std::nullopt) {
    MP_RETURN_IF_ERROR(calculator_graph_.AddPacketToInputStream(
        kEmbeddingsInName, MakePacket<EmbeddingResult>(std::move(embeddings))
                               .At(Timestamp(timestamp))));
    if (aggregation_timestamps.has_value()) {
      auto packet = std::make_unique<std::vector<Timestamp>>();
      for (const auto& timestamp : *aggregation_timestamps) {
        packet->emplace_back(Timestamp(timestamp));
      }
      MP_RETURN_IF_ERROR(calculator_graph_.AddPacketToInputStream(
          kTimestampsName, Adopt(packet.release()).At(Timestamp(timestamp))));
    }
    return absl::OkStatus();
  }

  template <typename T>
  absl::StatusOr<T> GetResult(OutputStreamPoller& poller) {
    MP_RETURN_IF_ERROR(calculator_graph_.WaitUntilIdle());
    MP_RETURN_IF_ERROR(calculator_graph_.CloseAllInputStreams());

    Packet packet;
    if (!poller.Next(&packet)) {
      return absl::InternalError("Unable to get output packet");
    }
    auto result = packet.Get<T>();
    MP_RETURN_IF_ERROR(calculator_graph_.WaitUntilDone());
    return result;
  }

 private:
  CalculatorGraph calculator_graph_;
};

TEST_F(EmbeddingAggregationCalculatorTest, SucceedsWithoutAggregation) {
  EmbeddingResult embedding = ParseTextProtoOrDie<EmbeddingResult>(
      R"pb(embeddings { head_index: 0 })pb");

  MP_ASSERT_OK_AND_ASSIGN(auto poller,
                          BuildGraph(/*connect_timestamps=*/false));
  MP_ASSERT_OK(Send(embedding));
  MP_ASSERT_OK_AND_ASSIGN(auto result, GetResult<EmbeddingResult>(poller));

  EXPECT_THAT(result, EqualsProto(ParseTextProtoOrDie<EmbeddingResult>(
                          R"pb(timestamp_ms: 0
                               embeddings { head_index: 0 })pb")));
}

TEST_F(EmbeddingAggregationCalculatorTest, SucceedsWithAggregation) {
  MP_ASSERT_OK_AND_ASSIGN(auto poller, BuildGraph(/*connect_timestamps=*/true));
  MP_ASSERT_OK(Send(ParseTextProtoOrDie<EmbeddingResult>(R"pb(embeddings {
                                                                head_index: 0
                                                              })pb")));
  MP_ASSERT_OK(Send(
      ParseTextProtoOrDie<EmbeddingResult>(
          R"pb(embeddings { head_index: 1 })pb"),
      /*timestamp=*/1000,
      /*aggregation_timestamps=*/std::optional<std::vector<int>>({0, 1000})));
  MP_ASSERT_OK_AND_ASSIGN(auto results,
                          GetResult<std::vector<EmbeddingResult>>(poller));

  EXPECT_THAT(results,
              Pointwise(EqualsProto(), {ParseTextProtoOrDie<EmbeddingResult>(
                                            R"pb(embeddings { head_index: 0 }
                                                 timestamp_ms: 0)pb"),
                                        ParseTextProtoOrDie<EmbeddingResult>(
                                            R"pb(embeddings { head_index: 1 }
                                                 timestamp_ms: 1)pb")}));
}

}  // namespace
}  // namespace mediapipe
