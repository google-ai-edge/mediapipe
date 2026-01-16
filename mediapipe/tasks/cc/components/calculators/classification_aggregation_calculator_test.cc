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
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "mediapipe/framework/api2/builder.h"
#include "mediapipe/framework/api2/port.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/formats/classification.pb.h"
#include "mediapipe/framework/output_stream_poller.h"
#include "mediapipe/framework/packet.h"
#include "mediapipe/framework/port/gmock.h"
#include "mediapipe/framework/port/gtest.h"
#include "mediapipe/framework/port/parse_text_proto.h"
#include "mediapipe/framework/port/status_macros.h"
#include "mediapipe/framework/port/status_matchers.h"
#include "mediapipe/framework/timestamp.h"
#include "mediapipe/tasks/cc/components/calculators/classification_aggregation_calculator.pb.h"
#include "mediapipe/tasks/cc/components/containers/proto/classifications.pb.h"
#include "tensorflow/lite/test_util.h"

namespace mediapipe {
namespace {

using ::mediapipe::ParseTextProtoOrDie;
using ::mediapipe::api2::Input;
using ::mediapipe::api2::Output;
using ::mediapipe::api2::builder::Graph;
using ::mediapipe::api2::builder::Source;
using ::mediapipe::tasks::components::containers::proto::ClassificationResult;
using ::testing::Pointwise;

constexpr char kClassificationInput0Tag[] = "CLASSIFICATIONS_0";
constexpr char kClassificationInput0Name[] = "classifications_0";
constexpr char kClassificationInput1Tag[] = "CLASSIFICATIONS_1";
constexpr char kClassificationInput1Name[] = "classifications_1";
constexpr char kTimestampsTag[] = "TIMESTAMPS";
constexpr char kTimestampsName[] = "timestamps";
constexpr char kClassificationsTag[] = "CLASSIFICATIONS";
constexpr char kClassificationsName[] = "classifications";
constexpr char kTimestampedClassificationsTag[] = "TIMESTAMPED_CLASSIFICATIONS";
constexpr char kTimestampedClassificationsName[] =
    "timestamped_classifications";

ClassificationList MakeClassificationList(int class_index) {
  return ParseTextProtoOrDie<ClassificationList>(absl::StrFormat(
      R"pb(
        classification { index: %d }
      )pb",
      class_index));
}

class ClassificationAggregationCalculatorTest : public tflite::testing::Test {
 protected:
  absl::StatusOr<OutputStreamPoller> BuildGraph(
      bool connect_timestamps = false) {
    Graph graph;
    auto& calculator = graph.AddNode("ClassificationAggregationCalculator");
    calculator
        .GetOptions<mediapipe::ClassificationAggregationCalculatorOptions>() =
        ParseTextProtoOrDie<
            mediapipe::ClassificationAggregationCalculatorOptions>(
            R"pb(head_names: "foo" head_names: "bar")pb");
    graph[Input<ClassificationList>(kClassificationInput0Tag)].SetName(
        kClassificationInput0Name) >>
        calculator.In(absl::StrFormat("%s:%d", kClassificationsTag, 0));
    graph[Input<ClassificationList>(kClassificationInput1Tag)].SetName(
        kClassificationInput1Name) >>
        calculator.In(absl::StrFormat("%s:%d", kClassificationsTag, 1));
    if (connect_timestamps) {
      graph[Input<std::vector<Timestamp>>(kTimestampsTag)].SetName(
          kTimestampsName) >>
          calculator.In(kTimestampsTag);
      calculator.Out(kTimestampedClassificationsTag)
              .SetName(kTimestampedClassificationsName) >>
          graph[Output<std::vector<ClassificationResult>>(
              kTimestampedClassificationsTag)];
    } else {
      calculator.Out(kClassificationsTag).SetName(kClassificationsName) >>
          graph[Output<ClassificationResult>(kClassificationsTag)];
    }

    MP_RETURN_IF_ERROR(calculator_graph_.Initialize(graph.GetConfig()));
    if (connect_timestamps) {
      MP_ASSIGN_OR_RETURN(auto poller, calculator_graph_.AddOutputStreamPoller(
                                           kTimestampedClassificationsName));
      MP_RETURN_IF_ERROR(calculator_graph_.StartRun(/*extra_side_packets=*/{}));
      return poller;
    }
    MP_ASSIGN_OR_RETURN(auto poller, calculator_graph_.AddOutputStreamPoller(
                                         kClassificationsName));
    MP_RETURN_IF_ERROR(calculator_graph_.StartRun(/*extra_side_packets=*/{}));
    return poller;
  }

  absl::Status Send(
      std::vector<ClassificationList> classifications, int timestamp = 0,
      std::optional<std::vector<int>> aggregation_timestamps = std::nullopt) {
    MP_RETURN_IF_ERROR(calculator_graph_.AddPacketToInputStream(
        kClassificationInput0Name,
        MakePacket<ClassificationList>(classifications[0])
            .At(Timestamp(timestamp))));
    MP_RETURN_IF_ERROR(calculator_graph_.AddPacketToInputStream(
        kClassificationInput1Name,
        MakePacket<ClassificationList>(classifications[1])
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

TEST_F(ClassificationAggregationCalculatorTest, SucceedsWithoutAggregation) {
  MP_ASSERT_OK_AND_ASSIGN(auto poller, BuildGraph());
  MP_ASSERT_OK(Send({MakeClassificationList(0), MakeClassificationList(1)}));
  MP_ASSERT_OK_AND_ASSIGN(auto result, GetResult<ClassificationResult>(poller));

  EXPECT_THAT(result,
              EqualsProto(ParseTextProtoOrDie<ClassificationResult>(
                  R"pb(timestamp_ms: 0,
                       classifications {
                         head_index: 0
                         head_name: "foo"
                         classification_list { classification { index: 0 } }
                       }
                       classifications {
                         head_index: 1
                         head_name: "bar"
                         classification_list { classification { index: 1 } }
                       })pb")));
}

TEST_F(ClassificationAggregationCalculatorTest, SucceedsWithAggregation) {
  MP_ASSERT_OK_AND_ASSIGN(auto poller, BuildGraph(/*connect_timestamps=*/true));
  MP_ASSERT_OK(Send({MakeClassificationList(0), MakeClassificationList(1)}));
  MP_ASSERT_OK(Send(
      {MakeClassificationList(2), MakeClassificationList(3)},
      /*timestamp=*/1000,
      /*aggregation_timestamps=*/std::optional<std::vector<int>>({0, 1000})));
  MP_ASSERT_OK_AND_ASSIGN(auto result,
                          GetResult<std::vector<ClassificationResult>>(poller));

  EXPECT_THAT(result,
              Pointwise(EqualsProto(),
                        {ParseTextProtoOrDie<ClassificationResult>(R"pb(
                           timestamp_ms: 0,
                           classifications {
                             head_index: 0
                             head_name: "foo"
                             classification_list { classification { index: 0 } }
                           }
                           classifications {
                             head_index: 1
                             head_name: "bar"
                             classification_list { classification { index: 1 } }
                           }
                         )pb"),
                         ParseTextProtoOrDie<ClassificationResult>(R"pb(
                           timestamp_ms: 1,
                           classifications {
                             head_index: 0
                             head_name: "foo"
                             classification_list { classification { index: 2 } }
                           }
                           classifications {
                             head_index: 1
                             head_name: "bar"
                             classification_list { classification { index: 3 } }
                           }
                         )pb")}));
}

}  // namespace
}  // namespace mediapipe
