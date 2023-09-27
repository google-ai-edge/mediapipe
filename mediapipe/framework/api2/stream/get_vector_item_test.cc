#include "mediapipe/framework/api2/stream/get_vector_item.h"

#include <vector>

#include "mediapipe/framework/api2/builder.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/formats/classification.pb.h"
#include "mediapipe/framework/formats/landmark.pb.h"
#include "mediapipe/framework/formats/rect.pb.h"
#include "mediapipe/framework/port/gmock.h"
#include "mediapipe/framework/port/gtest.h"
#include "mediapipe/framework/port/parse_text_proto.h"
#include "mediapipe/framework/port/status_matchers.h"

namespace mediapipe::api2::builder {
namespace {

using ::mediapipe::api2::builder::Graph;

TEST(GetItem, GetNormalizedLandmarkListVectorItem) {
  Graph graph;
  Stream<std::vector<NormalizedLandmarkList>> items =
      graph.In("ITEMS").Cast<std::vector<NormalizedLandmarkList>>();
  Stream<int> idx = graph.In("IDX").Cast<int>();
  Stream<NormalizedLandmarkList> item = GetItem(items, idx, graph);
  item.SetName("item");
  EXPECT_THAT(graph.GetConfig(),
              EqualsProto(ParseTextProtoOrDie<CalculatorGraphConfig>(R"pb(
                node {
                  calculator: "GetNormalizedLandmarkListVectorItemCalculator"
                  input_stream: "INDEX:__stream_0"
                  input_stream: "VECTOR:__stream_1"
                  output_stream: "ITEM:item"
                }
                input_stream: "IDX:__stream_0"
                input_stream: "ITEMS:__stream_1"
              )pb")));
  CalculatorGraph calculator_graph;
  MP_EXPECT_OK(calculator_graph.Initialize(graph.GetConfig()));
}

TEST(GetItem, GetLandmarkListVectorItem) {
  Graph graph;
  Stream<std::vector<LandmarkList>> items =
      graph.In("ITEMS").Cast<std::vector<LandmarkList>>();
  Stream<int> idx = graph.In("IDX").Cast<int>();
  Stream<LandmarkList> item = GetItem(items, idx, graph);
  item.SetName("item");
  EXPECT_THAT(graph.GetConfig(),
              EqualsProto(ParseTextProtoOrDie<CalculatorGraphConfig>(R"pb(
                node {
                  calculator: "GetLandmarkListVectorItemCalculator"
                  input_stream: "INDEX:__stream_0"
                  input_stream: "VECTOR:__stream_1"
                  output_stream: "ITEM:item"
                }
                input_stream: "IDX:__stream_0"
                input_stream: "ITEMS:__stream_1"
              )pb")));
  CalculatorGraph calculator_graph;
  MP_EXPECT_OK(calculator_graph.Initialize(graph.GetConfig()));
}

TEST(GetItem, GetClassificationListVectorItem) {
  Graph graph;
  Stream<std::vector<ClassificationList>> items =
      graph.In("ITEMS").Cast<std::vector<ClassificationList>>();
  Stream<int> idx = graph.In("IDX").Cast<int>();
  Stream<ClassificationList> item = GetItem(items, idx, graph);
  item.SetName("item");
  EXPECT_THAT(graph.GetConfig(),
              EqualsProto(ParseTextProtoOrDie<CalculatorGraphConfig>(R"pb(
                node {
                  calculator: "GetClassificationListVectorItemCalculator"
                  input_stream: "INDEX:__stream_0"
                  input_stream: "VECTOR:__stream_1"
                  output_stream: "ITEM:item"
                }
                input_stream: "IDX:__stream_0"
                input_stream: "ITEMS:__stream_1"
              )pb")));
  CalculatorGraph calculator_graph;
  MP_EXPECT_OK(calculator_graph.Initialize(graph.GetConfig()));
}

TEST(GetItem, GetNormalizedRectVectorItem) {
  Graph graph;
  Stream<std::vector<NormalizedRect>> items =
      graph.In("ITEMS").Cast<std::vector<NormalizedRect>>();
  Stream<int> idx = graph.In("IDX").Cast<int>();
  Stream<NormalizedRect> item = GetItem(items, idx, graph);
  item.SetName("item");
  EXPECT_THAT(graph.GetConfig(),
              EqualsProto(ParseTextProtoOrDie<CalculatorGraphConfig>(R"pb(
                node {
                  calculator: "GetNormalizedRectVectorItemCalculator"
                  input_stream: "INDEX:__stream_0"
                  input_stream: "VECTOR:__stream_1"
                  output_stream: "ITEM:item"
                }
                input_stream: "IDX:__stream_0"
                input_stream: "ITEMS:__stream_1"
              )pb")));
  CalculatorGraph calculator_graph;
  MP_EXPECT_OK(calculator_graph.Initialize(graph.GetConfig()));
}

TEST(GetItem, GetRectVectorItem) {
  Graph graph;
  Stream<std::vector<Rect>> items = graph.In("ITEMS").Cast<std::vector<Rect>>();
  Stream<int> idx = graph.In("IDX").Cast<int>();
  Stream<Rect> item = GetItem(items, idx, graph);
  item.SetName("item");
  EXPECT_THAT(graph.GetConfig(),
              EqualsProto(ParseTextProtoOrDie<CalculatorGraphConfig>(R"pb(
                node {
                  calculator: "GetRectVectorItemCalculator"
                  input_stream: "INDEX:__stream_0"
                  input_stream: "VECTOR:__stream_1"
                  output_stream: "ITEM:item"
                }
                input_stream: "IDX:__stream_0"
                input_stream: "ITEMS:__stream_1"
              )pb")));
  CalculatorGraph calculator_graph;
  MP_EXPECT_OK(calculator_graph.Initialize(graph.GetConfig()));
}

}  // namespace
}  // namespace mediapipe::api2::builder
