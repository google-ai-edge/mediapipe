#include "mediapipe/framework/api2/stream/split.h"

#include <cstdint>
#include <utility>
#include <vector>

#include "mediapipe/framework/api2/builder.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/formats/landmark.pb.h"
#include "mediapipe/framework/formats/rect.pb.h"
#include "mediapipe/framework/formats/tensor.h"
#include "mediapipe/framework/port/gmock.h"
#include "mediapipe/framework/port/gtest.h"
#include "mediapipe/framework/port/parse_text_proto.h"
#include "mediapipe/framework/port/status_matchers.h"

namespace mediapipe::api2::builder {
namespace {

TEST(SplitTest, SplitToRanges2Ranges) {
  Graph graph;
  Stream<std::vector<mediapipe::Tensor>> tensors =
      graph.In("TENSORS").Cast<std::vector<mediapipe::Tensor>>();
  std::vector<Stream<std::vector<mediapipe::Tensor>>> result =
      SplitToRanges(tensors, {{0, 1}, {1, 2}}, graph);
  EXPECT_EQ(result.size(), 2);
  EXPECT_THAT(
      graph.GetConfig(),
      EqualsProto(mediapipe::ParseTextProtoOrDie<CalculatorGraphConfig>(R"pb(
        node {
          calculator: "SplitTensorVectorCalculator"
          input_stream: "__stream_0"
          output_stream: "__stream_1"
          output_stream: "__stream_2"
          options {
            [mediapipe.SplitVectorCalculatorOptions.ext] {
              ranges { begin: 0 end: 1 }
              ranges { begin: 1 end: 2 }
            }
          }
        }
        input_stream: "TENSORS:__stream_0"
      )pb")));

  CalculatorGraph calculator_graph;
  MP_EXPECT_OK(calculator_graph.Initialize(graph.GetConfig()));
}

TEST(SplitTest, Split2Items) {
  Graph graph;
  Stream<std::vector<mediapipe::Tensor>> tensors =
      graph.In("TENSORS").Cast<std::vector<mediapipe::Tensor>>();
  std::vector<Stream<mediapipe::Tensor>> result = Split(tensors, {0, 1}, graph);
  EXPECT_EQ(result.size(), 2);
  EXPECT_THAT(
      graph.GetConfig(),
      EqualsProto(mediapipe::ParseTextProtoOrDie<CalculatorGraphConfig>(R"pb(
        node {
          calculator: "SplitTensorVectorCalculator"
          input_stream: "__stream_0"
          output_stream: "__stream_1"
          output_stream: "__stream_2"
          options {
            [mediapipe.SplitVectorCalculatorOptions.ext] {
              ranges { begin: 0 end: 1 }
              ranges { begin: 1 end: 2 }
              element_only: true
            }
          }
        }
        input_stream: "TENSORS:__stream_0"
      )pb")));

  CalculatorGraph calculator_graph;
  MP_EXPECT_OK(calculator_graph.Initialize(graph.GetConfig()));
}

TEST(SplitTest, Split2Uint64tItems) {
  Graph graph;
  Stream<std::vector<uint64_t>> ids =
      graph.In("IDS").Cast<std::vector<uint64_t>>();
  std::vector<Stream<uint64_t>> result = Split(ids, {0, 1}, graph);
  EXPECT_EQ(result.size(), 2);
  EXPECT_THAT(
      graph.GetConfig(),
      EqualsProto(mediapipe::ParseTextProtoOrDie<CalculatorGraphConfig>(R"pb(
        node {
          calculator: "SplitUint64tVectorCalculator"
          input_stream: "__stream_0"
          output_stream: "__stream_1"
          output_stream: "__stream_2"
          options {
            [mediapipe.SplitVectorCalculatorOptions.ext] {
              ranges { begin: 0 end: 1 }
              ranges { begin: 1 end: 2 }
              element_only: true
            }
          }
        }
        input_stream: "IDS:__stream_0"
      )pb")));

  CalculatorGraph calculator_graph;
  MP_EXPECT_OK(calculator_graph.Initialize(graph.GetConfig()));
}

TEST(SplitTest, SplitToRanges5Ranges) {
  Graph graph;
  Stream<std::vector<NormalizedRect>> tensors =
      graph.In("RECTS").Cast<std::vector<NormalizedRect>>();
  std::vector<Stream<std::vector<NormalizedRect>>> result =
      SplitToRanges(tensors, {{0, 1}, {1, 2}, {2, 3}, {3, 4}, {4, 5}}, graph);
  EXPECT_EQ(result.size(), 5);
  EXPECT_THAT(
      graph.GetConfig(),
      EqualsProto(mediapipe::ParseTextProtoOrDie<CalculatorGraphConfig>(R"pb(
        node {
          calculator: "SplitNormalizedRectVectorCalculator"
          input_stream: "__stream_0"
          output_stream: "__stream_1"
          output_stream: "__stream_2"
          output_stream: "__stream_3"
          output_stream: "__stream_4"
          output_stream: "__stream_5"
          options {
            [mediapipe.SplitVectorCalculatorOptions.ext] {
              ranges { begin: 0 end: 1 }
              ranges { begin: 1 end: 2 }
              ranges { begin: 2 end: 3 }
              ranges { begin: 3 end: 4 }
              ranges { begin: 4 end: 5 }
            }
          }
        }
        input_stream: "RECTS:__stream_0"
      )pb")));

  CalculatorGraph calculator_graph;
  MP_EXPECT_OK(calculator_graph.Initialize(graph.GetConfig()));
}

TEST(SplitTest, Split5Items) {
  Graph graph;
  Stream<std::vector<NormalizedRect>> tensors =
      graph.In("RECTS").Cast<std::vector<NormalizedRect>>();
  std::vector<Stream<NormalizedRect>> result =
      Split(tensors, {0, 1, 2, 3, 4}, graph);
  EXPECT_EQ(result.size(), 5);
  EXPECT_THAT(
      graph.GetConfig(),
      EqualsProto(mediapipe::ParseTextProtoOrDie<CalculatorGraphConfig>(R"pb(
        node {
          calculator: "SplitNormalizedRectVectorCalculator"
          input_stream: "__stream_0"
          output_stream: "__stream_1"
          output_stream: "__stream_2"
          output_stream: "__stream_3"
          output_stream: "__stream_4"
          output_stream: "__stream_5"
          options {
            [mediapipe.SplitVectorCalculatorOptions.ext] {
              ranges { begin: 0 end: 1 }
              ranges { begin: 1 end: 2 }
              ranges { begin: 2 end: 3 }
              ranges { begin: 3 end: 4 }
              ranges { begin: 4 end: 5 }
              element_only: true
            }
          }
        }
        input_stream: "RECTS:__stream_0"
      )pb")));

  CalculatorGraph calculator_graph;
  MP_EXPECT_OK(calculator_graph.Initialize(graph.GetConfig()));
}

TEST(SplitTest, SplitPassingVectorIndices) {
  Graph graph;
  Stream<std::vector<NormalizedRect>> tensors =
      graph.In("RECTS").Cast<std::vector<NormalizedRect>>();
  std::vector<int> indices = {250, 300};
  std::vector<Stream<NormalizedRect>> second_split_result =
      Split(tensors, indices, graph);
  EXPECT_EQ(second_split_result.size(), 2);
  EXPECT_THAT(
      graph.GetConfig(),
      EqualsProto(mediapipe::ParseTextProtoOrDie<CalculatorGraphConfig>(R"pb(
        node {
          calculator: "SplitNormalizedRectVectorCalculator"
          input_stream: "__stream_0"
          output_stream: "__stream_1"
          output_stream: "__stream_2"
          options {
            [mediapipe.SplitVectorCalculatorOptions.ext] {
              ranges { begin: 250 end: 251 }
              ranges { begin: 300 end: 301 }
              element_only: true
            }
          }
        }
        input_stream: "RECTS:__stream_0"
      )pb")));

  CalculatorGraph calculator_graph;
  MP_EXPECT_OK(calculator_graph.Initialize(graph.GetConfig()));
}

TEST(SplitTest, SplitToRangesPassingVectorRanges) {
  Graph graph;
  Stream<std::vector<NormalizedRect>> tensors =
      graph.In("RECTS").Cast<std::vector<NormalizedRect>>();
  std::vector<std::pair<int, int>> indices = {{250, 255}, {300, 301}};
  std::vector<Stream<std::vector<NormalizedRect>>> second_split_result =
      SplitToRanges(tensors, indices, graph);
  EXPECT_EQ(second_split_result.size(), 2);
  EXPECT_THAT(
      graph.GetConfig(),
      EqualsProto(mediapipe::ParseTextProtoOrDie<CalculatorGraphConfig>(R"pb(
        node {
          calculator: "SplitNormalizedRectVectorCalculator"
          input_stream: "__stream_0"
          output_stream: "__stream_1"
          output_stream: "__stream_2"
          options {
            [mediapipe.SplitVectorCalculatorOptions.ext] {
              ranges { begin: 250 end: 255 }
              ranges { begin: 300 end: 301 }
            }
          }
        }
        input_stream: "RECTS:__stream_0"
      )pb")));

  CalculatorGraph calculator_graph;
  MP_EXPECT_OK(calculator_graph.Initialize(graph.GetConfig()));
}

TEST(SplitTest, SplitToRangesNormalizedLandmarkList) {
  Graph graph;
  Stream<NormalizedLandmarkList> tensors =
      graph.In("LM_LIST").Cast<NormalizedLandmarkList>();
  std::vector<std::pair<int, int>> indices = {{250, 255}, {300, 301}};
  std::vector<Stream<NormalizedLandmarkList>> second_split_result =
      SplitToRanges(tensors, indices, graph);
  EXPECT_EQ(second_split_result.size(), 2);
  EXPECT_THAT(
      graph.GetConfig(),
      EqualsProto(mediapipe::ParseTextProtoOrDie<CalculatorGraphConfig>(R"pb(
        node {
          calculator: "SplitNormalizedLandmarkListCalculator"
          input_stream: "__stream_0"
          output_stream: "__stream_1"
          output_stream: "__stream_2"
          options {
            [mediapipe.SplitVectorCalculatorOptions.ext] {
              ranges { begin: 250 end: 255 }
              ranges { begin: 300 end: 301 }
            }
          }
        }
        input_stream: "LM_LIST:__stream_0"
      )pb")));

  CalculatorGraph calculator_graph;
  MP_EXPECT_OK(calculator_graph.Initialize(graph.GetConfig()));
}

TEST(SplitTest, SplitNormalizedLandmarkList) {
  Graph graph;
  Stream<NormalizedLandmarkList> tensors =
      graph.In("LM_LIST").Cast<NormalizedLandmarkList>();
  std::vector<Stream<NormalizedLandmark>> second_split_result =
      Split(tensors, {250, 300}, graph);
  EXPECT_EQ(second_split_result.size(), 2);
  EXPECT_THAT(
      graph.GetConfig(),
      EqualsProto(mediapipe::ParseTextProtoOrDie<CalculatorGraphConfig>(R"pb(
        node {
          calculator: "SplitNormalizedLandmarkListCalculator"
          input_stream: "__stream_0"
          output_stream: "__stream_1"
          output_stream: "__stream_2"
          options {
            [mediapipe.SplitVectorCalculatorOptions.ext] {
              ranges { begin: 250 end: 251 }
              ranges { begin: 300 end: 301 }
              element_only: true
            }
          }
        }
        input_stream: "LM_LIST:__stream_0"
      )pb")));

  CalculatorGraph calculator_graph;
  MP_EXPECT_OK(calculator_graph.Initialize(graph.GetConfig()));
}

TEST(SplitTest, SplitAndCombineRanges) {
  Graph graph;
  Stream<std::vector<mediapipe::Tensor>> tensors =
      graph.In("TENSORS").Cast<std::vector<mediapipe::Tensor>>();
  Stream<std::vector<mediapipe::Tensor>> result =
      SplitAndCombine(tensors, {{0, 1}, {2, 5}, {70, 75}}, graph);
  result.SetName("tensors_split_and_combined");
  EXPECT_THAT(
      graph.GetConfig(),
      EqualsProto(mediapipe::ParseTextProtoOrDie<CalculatorGraphConfig>(R"pb(
        node {
          calculator: "SplitTensorVectorCalculator"
          input_stream: "__stream_0"
          output_stream: "tensors_split_and_combined"
          options {
            [mediapipe.SplitVectorCalculatorOptions.ext] {
              ranges { begin: 0 end: 1 }
              ranges { begin: 2 end: 5 }
              ranges { begin: 70 end: 75 }
              combine_outputs: true
            }
          }
        }
        input_stream: "TENSORS:__stream_0"
      )pb")));

  CalculatorGraph calculator_graph;
  MP_EXPECT_OK(calculator_graph.Initialize(graph.GetConfig()));
}

TEST(SplitTest, SplitAndCombineIndividualIndices) {
  Graph graph;
  Stream<std::vector<mediapipe::Tensor>> tensors =
      graph.In("TENSORS").Cast<std::vector<mediapipe::Tensor>>();
  Stream<std::vector<mediapipe::Tensor>> result =
      SplitAndCombine(tensors, {0, 2, 70, 100}, graph);
  result.SetName("tensors_split_and_combined");
  EXPECT_THAT(
      graph.GetConfig(),
      EqualsProto(mediapipe::ParseTextProtoOrDie<CalculatorGraphConfig>(R"pb(
        node {
          calculator: "SplitTensorVectorCalculator"
          input_stream: "__stream_0"
          output_stream: "tensors_split_and_combined"
          options {
            [mediapipe.SplitVectorCalculatorOptions.ext] {
              ranges { begin: 0 end: 1 }
              ranges { begin: 2 end: 3 }
              ranges { begin: 70 end: 71 }
              ranges { begin: 100 end: 101 }
              combine_outputs: true
            }
          }
        }
        input_stream: "TENSORS:__stream_0"
      )pb")));

  CalculatorGraph calculator_graph;
  MP_EXPECT_OK(calculator_graph.Initialize(graph.GetConfig()));
}

TEST(SplitTest, SplitAndCombineLandmarkList) {
  Graph graph;
  Stream<LandmarkList> tensors = graph.In("LM_LIST").Cast<LandmarkList>();
  std::vector<std::pair<int, int>> ranges = {{250, 255}, {300, 301}};
  Stream<LandmarkList> landmark_list = SplitAndCombine(tensors, ranges, graph);
  landmark_list.SetName("landmark_list");
  EXPECT_THAT(
      graph.GetConfig(),
      EqualsProto(mediapipe::ParseTextProtoOrDie<CalculatorGraphConfig>(R"pb(
        node {
          calculator: "SplitLandmarkListCalculator"
          input_stream: "__stream_0"
          output_stream: "landmark_list"
          options {
            [mediapipe.SplitVectorCalculatorOptions.ext] {
              ranges { begin: 250 end: 255 }
              ranges { begin: 300 end: 301 }
              combine_outputs: true
            }
          }
        }
        input_stream: "LM_LIST:__stream_0"
      )pb")));

  CalculatorGraph calculator_graph;
  MP_EXPECT_OK(calculator_graph.Initialize(graph.GetConfig()));
}

TEST(SplitTest, SplitAndCombineLandmarkListIndividualIndices) {
  Graph graph;
  Stream<LandmarkList> tensors = graph.In("LM_LIST").Cast<LandmarkList>();
  std::vector<int> indices = {250, 300};
  Stream<LandmarkList> landmark_list = SplitAndCombine(tensors, indices, graph);
  landmark_list.SetName("landmark_list");
  EXPECT_THAT(
      graph.GetConfig(),
      EqualsProto(mediapipe::ParseTextProtoOrDie<CalculatorGraphConfig>(R"pb(
        node {
          calculator: "SplitLandmarkListCalculator"
          input_stream: "__stream_0"
          output_stream: "landmark_list"
          options {
            [mediapipe.SplitVectorCalculatorOptions.ext] {
              ranges { begin: 250 end: 251 }
              ranges { begin: 300 end: 301 }
              combine_outputs: true
            }
          }
        }
        input_stream: "LM_LIST:__stream_0"
      )pb")));

  CalculatorGraph calculator_graph;
  MP_EXPECT_OK(calculator_graph.Initialize(graph.GetConfig()));
}

TEST(SplitTest, SplitAndCombineJointList) {
  Graph graph;
  Stream<JointList> tensors = graph.In("JT_LIST").Cast<JointList>();
  std::vector<std::pair<int, int>> ranges = {{250, 255}, {300, 301}};
  Stream<JointList> joint_list = SplitAndCombine(tensors, ranges, graph);
  joint_list.SetName("joint_list");
  EXPECT_THAT(
      graph.GetConfig(),
      EqualsProto(mediapipe::ParseTextProtoOrDie<CalculatorGraphConfig>(R"pb(
        node {
          calculator: "SplitJointListCalculator"
          input_stream: "__stream_0"
          output_stream: "joint_list"
          options {
            [mediapipe.SplitVectorCalculatorOptions.ext] {
              ranges { begin: 250 end: 255 }
              ranges { begin: 300 end: 301 }
              combine_outputs: true
            }
          }
        }
        input_stream: "JT_LIST:__stream_0"
      )pb")));

  CalculatorGraph calculator_graph;
  MP_EXPECT_OK(calculator_graph.Initialize(graph.GetConfig()));
}

TEST(SplitTest, SplitAndCombineJointListIndividualIndices) {
  Graph graph;
  Stream<JointList> tensors = graph.In("LM_LIST").Cast<JointList>();
  std::vector<int> indices = {250, 300};
  Stream<JointList> joint_list = SplitAndCombine(tensors, indices, graph);
  joint_list.SetName("joint_list");
  EXPECT_THAT(
      graph.GetConfig(),
      EqualsProto(mediapipe::ParseTextProtoOrDie<CalculatorGraphConfig>(R"pb(
        node {
          calculator: "SplitJointListCalculator"
          input_stream: "__stream_0"
          output_stream: "joint_list"
          options {
            [mediapipe.SplitVectorCalculatorOptions.ext] {
              ranges { begin: 250 end: 251 }
              ranges { begin: 300 end: 301 }
              combine_outputs: true
            }
          }
        }
        input_stream: "LM_LIST:__stream_0"
      )pb")));

  CalculatorGraph calculator_graph;
  MP_EXPECT_OK(calculator_graph.Initialize(graph.GetConfig()));
}

}  // namespace
}  // namespace mediapipe::api2::builder
