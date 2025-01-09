#include "mediapipe/framework/api2/stream/concatenate.h"

#include <vector>

#include "mediapipe/framework/api2/builder.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/formats/body_rig.pb.h"
#include "mediapipe/framework/formats/landmark.pb.h"
#include "mediapipe/framework/formats/tensor.h"
#include "mediapipe/framework/port/gmock.h"
#include "mediapipe/framework/port/gtest.h"
#include "mediapipe/framework/port/parse_text_proto.h"
#include "mediapipe/framework/port/status_matchers.h"

namespace mediapipe::api2::builder {
namespace {

TEST(Concatenate, ConcatenateLandmarkList) {
  Graph graph;
  std::vector<Stream<LandmarkList>> items = {
      graph.In("LMK_LIST")[0].Cast<LandmarkList>(),
      graph.In("LMK_LIST")[1].Cast<LandmarkList>()};
  Stream<LandmarkList> landmark_list = Concatenate(items, graph);
  landmark_list.SetName("landmark_list");
  EXPECT_THAT(graph.GetConfig(),
              EqualsProto(ParseTextProtoOrDie<CalculatorGraphConfig>(R"pb(
                node {
                  calculator: "ConcatenateLandmarkListCalculator"
                  input_stream: "__stream_0"
                  input_stream: "__stream_1"
                  output_stream: "landmark_list"
                  options {
                    [mediapipe.ConcatenateVectorCalculatorOptions.ext] {
                      only_emit_if_all_present: false
                    }
                  }
                }
                input_stream: "LMK_LIST:0:__stream_0"
                input_stream: "LMK_LIST:1:__stream_1"
              )pb")));

  CalculatorGraph calcualtor_graph;
  MP_EXPECT_OK(calcualtor_graph.Initialize(graph.GetConfig()));
}

TEST(Concatenate, ConcatenateLandmarkList_IfAllPresent) {
  Graph graph;
  std::vector<Stream<LandmarkList>> items = {
      graph.In("LMK_LIST")[0].Cast<LandmarkList>(),
      graph.In("LMK_LIST")[1].Cast<LandmarkList>()};
  Stream<LandmarkList> landmark_list = ConcatenateIfAllPresent(items, graph);
  landmark_list.SetName("landmark_list");
  EXPECT_THAT(graph.GetConfig(),
              EqualsProto(ParseTextProtoOrDie<CalculatorGraphConfig>(R"pb(
                node {
                  calculator: "ConcatenateLandmarkListCalculator"
                  input_stream: "__stream_0"
                  input_stream: "__stream_1"
                  output_stream: "landmark_list"
                  options {
                    [mediapipe.ConcatenateVectorCalculatorOptions.ext] {
                      only_emit_if_all_present: true
                    }
                  }
                }
                input_stream: "LMK_LIST:0:__stream_0"
                input_stream: "LMK_LIST:1:__stream_1"
              )pb")));

  CalculatorGraph calcualtor_graph;
  MP_EXPECT_OK(calcualtor_graph.Initialize(graph.GetConfig()));
}

TEST(Concatenate, ConcatenateJointList) {
  Graph graph;
  std::vector<Stream<JointList>> items = {
      graph.In("JT_LIST")[0].Cast<JointList>(),
      graph.In("JT_LIST")[1].Cast<JointList>()};
  Stream<JointList> joint_list = Concatenate(items, graph);
  joint_list.SetName("joint_list");
  EXPECT_THAT(graph.GetConfig(),
              EqualsProto(ParseTextProtoOrDie<CalculatorGraphConfig>(R"pb(
                node {
                  calculator: "ConcatenateJointListCalculator"
                  input_stream: "__stream_0"
                  input_stream: "__stream_1"
                  output_stream: "joint_list"
                  options {
                    [mediapipe.ConcatenateVectorCalculatorOptions.ext] {
                      only_emit_if_all_present: false
                    }
                  }
                }
                input_stream: "JT_LIST:0:__stream_0"
                input_stream: "JT_LIST:1:__stream_1"
              )pb")));

  CalculatorGraph calcualtor_graph;
  MP_EXPECT_OK(calcualtor_graph.Initialize(graph.GetConfig()));
}

TEST(Concatenate, ConcatenateJointList_IfAllPresent) {
  Graph graph;
  std::vector<Stream<JointList>> items = {
      graph.In("JT_LIST")[0].Cast<JointList>(),
      graph.In("JT_LIST")[1].Cast<JointList>()};
  Stream<JointList> joint_list = ConcatenateIfAllPresent(items, graph);
  joint_list.SetName("joint_list");
  EXPECT_THAT(graph.GetConfig(),
              EqualsProto(ParseTextProtoOrDie<CalculatorGraphConfig>(R"pb(
                node {
                  calculator: "ConcatenateJointListCalculator"
                  input_stream: "__stream_0"
                  input_stream: "__stream_1"
                  output_stream: "joint_list"
                  options {
                    [mediapipe.ConcatenateVectorCalculatorOptions.ext] {
                      only_emit_if_all_present: true
                    }
                  }
                }
                input_stream: "JT_LIST:0:__stream_0"
                input_stream: "JT_LIST:1:__stream_1"
              )pb")));

  CalculatorGraph calcualtor_graph;
  MP_EXPECT_OK(calcualtor_graph.Initialize(graph.GetConfig()));
}

TEST(Concatenate, ConcatenateTensorVectorList) {
  Graph graph;
  std::vector<Stream<std::vector<Tensor>>> items = {
      graph.In("VT_LIST")[0].Cast<std::vector<Tensor>>(),
      graph.In("VT_LIST")[1].Cast<std::vector<Tensor>>()};
  Stream<std::vector<Tensor>> tensors = Concatenate(items, graph);
  tensors.SetName("joint_list");
  EXPECT_THAT(graph.GetConfig(),
              EqualsProto(ParseTextProtoOrDie<CalculatorGraphConfig>(R"pb(
                node {
                  calculator: "ConcatenateTensorVectorCalculator"
                  input_stream: "__stream_0"
                  input_stream: "__stream_1"
                  output_stream: "joint_list"
                  options {
                    [mediapipe.ConcatenateVectorCalculatorOptions.ext] {
                      only_emit_if_all_present: false
                    }
                  }
                }
                input_stream: "VT_LIST:0:__stream_0"
                input_stream: "VT_LIST:1:__stream_1"
              )pb")));

  CalculatorGraph calcualtor_graph;
  MP_EXPECT_OK(calcualtor_graph.Initialize(graph.GetConfig()));
}

TEST(Concatenate, ConcatenateTensorVectorList_IfAllPresent) {
  Graph graph;
  std::vector<Stream<std::vector<Tensor>>> items = {
      graph.In("VT_LIST")[0].Cast<std::vector<Tensor>>(),
      graph.In("VT_LIST")[1].Cast<std::vector<Tensor>>()};

  Stream<std::vector<Tensor>> tensors = ConcatenateIfAllPresent(items, graph);
  tensors.SetName("joint_list");
  EXPECT_THAT(graph.GetConfig(),
              EqualsProto(ParseTextProtoOrDie<CalculatorGraphConfig>(R"pb(
                node {
                  calculator: "ConcatenateTensorVectorCalculator"
                  input_stream: "__stream_0"
                  input_stream: "__stream_1"
                  output_stream: "joint_list"
                  options {
                    [mediapipe.ConcatenateVectorCalculatorOptions.ext] {
                      only_emit_if_all_present: true
                    }
                  }
                }
                input_stream: "VT_LIST:0:__stream_0"
                input_stream: "VT_LIST:1:__stream_1"
              )pb")));

  CalculatorGraph calcualtor_graph;
  MP_EXPECT_OK(calcualtor_graph.Initialize(graph.GetConfig()));
}

}  // namespace
}  // namespace mediapipe::api2::builder
