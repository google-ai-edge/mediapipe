// Copyright 2019 The MediaPipe Authors.
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

#include <memory>
#include <vector>

#include "mediapipe/calculators/tensor/image_to_tensor_calculator.pb.h"
#include "mediapipe/calculators/tensor/inference_calculator.pb.h"
#include "mediapipe/calculators/tensor/tensors_to_detections_calculator.pb.h"
#include "mediapipe/calculators/tflite/ssd_anchors_calculator.pb.h"
#include "mediapipe/calculators/util/non_max_suppression_calculator.pb.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/formats/detection.pb.h"
#include "mediapipe/framework/formats/rect.pb.h"
#include "mediapipe/framework/port/file_helpers.h"
#include "mediapipe/framework/port/gtest.h"
#include "mediapipe/framework/port/parse_text_proto.h"
#include "mediapipe/framework/port/status_matchers.h"
#include "mediapipe/framework/tool/options_util.h"
#include "mediapipe/framework/tool/test_util.h"
#include "mediapipe/gpu/gpu_origin.pb.h"
#include "mediapipe/modules/face_detection/face_detection.pb.h"

#if !defined(__APPLE__) && !__ANDROID__
#include "mediapipe/gpu/gl_app_texture_support.h"
#include "mediapipe/gpu/gl_calculator_helper.h"
#include "mediapipe/gpu/gpu_test_base.h"
#endif  // !defined(__APPLE__) && !__ANDROID__

namespace mediapipe {
namespace {
using mediapipe::FaceDetectionOptions;

// Ensure protobuf registration.
void RegisterProtobufTypes() {
  MakePacket<mediapipe::InferenceCalculatorOptions::Delegate>();
  MakePacket<mediapipe::FaceDetectionOptions>();
  MakePacket<mediapipe::InferenceCalculatorOptions>();

  MakePacket<mediapipe::ImageToTensorCalculatorOptions>();
  MakePacket<mediapipe::SsdAnchorsCalculatorOptions>();
  MakePacket<mediapipe::TensorsToDetectionsCalculatorOptions>();
  MakePacket<mediapipe::NonMaxSuppressionCalculatorOptions>();
}

// Returns a Packet with an ImageFrame showing a face.
Packet TestImageFrame() {
  std::unique_ptr<ImageFrame> input_image = LoadTestPng(
      file::JoinPath(GetTestRootDir(), "mediapipe/objc/testdata/sergey.png"));
  EXPECT_EQ(input_image->Height(), 600);
  return MakePacket<ImageFrame>(std::move(*input_image));
}

// Returns the registered type name for the basic face-detection-graph.
std::string GetFaceDetectionGraphType() { return "FaceDetectionWithoutRoi"; }

// Returns the config from "face_detection_without_roi.pbtxt".
CalculatorGraphConfig GetFaceDetectionGraph() {
  return GraphRegistry().CreateByName("", GetFaceDetectionGraphType()).value();
}

// Returns the config from "face_detection.pbtxt".
CalculatorGraphConfig GetFaceDetectionWithRoiGraph() {
  return GraphRegistry().CreateByName("", "FaceDetection").value();
}

// Returns the config from "face_detection_short_range.pbtxt".
CalculatorGraphConfig GetFaceDetectionShortRangeCpu() {
  CalculatorGraphConfig config =
      GraphRegistry().CreateByName("", "FaceDetectionShortRangeCpu").value();
  return config;
}

// Returns the FaceDetectionOptions from "face_detection_short_range_cpu.pbtxt".
FaceDetectionOptions GetFaceDetectionShortRangeOptions() {
  CalculatorGraphConfig config;
  LoadTestGraph(&config,
                GetTestFilePath("mediapipe/modules/face_detection/"
                                "face_detection_short_range.binarypb"));
  tool::OptionsMap map;
  map.Initialize(config.node(0));
  return map.Get<FaceDetectionOptions>();
}

// Returns the FaceDetectionOptions from "face_detection_full_range_cpu.pbtxt".
FaceDetectionOptions GetFaceDetectionFullRangeOptions() {
  CalculatorGraphConfig config;
  LoadTestGraph(&config, GetTestFilePath("mediapipe/modules/face_detection/"
                                         "face_detection_full_range.binarypb"));
  tool::OptionsMap map;
  map.Initialize(config.node(0));
  return map.Get<FaceDetectionOptions>();
}

// Returns the FaceDetectionOptions needed to enable CPU processing.
FaceDetectionOptions GetCpuOptions() {
  FaceDetectionOptions result;
  result.mutable_delegate()->xnnpack();
  return result;
}

// Returns the FaceDetectionOptions needed to enable GPU processing.
FaceDetectionOptions GetGpuOptions() {
  FaceDetectionOptions result;
  result.set_gpu_origin(mediapipe::GpuOrigin_Mode::GpuOrigin_Mode_TOP_LEFT);
  result.mutable_delegate()->mutable_gpu()->set_use_advanced_gpu_api(true);
  return result;
}

// Returns an example region of interest rectangle.
mediapipe::NormalizedRect GetTestRoi() {
  mediapipe::NormalizedRect result;
  result.set_x_center(0.5);
  result.set_y_center(0.5);
  result.set_width(0.8);
  result.set_height(0.8);
  return result;
}

// Tests for options input and output packets and streams.
class FaceDetectionTest : public ::testing::Test {
 protected:
  void SetUp() override { RegisterProtobufTypes(); }
  void TearDown() override {}
};

TEST_F(FaceDetectionTest, ExpandFaceDetectionShortRangeCpu) {
  CalculatorGraphConfig config = GetFaceDetectionShortRangeCpu();
  Packet frame1 = TestImageFrame();

  CalculatorGraph graph;
  MP_ASSERT_OK(graph.Initialize(config));

  std::vector<Packet> output;
  MP_ASSERT_OK(graph.ObserveOutputStream("detections", [&](const Packet& p) {
    output.push_back(p);
    return absl::OkStatus();
  }));
  MP_ASSERT_OK(graph.StartRun({}));
  MP_ASSERT_OK(
      graph.AddPacketToInputStream("image", frame1.At(Timestamp(20000))));
  MP_ASSERT_OK(graph.CloseAllPacketSources());
  MP_EXPECT_OK(graph.WaitUntilDone());
  ASSERT_EQ(output.size(), 1);
  EXPECT_EQ(output.front().Get<std::vector<mediapipe::Detection>>().size(), 1);
}

TEST_F(FaceDetectionTest, ExpandFaceDetection) {
  CalculatorGraphConfig config = GetFaceDetectionGraph();
  mediapipe::FaceDetectionOptions face_options =
      GetFaceDetectionShortRangeOptions();
  face_options.MergeFrom(GetCpuOptions());
  config.clear_graph_options();
  config.add_graph_options()->PackFrom(face_options);
  Packet frame1 = TestImageFrame();

  CalculatorGraph graph;
  MP_ASSERT_OK(graph.Initialize(config));

  std::vector<Packet> output;
  MP_ASSERT_OK(graph.ObserveOutputStream("detections", [&](const Packet& p) {
    output.push_back(p);
    return absl::OkStatus();
  }));
  MP_ASSERT_OK(graph.StartRun({}));
  MP_ASSERT_OK(
      graph.AddPacketToInputStream("image", frame1.At(Timestamp(20000))));
  MP_ASSERT_OK(graph.CloseAllPacketSources());
  MP_EXPECT_OK(graph.WaitUntilDone());
  ASSERT_EQ(output.size(), 1);
  EXPECT_EQ(output.front().Get<std::vector<mediapipe::Detection>>().size(), 1);
}

TEST_F(FaceDetectionTest, FaceDetectionShortRangeApi) {
  CalculatorGraphConfig config = GetFaceDetectionGraph();
  config.clear_graph_options();
  mediapipe::FaceDetectionOptions face_options =
      GetFaceDetectionShortRangeOptions();
  Subgraph::SubgraphOptions graph_options;
  face_options.MergeFrom(GetCpuOptions());
  graph_options.add_node_options()->PackFrom(face_options);
  Packet frame1 = TestImageFrame();

  CalculatorGraph graph;
  MP_ASSERT_OK(graph.Initialize({config}, {}, {}, GetFaceDetectionGraphType(),
                                &graph_options));

  std::vector<Packet> output;
  MP_ASSERT_OK(graph.ObserveOutputStream("detections", [&](const Packet& p) {
    output.push_back(p);
    return absl::OkStatus();
  }));
  MP_ASSERT_OK(graph.StartRun({}));
  MP_ASSERT_OK(
      graph.AddPacketToInputStream("image", frame1.At(Timestamp(20000))));
  MP_ASSERT_OK(graph.CloseAllPacketSources());
  MP_EXPECT_OK(graph.WaitUntilDone());
  ASSERT_EQ(output.size(), 1);
  EXPECT_EQ(output.front().Get<std::vector<mediapipe::Detection>>().size(), 1);
}

TEST_F(FaceDetectionTest, FaceDetectionWrapperApi) {
  CalculatorGraphConfig config = GetFaceDetectionGraph();
  config.clear_graph_options();
  mediapipe::FaceDetectionOptions face_options =
      GetFaceDetectionShortRangeOptions();
  face_options.MergeFrom(GetCpuOptions());
  Subgraph::SubgraphOptions graph_options;
  graph_options.add_node_options()->PackFrom(face_options);
  Packet frame1 = TestImageFrame();

  CalculatorGraph graph;
  MP_ASSERT_OK(graph.Initialize({config}, {}, {}, GetFaceDetectionGraphType(),
                                &graph_options));

  std::vector<Packet> output;
  MP_ASSERT_OK(graph.ObserveOutputStream("detections", [&](const Packet& p) {
    output.push_back(p);
    return absl::OkStatus();
  }));
  MP_ASSERT_OK(graph.StartRun({}));
  MP_ASSERT_OK(
      graph.AddPacketToInputStream("image", frame1.At(Timestamp(20000))));
  MP_ASSERT_OK(graph.CloseAllPacketSources());
  MP_EXPECT_OK(graph.WaitUntilDone());
  ASSERT_EQ(output.size(), 1);
  EXPECT_EQ(output.front().Get<std::vector<mediapipe::Detection>>().size(), 1);
}

TEST_F(FaceDetectionTest, FaceDetectionFullRangeApi) {
  CalculatorGraphConfig config = GetFaceDetectionGraph();
  config.clear_graph_options();
  mediapipe::FaceDetectionOptions face_options =
      GetFaceDetectionFullRangeOptions();
  Subgraph::SubgraphOptions graph_options;
  face_options.MergeFrom(GetCpuOptions());
  graph_options.add_node_options()->PackFrom(face_options);
  Packet frame1 = TestImageFrame();

  CalculatorGraph graph;
  MP_ASSERT_OK(graph.Initialize({config}, {}, {}, GetFaceDetectionGraphType(),
                                &graph_options));

  std::vector<Packet> output;
  MP_ASSERT_OK(graph.ObserveOutputStream("detections", [&](const Packet& p) {
    output.push_back(p);
    return absl::OkStatus();
  }));
  MP_ASSERT_OK(graph.StartRun({}));
  MP_ASSERT_OK(
      graph.AddPacketToInputStream("image", frame1.At(Timestamp(20000))));
  MP_ASSERT_OK(graph.CloseAllPacketSources());
  MP_EXPECT_OK(graph.WaitUntilDone());
  ASSERT_EQ(output.size(), 1);
  EXPECT_EQ(output.front().Get<std::vector<mediapipe::Detection>>().size(), 1);
}

TEST_F(FaceDetectionTest, FaceDetectionShortRangeByRoiCpu) {
  CalculatorGraphConfig config = GetFaceDetectionWithRoiGraph();
  config.clear_graph_options();
  mediapipe::FaceDetectionOptions face_options =
      GetFaceDetectionShortRangeOptions();
  face_options.MergeFrom(GetCpuOptions());
  Subgraph::SubgraphOptions graph_options;
  graph_options.add_node_options()->PackFrom(face_options);
  Packet frame1 = TestImageFrame();

  CalculatorGraph graph;
  MP_ASSERT_OK(
      graph.Initialize({config}, {}, {}, "FaceDetection", &graph_options));

  std::vector<Packet> output;
  MP_ASSERT_OK(graph.ObserveOutputStream("detections", [&](const Packet& p) {
    output.push_back(p);
    return absl::OkStatus();
  }));
  MP_ASSERT_OK(graph.StartRun({}));
  MP_ASSERT_OK(
      graph.AddPacketToInputStream("image", frame1.At(Timestamp(20000))));
  MP_ASSERT_OK(graph.AddPacketToInputStream(
      "roi", MakePacket<mediapipe::NormalizedRect>(GetTestRoi())
                 .At(Timestamp(20000))));
  MP_ASSERT_OK(graph.CloseAllPacketSources());
  MP_EXPECT_OK(graph.WaitUntilDone());
  ASSERT_EQ(output.size(), 1);
  EXPECT_EQ(output.front().Get<std::vector<mediapipe::Detection>>().size(), 1);
}

// These GpuBuffer tests are disabled on mobile for now.
#if !defined(__APPLE__) && !__ANDROID__

class FaceDetectionGpuTest : public mediapipe::GpuTestBase {
 protected:
  void SetUp() override {}
  void TearDown() override {}

  // Returns a Packet with a GpuBuffer from an ImageFrame.
  Packet GpuBuffer(Packet image_frame) {
    std::unique_ptr<mediapipe::GpuBuffer> gpu_buffer;
    helper_.RunInGlContext([this, &image_frame, &gpu_buffer] {
      auto src = helper_.CreateSourceTexture(image_frame.Get<ImageFrame>());
      gpu_buffer = src.GetFrame<mediapipe::GpuBuffer>();
    });
    return Adopt(gpu_buffer.release());
  }
};

TEST_F(FaceDetectionGpuTest, FaceDetectionFullRangeGpu) {
  CalculatorGraphConfig config = GetFaceDetectionGraph();
  config.clear_graph_options();
  mediapipe::FaceDetectionOptions face_options =
      GetFaceDetectionFullRangeOptions();
  face_options.MergeFrom(GetGpuOptions());

  Subgraph::SubgraphOptions graph_options;
  graph_options.add_node_options()->PackFrom(face_options);
  Packet frame1 = GpuBuffer(TestImageFrame());

  CalculatorGraph graph;
  MP_ASSERT_OK(graph.Initialize({config}, {}, {}, GetFaceDetectionGraphType(),
                                &graph_options));

  MP_ASSERT_OK(mediapipe::SetExternalGlContextForGraph(
      &graph, helper_.GetGlContext().native_context()));
  std::vector<Packet> output;
  MP_ASSERT_OK(graph.ObserveOutputStream("detections", [&](const Packet& p) {
    output.push_back(p);
    return absl::OkStatus();
  }));
  MP_ASSERT_OK(graph.StartRun({}));
  MP_ASSERT_OK(
      graph.AddPacketToInputStream("image", frame1.At(Timestamp(20000))));
  MP_ASSERT_OK(graph.CloseAllPacketSources());
  MP_EXPECT_OK(graph.WaitUntilDone());
  ASSERT_EQ(output.size(), 1);
  EXPECT_EQ(output.front().Get<std::vector<mediapipe::Detection>>().size(), 1);
}

TEST_F(FaceDetectionGpuTest, FaceDetectionShortRangeGpu) {
  CalculatorGraphConfig config = GetFaceDetectionGraph();
  config.clear_graph_options();
  mediapipe::FaceDetectionOptions face_options =
      GetFaceDetectionShortRangeOptions();
  face_options.MergeFrom(GetGpuOptions());

  Subgraph::SubgraphOptions graph_options;
  graph_options.add_node_options()->PackFrom(face_options);
  Packet frame1 = GpuBuffer(TestImageFrame());

  CalculatorGraph graph;
  MP_ASSERT_OK(graph.Initialize({config}, {}, {}, GetFaceDetectionGraphType(),
                                &graph_options));

  MP_ASSERT_OK(mediapipe::SetExternalGlContextForGraph(
      &graph, helper_.GetGlContext().native_context()));
  std::vector<Packet> output;
  MP_ASSERT_OK(graph.ObserveOutputStream("detections", [&](const Packet& p) {
    output.push_back(p);
    return absl::OkStatus();
  }));
  MP_ASSERT_OK(graph.StartRun({}));
  MP_ASSERT_OK(
      graph.AddPacketToInputStream("image", frame1.At(Timestamp(20000))));
  MP_ASSERT_OK(graph.CloseAllPacketSources());
  MP_EXPECT_OK(graph.WaitUntilDone());
  ASSERT_EQ(output.size(), 1);
  EXPECT_EQ(output.front().Get<std::vector<mediapipe::Detection>>().size(), 1);
}

#endif  // #if !defined(__APPLE__) && !__ANDROID__

}  // namespace
}  // namespace mediapipe
