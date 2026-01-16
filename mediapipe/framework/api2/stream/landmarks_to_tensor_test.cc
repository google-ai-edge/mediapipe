#include "mediapipe/framework/api2/stream/landmarks_to_tensor.h"

#include <utility>
#include <vector>

#include "mediapipe/calculators/tensor/landmarks_to_tensor_calculator.pb.h"
#include "mediapipe/framework/api2/builder.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/formats/landmark.pb.h"
#include "mediapipe/framework/formats/tensor.h"
#include "mediapipe/framework/port/gmock.h"
#include "mediapipe/framework/port/gtest.h"
#include "mediapipe/framework/port/parse_text_proto.h"
#include "mediapipe/framework/port/status_matchers.h"

namespace mediapipe::api2::builder {
namespace {

TEST(ConvertLandmarksToTensor, ConvertLandmarksToTensor) {
  Graph graph;

  Stream<LandmarkList> landmarks = graph.In("LANDMARKS").Cast<LandmarkList>();
  Stream<std::vector<Tensor>> tensors =
      ConvertLandmarksToTensor(landmarks,
                               {LandmarksToTensorCalculatorOptions::X,
                                LandmarksToTensorCalculatorOptions::Y,
                                LandmarksToTensorCalculatorOptions::Z},
                               /*flatten=*/true, graph);
  tensors.SetName("tensors");

  EXPECT_THAT(graph.GetConfig(),
              EqualsProto(ParseTextProtoOrDie<CalculatorGraphConfig>(R"pb(
                node {
                  calculator: "LandmarksToTensorCalculator"
                  input_stream: "LANDMARKS:__stream_0"
                  output_stream: "TENSORS:tensors"
                  options {
                    [mediapipe.LandmarksToTensorCalculatorOptions.ext] {
                      attributes: [ X, Y, Z ]
                      flatten: true
                    }
                  }
                }
                input_stream: "LANDMARKS:__stream_0"
              )pb")));

  CalculatorGraph calcualtor_graph;
  MP_EXPECT_OK(calcualtor_graph.Initialize(graph.GetConfig()));
}

TEST(ConvertLandmarksToTensor, ConvertNormalizedLandmarksToTensor) {
  Graph graph;

  Stream<NormalizedLandmarkList> landmarks =
      graph.In("LANDMARKS").Cast<NormalizedLandmarkList>();
  Stream<std::pair<int, int>> image_size =
      graph.In("IMAGE_SIZE").Cast<std::pair<int, int>>();
  Stream<std::vector<Tensor>> tensors = ConvertNormalizedLandmarksToTensor(
      landmarks, image_size,
      {LandmarksToTensorCalculatorOptions::X,
       LandmarksToTensorCalculatorOptions::Y,
       LandmarksToTensorCalculatorOptions::Z},
      /*flatten=*/false, graph);
  tensors.SetName("tensors");

  EXPECT_THAT(graph.GetConfig(),
              EqualsProto(ParseTextProtoOrDie<CalculatorGraphConfig>(R"pb(
                node {
                  calculator: "LandmarksToTensorCalculator"
                  input_stream: "IMAGE_SIZE:__stream_0"
                  input_stream: "NORM_LANDMARKS:__stream_1"
                  output_stream: "TENSORS:tensors"
                  options {
                    [mediapipe.LandmarksToTensorCalculatorOptions.ext] {
                      attributes: [ X, Y, Z ]
                      flatten: false
                    }
                  }
                }
                input_stream: "IMAGE_SIZE:__stream_0"
                input_stream: "LANDMARKS:__stream_1"
              )pb")));

  CalculatorGraph calcualtor_graph;
  MP_EXPECT_OK(calcualtor_graph.Initialize(graph.GetConfig()));
}

}  // namespace
}  // namespace mediapipe::api2::builder
