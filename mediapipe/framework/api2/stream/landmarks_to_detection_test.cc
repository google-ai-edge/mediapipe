#include "mediapipe/framework/api2/stream/landmarks_to_detection.h"

#include "mediapipe/framework/api2/builder.h"
#include "mediapipe/framework/formats/detection.pb.h"
#include "mediapipe/framework/formats/landmark.pb.h"
#include "mediapipe/framework/port/gmock.h"
#include "mediapipe/framework/port/gtest.h"
#include "mediapipe/framework/port/parse_text_proto.h"
#include "mediapipe/framework/port/status_matchers.h"

namespace mediapipe::api2::builder {
namespace {

TEST(LandmarksToDetection, VerifyConfig) {
  mediapipe::api2::builder::Graph graph;

  Stream<NormalizedLandmarkList> landmarks =
      graph.In("LANDMARKS").Cast<NormalizedLandmarkList>();
  Stream<Detection> detection = ConvertLandmarksToDetection(landmarks, graph);
  detection.SetName("detection");

  EXPECT_THAT(
      graph.GetConfig(),
      EqualsProto(mediapipe::ParseTextProtoOrDie<CalculatorGraphConfig>(R"pb(
        node {
          calculator: "LandmarksToDetectionCalculator"
          input_stream: "NORM_LANDMARKS:__stream_0"
          output_stream: "DETECTION:detection"
        }
        input_stream: "LANDMARKS:__stream_0"
      )pb")));
}

}  // namespace
}  // namespace mediapipe::api2::builder
