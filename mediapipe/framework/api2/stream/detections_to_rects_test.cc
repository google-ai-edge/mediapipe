#include "mediapipe/framework/api2/stream/detections_to_rects.h"

#include <utility>
#include <vector>

#include "mediapipe/framework/api2/builder.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/formats/detection.pb.h"
#include "mediapipe/framework/formats/rect.pb.h"
#include "mediapipe/framework/port/gmock.h"
#include "mediapipe/framework/port/gtest.h"
#include "mediapipe/framework/port/parse_text_proto.h"
#include "mediapipe/framework/port/status_matchers.h"

namespace mediapipe::api2::builder {
namespace {

TEST(DetectionsToRects, ConvertAlignmentPointsDetectionToRect) {
  mediapipe::api2::builder::Graph graph;

  Stream<Detection> detection = graph.In("DETECTION").Cast<Detection>();
  detection.SetName("detection");
  Stream<std::pair<int, int>> size =
      graph.In("SIZE").Cast<std::pair<int, int>>();
  size.SetName("size");
  Stream<NormalizedRect> rect = ConvertAlignmentPointsDetectionToRect(
      detection, size, /*start_keypoint_index=*/0, /*end_keypoint_index=*/100,
      /*target_angle=*/200, graph);
  rect.SetName("rect");

  EXPECT_THAT(
      graph.GetConfig(),
      EqualsProto(mediapipe::ParseTextProtoOrDie<CalculatorGraphConfig>(R"pb(
        node {
          calculator: "AlignmentPointsRectsCalculator"
          input_stream: "DETECTION:detection"
          input_stream: "IMAGE_SIZE:size"
          output_stream: "NORM_RECT:rect"
          options {
            [mediapipe.DetectionsToRectsCalculatorOptions.ext] {
              rotation_vector_start_keypoint_index: 0
              rotation_vector_end_keypoint_index: 100
              rotation_vector_target_angle_degrees: 200
            }
          }
        }
        input_stream: "DETECTION:detection"
        input_stream: "SIZE:size"
      )pb")));

  CalculatorGraph calcualtor_graph;
  MP_EXPECT_OK(calcualtor_graph.Initialize(graph.GetConfig()));
}

TEST(DetectionsToRects, ConvertAlignmentPointsDetectionsToRect) {
  mediapipe::api2::builder::Graph graph;

  Stream<std::vector<Detection>> detections =
      graph.In("DETECTIONS").Cast<std::vector<Detection>>();
  detections.SetName("detections");
  Stream<std::pair<int, int>> size =
      graph.In("SIZE").Cast<std::pair<int, int>>();
  size.SetName("size");
  Stream<NormalizedRect> rect = ConvertAlignmentPointsDetectionsToRect(
      detections, size, /*start_keypoint_index=*/0, /*end_keypoint_index=*/100,
      /*target_angle=*/200, graph);
  rect.SetName("rect");

  EXPECT_THAT(
      graph.GetConfig(),
      EqualsProto(mediapipe::ParseTextProtoOrDie<CalculatorGraphConfig>(R"pb(
        node {
          calculator: "AlignmentPointsRectsCalculator"
          input_stream: "DETECTIONS:detections"
          input_stream: "IMAGE_SIZE:size"
          output_stream: "NORM_RECT:rect"
          options {
            [mediapipe.DetectionsToRectsCalculatorOptions.ext] {
              rotation_vector_start_keypoint_index: 0
              rotation_vector_end_keypoint_index: 100
              rotation_vector_target_angle_degrees: 200
            }
          }
        }
        input_stream: "DETECTIONS:detections"
        input_stream: "SIZE:size"
      )pb")));

  CalculatorGraph calcualtor_graph;
  MP_EXPECT_OK(calcualtor_graph.Initialize(graph.GetConfig()));
}

TEST(DetectionsToRects, ConvertDetectionToRect) {
  mediapipe::api2::builder::Graph graph;

  Stream<Detection> detection = graph.In("DETECTION").Cast<Detection>();
  detection.SetName("detection");
  Stream<std::pair<int, int>> size =
      graph.In("SIZE").Cast<std::pair<int, int>>();
  size.SetName("size");
  Stream<NormalizedRect> rect = ConvertDetectionToRect(
      detection, size, /*start_keypoint_index=*/0, /*end_keypoint_index=*/100,
      /*target_angle=*/200, graph);
  rect.SetName("rect");

  EXPECT_THAT(
      graph.GetConfig(),
      EqualsProto(mediapipe::ParseTextProtoOrDie<CalculatorGraphConfig>(R"pb(
        node {
          calculator: "DetectionsToRectsCalculator"
          input_stream: "DETECTION:detection"
          input_stream: "IMAGE_SIZE:size"
          output_stream: "NORM_RECT:rect"
          options {
            [mediapipe.DetectionsToRectsCalculatorOptions.ext] {
              rotation_vector_start_keypoint_index: 0
              rotation_vector_end_keypoint_index: 100
              rotation_vector_target_angle_degrees: 200
            }
          }
        }
        input_stream: "DETECTION:detection"
        input_stream: "SIZE:size"
      )pb")));

  CalculatorGraph calcualtor_graph;
  MP_EXPECT_OK(calcualtor_graph.Initialize(graph.GetConfig()));
}

TEST(DetectionsToRects, ConvertDetectionsToRects) {
  mediapipe::api2::builder::Graph graph;

  Stream<std::vector<Detection>> detections =
      graph.In("DETECTIONS").Cast<std::vector<Detection>>();
  detections.SetName("detections");
  Stream<std::pair<int, int>> size =
      graph.In("SIZE").Cast<std::pair<int, int>>();
  size.SetName("size");
  Stream<std::vector<NormalizedRect>> rects = ConvertDetectionsToRects(
      detections, size, /*start_keypoint_index=*/0, /*end_keypoint_index=*/100,
      /*target_angle=*/200, graph);
  rects.SetName("rects");

  EXPECT_THAT(
      graph.GetConfig(),
      EqualsProto(mediapipe::ParseTextProtoOrDie<CalculatorGraphConfig>(R"pb(
        node {
          calculator: "DetectionsToRectsCalculator"
          input_stream: "DETECTIONS:detections"
          input_stream: "IMAGE_SIZE:size"
          output_stream: "NORM_RECTS:rects"
          options {
            [mediapipe.DetectionsToRectsCalculatorOptions.ext] {
              rotation_vector_start_keypoint_index: 0
              rotation_vector_end_keypoint_index: 100
              rotation_vector_target_angle_degrees: 200
            }
          }
        }
        input_stream: "DETECTIONS:detections"
        input_stream: "SIZE:size"
      )pb")));

  CalculatorGraph calcualtor_graph;
  MP_EXPECT_OK(calcualtor_graph.Initialize(graph.GetConfig()));
}

TEST(DetectionsToRects, ConvertDetectionsToRectUsingKeypoints) {
  mediapipe::api2::builder::Graph graph;

  Stream<std::vector<Detection>> detections =
      graph.In("DETECTIONS").Cast<std::vector<Detection>>();
  detections.SetName("detections");
  Stream<std::pair<int, int>> size =
      graph.In("SIZE").Cast<std::pair<int, int>>();
  size.SetName("size");
  Stream<NormalizedRect> rect = ConvertDetectionsToRectUsingKeypoints(
      detections, size, /*start_keypoint_index=*/0, /*end_keypoint_index=*/100,
      /*target_angle=*/200, graph);
  rect.SetName("rect");

  EXPECT_THAT(
      graph.GetConfig(),
      EqualsProto(mediapipe::ParseTextProtoOrDie<CalculatorGraphConfig>(R"pb(
        node {
          calculator: "DetectionsToRectsCalculator"
          input_stream: "DETECTIONS:detections"
          input_stream: "IMAGE_SIZE:size"
          output_stream: "NORM_RECT:rect"
          options {
            [mediapipe.DetectionsToRectsCalculatorOptions.ext] {
              rotation_vector_start_keypoint_index: 0
              rotation_vector_end_keypoint_index: 100
              rotation_vector_target_angle_degrees: 200
              conversion_mode: USE_KEYPOINTS
            }
          }
        }
        input_stream: "DETECTIONS:detections"
        input_stream: "SIZE:size"
      )pb")));

  CalculatorGraph calcualtor_graph;
  MP_EXPECT_OK(calcualtor_graph.Initialize(graph.GetConfig()));
}

}  // namespace
}  // namespace mediapipe::api2::builder
