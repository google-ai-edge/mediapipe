#include "mediapipe/framework/api2/stream/smoothing.h"

#include <cstdint>
#include <optional>
#include <utility>
#include <vector>

#include "absl/types/optional.h"
#include "mediapipe/calculators/util/landmarks_smoothing_calculator.pb.h"
#include "mediapipe/framework/api2/builder.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/formats/landmark.pb.h"
#include "mediapipe/framework/formats/rect.pb.h"
#include "mediapipe/framework/port/gmock.h"
#include "mediapipe/framework/port/gtest.h"
#include "mediapipe/framework/port/parse_text_proto.h"
#include "mediapipe/framework/port/status_matchers.h"

namespace mediapipe::api2::builder {
namespace {

TEST(Smoothing, NormLandmarks) {
  mediapipe::api2::builder::Graph graph;

  Stream<NormalizedLandmarkList> norm_landmarks =
      graph.In("NORM_LANDMARKS").Cast<NormalizedLandmarkList>();
  Stream<std::pair<int, int>> image_size =
      graph.In("IMAGE_SIZE").Cast<std::pair<int, int>>();
  Stream<NormalizedRect> scale_roi =
      graph.In("SCALE_ROI").Cast<NormalizedRect>();
  SmoothLandmarks(
      norm_landmarks, image_size, scale_roi,
      {.min_cutoff = 0.5f, .beta = 100.0f, .derivate_cutoff = 20.0f}, graph)
      .SetName("smoothed_norm_landmarks");

  EXPECT_THAT(
      graph.GetConfig(),
      EqualsProto(mediapipe::ParseTextProtoOrDie<CalculatorGraphConfig>(R"pb(
        node {
          calculator: "LandmarksSmoothingCalculator"
          input_stream: "IMAGE_SIZE:__stream_0"
          input_stream: "NORM_LANDMARKS:__stream_1"
          input_stream: "OBJECT_SCALE_ROI:__stream_2"
          output_stream: "NORM_FILTERED_LANDMARKS:smoothed_norm_landmarks"
          options {
            [mediapipe.LandmarksSmoothingCalculatorOptions.ext] {
              one_euro_filter {
                min_cutoff: 0.5
                beta: 100
                derivate_cutoff: 20
                disable_value_scaling: false
              }
            }
          }
        }
        input_stream: "IMAGE_SIZE:__stream_0"
        input_stream: "NORM_LANDMARKS:__stream_1"
        input_stream: "SCALE_ROI:__stream_2"
      )pb")));

  CalculatorGraph calculator_graph;
  MP_EXPECT_OK(calculator_graph.Initialize(graph.GetConfig()));
}

TEST(Smoothing, Landmarks) {
  mediapipe::api2::builder::Graph graph;

  Stream<LandmarkList> landmarks = graph.In("LANDMARKS").Cast<LandmarkList>();
  SmoothLandmarks(landmarks, /*scale_roi=*/std::nullopt,
                  {.min_cutoff = 1.5f, .beta = 90.0f, .derivate_cutoff = 10.0f},
                  graph)
      .SetName("smoothed_landmarks");

  EXPECT_THAT(
      graph.GetConfig(),
      EqualsProto(mediapipe::ParseTextProtoOrDie<CalculatorGraphConfig>(R"pb(
        node {
          calculator: "LandmarksSmoothingCalculator"
          input_stream: "LANDMARKS:__stream_0"
          output_stream: "FILTERED_LANDMARKS:smoothed_landmarks"
          options {
            [mediapipe.LandmarksSmoothingCalculatorOptions.ext] {
              one_euro_filter {
                min_cutoff: 1.5
                beta: 90
                derivate_cutoff: 10
                disable_value_scaling: true
              }
            }
          }
        }
        input_stream: "LANDMARKS:__stream_0"
      )pb")));

  CalculatorGraph calculator_graph;
  MP_EXPECT_OK(calculator_graph.Initialize(graph.GetConfig()));
}

TEST(Smoothing, MultiLandmarks) {
  mediapipe::api2::builder::Graph graph;

  Stream<std::vector<NormalizedLandmarkList>> norm_landmarks =
      graph.In("NORM_LANDMARKS").Cast<std::vector<NormalizedLandmarkList>>();
  Stream<std::vector<int64_t>> tracking_ids =
      graph.In("TRACKING_IDS").Cast<std::vector<int64_t>>();
  Stream<std::pair<int, int>> image_size =
      graph.In("IMAGE_SIZE").Cast<std::pair<int, int>>();
  Stream<std::vector<NormalizedRect>> scale_roi =
      graph.In("SCALE_ROI").Cast<std::vector<NormalizedRect>>();
  auto config = LandmarksSmoothingCalculatorOptions();
  config.mutable_no_filter();
  SmoothMultiLandmarks(norm_landmarks, tracking_ids, image_size, scale_roi,
                       config, graph)
      .SetName("smoothed_norm_landmarks");

  EXPECT_THAT(
      graph.GetConfig(),
      EqualsProto(mediapipe::ParseTextProtoOrDie<CalculatorGraphConfig>(R"pb(
        node {
          calculator: "MultiLandmarksSmoothingCalculator"
          input_stream: "IMAGE_SIZE:__stream_0"
          input_stream: "NORM_LANDMARKS:__stream_1"
          input_stream: "OBJECT_SCALE_ROI:__stream_2"
          input_stream: "TRACKING_IDS:__stream_3"
          output_stream: "NORM_FILTERED_LANDMARKS:smoothed_norm_landmarks"
          options {
            [mediapipe.LandmarksSmoothingCalculatorOptions.ext] { no_filter {} }
          }
        }
        input_stream: "IMAGE_SIZE:__stream_0"
        input_stream: "NORM_LANDMARKS:__stream_1"
        input_stream: "SCALE_ROI:__stream_2"
        input_stream: "TRACKING_IDS:__stream_3"
      )pb")));

  CalculatorGraph calculator_graph;
  MP_EXPECT_OK(calculator_graph.Initialize(graph.GetConfig()));
}

TEST(Smoothing, MultiWorldLandmarks) {
  mediapipe::api2::builder::Graph graph;

  Stream<std::vector<LandmarkList>> landmarks =
      graph.In("LANDMARKS").Cast<std::vector<LandmarkList>>();
  Stream<std::vector<int64_t>> tracking_ids =
      graph.In("TRACKING_IDS").Cast<std::vector<int64_t>>();
  auto config = LandmarksSmoothingCalculatorOptions();
  config.mutable_no_filter();
  SmoothMultiWorldLandmarks(landmarks, tracking_ids, /*scale_roi=*/std::nullopt,
                            config, graph)
      .SetName("smoothed_landmarks");

  EXPECT_THAT(
      graph.GetConfig(),
      EqualsProto(mediapipe::ParseTextProtoOrDie<CalculatorGraphConfig>(R"pb(
        node {
          calculator: "MultiWorldLandmarksSmoothingCalculator"
          input_stream: "LANDMARKS:__stream_0"
          input_stream: "TRACKING_IDS:__stream_1"
          output_stream: "FILTERED_LANDMARKS:smoothed_landmarks"
          options {
            [mediapipe.LandmarksSmoothingCalculatorOptions.ext] { no_filter {} }
          }
        }
        input_stream: "LANDMARKS:__stream_0"
        input_stream: "TRACKING_IDS:__stream_1"
      )pb")));

  CalculatorGraph calculator_graph;
  MP_EXPECT_OK(calculator_graph.Initialize(graph.GetConfig()));
}

TEST(Smoothing, NormLandmarksVisibility) {
  mediapipe::api2::builder::Graph graph;

  Stream<NormalizedLandmarkList> norm_landmarks =
      graph.In("NORM_LANDMARKS").Cast<NormalizedLandmarkList>();
  Stream<NormalizedLandmarkList> smoothed_norm_landmarks =
      SmoothLandmarksVisibility(norm_landmarks, /*low_pass_filter_alpha=*/0.9f,
                                graph);
  smoothed_norm_landmarks.SetName("smoothed_norm_landmarks");
  EXPECT_THAT(
      graph.GetConfig(),
      EqualsProto(mediapipe::ParseTextProtoOrDie<CalculatorGraphConfig>(R"pb(
        node {
          calculator: "VisibilitySmoothingCalculator"
          input_stream: "NORM_LANDMARKS:__stream_0"
          output_stream: "NORM_FILTERED_LANDMARKS:smoothed_norm_landmarks"
          options {
            [mediapipe.VisibilitySmoothingCalculatorOptions.ext] {
              low_pass_filter { alpha: 0.9 }
            }
          }
        }
        input_stream: "NORM_LANDMARKS:__stream_0"
      )pb")));

  CalculatorGraph calculator_graph;
  MP_EXPECT_OK(calculator_graph.Initialize(graph.GetConfig()));
}

TEST(Smoothing, LandmarksVisibility) {
  mediapipe::api2::builder::Graph graph;

  Stream<LandmarkList> landmarks = graph.In("LANDMARKS").Cast<LandmarkList>();
  Stream<LandmarkList> smoothed_landmarks = SmoothLandmarksVisibility(
      landmarks, /*low_pass_filter_alpha=*/0.9f, graph);
  smoothed_landmarks.SetName("smoothed_landmarks");
  EXPECT_THAT(
      graph.GetConfig(),
      EqualsProto(mediapipe::ParseTextProtoOrDie<CalculatorGraphConfig>(R"pb(
        node {
          calculator: "VisibilitySmoothingCalculator"
          input_stream: "LANDMARKS:__stream_0"
          output_stream: "FILTERED_LANDMARKS:smoothed_landmarks"
          options {
            [mediapipe.VisibilitySmoothingCalculatorOptions.ext] {
              low_pass_filter { alpha: 0.9 }
            }
          }
        }
        input_stream: "LANDMARKS:__stream_0"
      )pb")));

  CalculatorGraph calculator_graph;
  MP_EXPECT_OK(calculator_graph.Initialize(graph.GetConfig()));
}

}  // namespace
}  // namespace mediapipe::api2::builder
