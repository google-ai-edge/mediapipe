#include "mediapipe/framework/api2/stream/rect_transformation.h"

#include <utility>
#include <vector>

#include "mediapipe/framework/api2/builder.h"
#include "mediapipe/framework/formats/rect.pb.h"
#include "mediapipe/framework/port/gmock.h"
#include "mediapipe/framework/port/gtest.h"
#include "mediapipe/framework/port/parse_text_proto.h"

namespace mediapipe::api2::builder {

namespace {

using ::mediapipe::NormalizedRect;

TEST(RectTransformation, ScaleAndMakeSquare) {
  mediapipe::api2::builder::Graph graph;

  Stream<NormalizedRect> rect = graph.In("RECT").Cast<NormalizedRect>();
  Stream<std::pair<int, int>> size =
      graph.In("SIZE").Cast<std::pair<int, int>>();
  Stream<NormalizedRect> transformed_rect = ScaleAndMakeSquare(
      rect, size, /*scale_x_factor=*/2, /*scale_y_factor=*/7, graph);
  transformed_rect.SetName("transformed_rect");

  EXPECT_THAT(
      graph.GetConfig(),
      EqualsProto(mediapipe::ParseTextProtoOrDie<CalculatorGraphConfig>(R"pb(
        node {
          calculator: "RectTransformationCalculator"
          input_stream: "IMAGE_SIZE:__stream_1"
          input_stream: "NORM_RECT:__stream_0"
          output_stream: "transformed_rect"
          options {
            [mediapipe.RectTransformationCalculatorOptions.ext] {
              scale_x: 2
              scale_y: 7
              square_long: true
            }
          }
        }
        input_stream: "RECT:__stream_0"
        input_stream: "SIZE:__stream_1"
      )pb")));
}

TEST(RectTransformation, Scale) {
  mediapipe::api2::builder::Graph graph;

  Stream<NormalizedRect> rect = graph.In("RECT").Cast<NormalizedRect>();
  Stream<std::pair<int, int>> size =
      graph.In("SIZE").Cast<std::pair<int, int>>();
  Stream<NormalizedRect> transformed_rect =
      Scale(rect, size, /*scale_x_factor=*/2, /*scale_y_factor=*/7, graph);
  transformed_rect.SetName("transformed_rect");

  EXPECT_THAT(
      graph.GetConfig(),
      EqualsProto(mediapipe::ParseTextProtoOrDie<CalculatorGraphConfig>(R"pb(
        node {
          calculator: "RectTransformationCalculator"
          input_stream: "IMAGE_SIZE:__stream_1"
          input_stream: "NORM_RECT:__stream_0"
          output_stream: "transformed_rect"
          options {
            [mediapipe.RectTransformationCalculatorOptions.ext] {
              scale_x: 2
              scale_y: 7
            }
          }
        }
        input_stream: "RECT:__stream_0"
        input_stream: "SIZE:__stream_1"
      )pb")));
}

TEST(RectTransformation, ScaleAndShift) {
  mediapipe::api2::builder::Graph graph;

  Stream<NormalizedRect> rect = graph.In("RECT").Cast<NormalizedRect>();
  Stream<std::pair<int, int>> size =
      graph.In("SIZE").Cast<std::pair<int, int>>();
  Stream<NormalizedRect> transformed_rect =
      ScaleAndShift(rect, size, /*scale_x_factor=*/2, /*scale_y_factor=*/7,
                    /*shift_x=*/10, /*shift_y=*/0.5f, graph);
  transformed_rect.SetName("transformed_rect");

  EXPECT_THAT(
      graph.GetConfig(),
      EqualsProto(mediapipe::ParseTextProtoOrDie<CalculatorGraphConfig>(R"pb(
        node {
          calculator: "RectTransformationCalculator"
          input_stream: "IMAGE_SIZE:__stream_1"
          input_stream: "NORM_RECT:__stream_0"
          output_stream: "transformed_rect"
          options {
            [mediapipe.RectTransformationCalculatorOptions.ext] {
              scale_x: 2
              scale_y: 7
              shift_x: 10
              shift_y: 0.5
            }
          }
        }
        input_stream: "RECT:__stream_0"
        input_stream: "SIZE:__stream_1"
      )pb")));
}

TEST(RectTransformation, ScaleAndShiftAndMakeSquareLong) {
  mediapipe::api2::builder::Graph graph;

  Stream<NormalizedRect> rect = graph.In("RECT").Cast<NormalizedRect>();
  Stream<std::pair<int, int>> size =
      graph.In("SIZE").Cast<std::pair<int, int>>();
  Stream<NormalizedRect> transformed_rect = ScaleAndShiftAndMakeSquareLong(
      rect, size, /*scale_x_factor=*/2, /*scale_y_factor=*/7,
      /*shift_x=*/10, /*shift_y=*/0.5f, graph);
  transformed_rect.SetName("transformed_rect");

  EXPECT_THAT(
      graph.GetConfig(),
      EqualsProto(mediapipe::ParseTextProtoOrDie<CalculatorGraphConfig>(R"pb(
        node {
          calculator: "RectTransformationCalculator"
          input_stream: "IMAGE_SIZE:__stream_1"
          input_stream: "NORM_RECT:__stream_0"
          output_stream: "transformed_rect"
          options {
            [mediapipe.RectTransformationCalculatorOptions.ext] {
              scale_x: 2
              scale_y: 7
              shift_x: 10
              shift_y: 0.5
              square_long: true
            }
          }
        }
        input_stream: "RECT:__stream_0"
        input_stream: "SIZE:__stream_1"
      )pb")));
}

TEST(RectTransformation, ScaleAndShiftMultipleRects) {
  mediapipe::api2::builder::Graph graph;

  Stream<std::vector<NormalizedRect>> rects =
      graph.In("RECTS").Cast<std::vector<NormalizedRect>>();
  Stream<std::pair<int, int>> size =
      graph.In("SIZE").Cast<std::pair<int, int>>();
  Stream<std::vector<NormalizedRect>> transformed_rects =
      ScaleAndShift(rects, size, /*scale_x_factor=*/2, /*scale_y_factor=*/7,
                    /*shift_x=*/10, /*shift_y=*/0.5f, graph);
  transformed_rects.SetName("transformed_rects");

  EXPECT_THAT(
      graph.GetConfig(),
      EqualsProto(mediapipe::ParseTextProtoOrDie<CalculatorGraphConfig>(R"pb(
        node {
          calculator: "RectTransformationCalculator"
          input_stream: "IMAGE_SIZE:__stream_1"
          input_stream: "NORM_RECTS:__stream_0"
          output_stream: "transformed_rects"
          options {
            [mediapipe.RectTransformationCalculatorOptions.ext] {
              scale_x: 2
              scale_y: 7
              shift_x: 10
              shift_y: 0.5
            }
          }
        }
        input_stream: "RECTS:__stream_0"
        input_stream: "SIZE:__stream_1"
      )pb")));
}

TEST(RectTransformation, ScaleAndShiftAndMakeSquareLongMultipleRects) {
  mediapipe::api2::builder::Graph graph;

  Stream<std::vector<NormalizedRect>> rects =
      graph.In("RECTS").Cast<std::vector<NormalizedRect>>();
  Stream<std::pair<int, int>> size =
      graph.In("SIZE").Cast<std::pair<int, int>>();
  Stream<std::vector<NormalizedRect>> transformed_rects =
      ScaleAndShiftAndMakeSquareLong(rects, size, /*scale_x_factor=*/2,
                                     /*scale_y_factor=*/7,
                                     /*shift_x=*/10, /*shift_y=*/0.5f, graph);
  transformed_rects.SetName("transformed_rects");

  EXPECT_THAT(
      graph.GetConfig(),
      EqualsProto(mediapipe::ParseTextProtoOrDie<CalculatorGraphConfig>(R"pb(
        node {
          calculator: "RectTransformationCalculator"
          input_stream: "IMAGE_SIZE:__stream_1"
          input_stream: "NORM_RECTS:__stream_0"
          output_stream: "transformed_rects"
          options {
            [mediapipe.RectTransformationCalculatorOptions.ext] {
              scale_x: 2
              scale_y: 7
              shift_x: 10
              shift_y: 0.5
              square_long: true
            }
          }
        }
        input_stream: "RECTS:__stream_0"
        input_stream: "SIZE:__stream_1"
      )pb")));
}

}  // namespace
}  // namespace mediapipe::api2::builder
