#include "mediapipe/framework/api2/stream/segmentation_smoothing.h"

#include "mediapipe/framework/api2/builder.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/formats/image.h"
#include "mediapipe/framework/port/gmock.h"
#include "mediapipe/framework/port/gtest.h"
#include "mediapipe/framework/port/parse_text_proto.h"

namespace mediapipe::api2::builder {
namespace {

using ::mediapipe::Image;

TEST(SegmentationSmoothing, VerifyConfig) {
  Graph graph;

  Stream<Image> mask = graph.In("MASK").Cast<Image>();
  Stream<Image> prev_mask = graph.In("PREV_MASK").Cast<Image>();
  Stream<Image> smoothed_mask = SmoothSegmentationMask(
      mask, prev_mask, /*combine_with_previous_ratio=*/0.1f, graph);
  smoothed_mask.SetName("smoothed_mask");

  EXPECT_THAT(graph.GetConfig(),
              EqualsProto(ParseTextProtoOrDie<CalculatorGraphConfig>(R"pb(
                node {
                  calculator: "SegmentationSmoothingCalculator"
                  input_stream: "MASK:__stream_0"
                  input_stream: "MASK_PREVIOUS:__stream_1"
                  output_stream: "MASK_SMOOTHED:smoothed_mask"
                  options {
                    [mediapipe.SegmentationSmoothingCalculatorOptions.ext] {
                      combine_with_previous_ratio: 0.1
                    }
                  }
                }
                input_stream: "MASK:__stream_0"
                input_stream: "PREV_MASK:__stream_1"
              )pb")));
}

}  // namespace
}  // namespace mediapipe::api2::builder
