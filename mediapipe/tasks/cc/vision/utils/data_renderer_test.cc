/* Copyright 2023 The MediaPipe Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "mediapipe/tasks/cc/vision/utils/data_renderer.h"

#include <utility>
#include <vector>

#include "absl/types/span.h"
#include "mediapipe/framework/api2/builder.h"
#include "mediapipe/framework/calculator.pb.h"
#include "mediapipe/framework/formats/image.h"
#include "mediapipe/framework/formats/landmark.pb.h"
#include "mediapipe/framework/formats/rect.pb.h"
#include "mediapipe/framework/port/gmock.h"
#include "mediapipe/framework/port/gtest.h"
#include "mediapipe/framework/port/parse_text_proto.h"
#include "mediapipe/util/render_data.pb.h"

namespace mediapipe::tasks::vision::utils {
namespace {

using ::mediapipe::CalculatorGraphConfig;
using ::mediapipe::EqualsProto;
using ::mediapipe::NormalizedRect;
using ::mediapipe::api2::builder::Graph;
using ::mediapipe::api2::builder::Stream;

TEST(DataRenderer, Render) {
  Graph graph;
  Stream<Image> image_in = graph.In("IMAGE").Cast<Image>();
  Stream<RenderData> render_data_in =
      graph.In("RENDER_DATA").Cast<RenderData>();
  std::vector<Stream<RenderData>> render_data_list = {render_data_in};
  Stream<Image> image_out =
      Render(image_in, absl::Span<Stream<RenderData>>(render_data_list), graph);
  image_out.SetName("image_out");
  EXPECT_THAT(
      graph.GetConfig(),
      EqualsProto(mediapipe::ParseTextProtoOrDie<CalculatorGraphConfig>(R"pb(
        node {
          calculator: "AnnotationOverlayCalculator"
          input_stream: "__stream_1"
          input_stream: "UIMAGE:__stream_0"
          output_stream: "UIMAGE:image_out"
        }
        input_stream: "IMAGE:__stream_0"
        input_stream: "RENDER_DATA:__stream_1"
      )pb")));
}

TEST(DataRenderer, RenderLandmarks) {
  Graph graph;
  Stream<NormalizedLandmarkList> rect =
      graph.In("NORM_LANDMARKS").Cast<NormalizedLandmarkList>();
  Stream<RenderData> render_data =
      RenderLandmarks(rect, std::nullopt, {}, graph);
  render_data.SetName("render_data");
  EXPECT_THAT(
      graph.GetConfig(),
      EqualsProto(mediapipe::ParseTextProtoOrDie<CalculatorGraphConfig>(R"pb(
        node {
          calculator: "LandmarksToRenderDataCalculator"
          input_stream: "NORM_LANDMARKS:__stream_0"
          output_stream: "RENDER_DATA:render_data"
          options {
            [mediapipe.LandmarksToRenderDataCalculatorOptions.ext] {}
          }
        }
        input_stream: "NORM_LANDMARKS:__stream_0"
      )pb")));
}

TEST(DataRenderer, GetRenderScale) {
  Graph graph;
  Stream<std::pair<int, int>> image_size =
      graph.In("IMAGE_SIZE").Cast<std::pair<int, int>>();
  Stream<NormalizedRect> roi = graph.In("ROI").Cast<NormalizedRect>();
  Stream<float> render_scale = GetRenderScale(image_size, roi, 0.0001, graph);
  render_scale.SetName("render_scale");
  EXPECT_THAT(
      graph.GetConfig(),
      EqualsProto(mediapipe::ParseTextProtoOrDie<CalculatorGraphConfig>(R"pb(
        node {
          calculator: "RectToRenderScaleCalculator"
          input_stream: "IMAGE_SIZE:__stream_0"
          input_stream: "NORM_RECT:__stream_1"
          output_stream: "RENDER_SCALE:render_scale"
          options {
            [mediapipe.RectToRenderScaleCalculatorOptions.ext] {
              multiplier: 0.0001
            }
          }
        }
        input_stream: "IMAGE_SIZE:__stream_0"
        input_stream: "ROI:__stream_1"
      )pb")));
}

TEST(DataRenderer, RenderRect) {
  Graph graph;
  Stream<NormalizedRect> rect = graph.In("NORM_RECT").Cast<NormalizedRect>();
  Stream<RenderData> render_data = RenderRect(rect, {}, graph);
  render_data.SetName("render_data");
  EXPECT_THAT(
      graph.GetConfig(),
      EqualsProto(mediapipe::ParseTextProtoOrDie<CalculatorGraphConfig>(R"pb(
        node {
          calculator: "RectToRenderDataCalculator"
          input_stream: "NORM_RECT:__stream_0"
          output_stream: "RENDER_DATA:render_data"
          options {
            [mediapipe.RectToRenderDataCalculatorOptions.ext] {}
          }
        }
        input_stream: "NORM_RECT:__stream_0"
      )pb")));
}

}  // namespace
}  // namespace mediapipe::tasks::vision::utils
