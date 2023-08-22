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

#include <optional>
#include <utility>
#include <vector>

#include "absl/types/span.h"
#include "mediapipe/calculators/util/landmarks_to_render_data_calculator.pb.h"
#include "mediapipe/calculators/util/rect_to_render_data_calculator.pb.h"
#include "mediapipe/calculators/util/rect_to_render_scale_calculator.pb.h"
#include "mediapipe/framework/api2/builder.h"
#include "mediapipe/framework/formats/image.h"
#include "mediapipe/framework/formats/landmark.pb.h"
#include "mediapipe/framework/formats/rect.pb.h"
#include "mediapipe/util/render_data.pb.h"

namespace mediapipe::tasks::vision::utils {

using ::mediapipe::api2::builder::Graph;
using ::mediapipe::api2::builder::Stream;

Stream<Image> Render(Stream<Image> image,
                     absl::Span<Stream<mediapipe::RenderData>> render_data_list,
                     Graph& graph) {
  auto& annotation_overlay = graph.AddNode("AnnotationOverlayCalculator");
  image >> annotation_overlay.In("UIMAGE");
  for (int i = 0; i < render_data_list.size(); ++i) {
    render_data_list[i] >> annotation_overlay.In(i);
  }
  return annotation_overlay.Out("UIMAGE").Cast<Image>();
}

Stream<mediapipe::RenderData> RenderLandmarks(
    Stream<mediapipe::NormalizedLandmarkList> landmarks,
    std::optional<api2::builder::Stream<float>> render_scale,
    const mediapipe::LandmarksToRenderDataCalculatorOptions& renderer_options,
    Graph& graph) {
  auto& landmarks_render = graph.AddNode("LandmarksToRenderDataCalculator");
  landmarks_render
      .GetOptions<mediapipe::LandmarksToRenderDataCalculatorOptions>()
      .CopyFrom(renderer_options);
  landmarks >> landmarks_render.In("NORM_LANDMARKS");
  if (render_scale.has_value()) {
    *render_scale >> landmarks_render.In("RENDER_SCALE");
  }
  auto render_data = landmarks_render.Out("RENDER_DATA");
  return render_data.Cast<mediapipe::RenderData>();
}

Stream<float> GetRenderScale(Stream<std::pair<int, int>> image_size,
                             Stream<NormalizedRect> roi, float multiplier,
                             Graph& graph) {
  auto& to_render_scale = graph.AddNode("RectToRenderScaleCalculator");
  to_render_scale.GetOptions<mediapipe::RectToRenderScaleCalculatorOptions>()
      .set_multiplier(multiplier);
  roi >> to_render_scale.In("NORM_RECT");
  image_size >> to_render_scale.In("IMAGE_SIZE");
  return to_render_scale.Out("RENDER_SCALE").Cast<float>();
}

Stream<mediapipe::RenderData> RenderRect(
    Stream<NormalizedRect> rect,
    const mediapipe::RectToRenderDataCalculatorOptions& renderer_options,
    Graph& graph) {
  auto& rect_render = graph.AddNode("RectToRenderDataCalculator");
  rect_render.GetOptions<mediapipe::RectToRenderDataCalculatorOptions>()
      .CopyFrom(renderer_options);
  rect >> rect_render.In("NORM_RECT");
  auto render_data = rect_render.Out("RENDER_DATA");
  return render_data.Cast<mediapipe::RenderData>();
}

}  // namespace mediapipe::tasks::vision::utils
