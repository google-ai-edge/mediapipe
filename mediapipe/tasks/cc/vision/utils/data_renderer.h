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

#ifndef MEDIAPIPE_TASKS_CC_VISION_UTILS_DATA_RENDERER_H_
#define MEDIAPIPE_TASKS_CC_VISION_UTILS_DATA_RENDERER_H_

#include <optional>
#include <utility>

#include "absl/types/span.h"
#include "mediapipe/calculators/util/landmarks_to_render_data_calculator.pb.h"
#include "mediapipe/calculators/util/rect_to_render_data_calculator.pb.h"
#include "mediapipe/framework/api2/builder.h"
#include "mediapipe/framework/formats/image.h"
#include "mediapipe/framework/formats/landmark.pb.h"
#include "mediapipe/framework/formats/rect.pb.h"
#include "mediapipe/util/render_data.pb.h"

namespace mediapipe::tasks::vision::utils {

// Adds a node to the provided graph that renders the render_data_list on the
// given image, and returns the rendered image.
api2::builder::Stream<Image> Render(
    api2::builder::Stream<Image> image,
    absl::Span<api2::builder::Stream<mediapipe::RenderData>> render_data_list,
    api2::builder::Graph& graph);

// Adds a node to the provided graph that infers the render scale from the image
// size and the object RoI. It will give you bigger rendered primitives for
// bigger/closer objects and smaller primitives for smaller/far objects. The
// primitives scale is proportional to `roi_size * multiplier`.
//
// See more details in
// mediapipe/calculators/util/rect_to_render_scale_calculator.cc
api2::builder::Stream<float> GetRenderScale(
    api2::builder::Stream<std::pair<int, int>> image_size,
    api2::builder::Stream<NormalizedRect> roi, float multiplier,
    api2::builder::Graph& graph);

// Adds a node to the provided graph that gets the landmarks render data
// according to the renderer_options.
api2::builder::Stream<mediapipe::RenderData> RenderLandmarks(
    api2::builder::Stream<mediapipe::NormalizedLandmarkList> landmarks,
    std::optional<api2::builder::Stream<float>> render_scale,
    const mediapipe::LandmarksToRenderDataCalculatorOptions& renderer_options,
    api2::builder::Graph& graph);

// Adds a node to the provided graph that gets the rect render data according to
// the renderer_options.
api2::builder::Stream<mediapipe::RenderData> RenderRect(
    api2::builder::Stream<NormalizedRect> rect,
    const mediapipe::RectToRenderDataCalculatorOptions& renderer_options,
    api2::builder::Graph& graph);

}  // namespace mediapipe::tasks::vision::utils

#endif  // MEDIAPIPE_TASKS_CC_VISION_UTILS_DATA_RENDERER_H_
