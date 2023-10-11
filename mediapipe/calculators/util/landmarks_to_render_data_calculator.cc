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
#include "mediapipe/calculators/util/landmarks_to_render_data_calculator.h"

#include "absl/memory/memory.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "mediapipe/calculators/util/landmarks_to_render_data_calculator.pb.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/calculator_options.pb.h"
#include "mediapipe/framework/formats/landmark.pb.h"
#include "mediapipe/framework/formats/location_data.pb.h"
#include "mediapipe/framework/port/ret_check.h"
#include "mediapipe/util/color.pb.h"
#include "mediapipe/util/render_data.pb.h"
namespace mediapipe {

namespace {

constexpr char kLandmarksTag[] = "LANDMARKS";
constexpr char kNormLandmarksTag[] = "NORM_LANDMARKS";
constexpr char kRenderScaleTag[] = "RENDER_SCALE";
constexpr char kRenderDataTag[] = "RENDER_DATA";
constexpr char kLandmarkLabel[] = "KEYPOINT";

inline Color DefaultMinDepthLineColor() {
  Color color;
  color.set_r(0);
  color.set_g(0);
  color.set_b(0);
  return color;
}

inline Color DefaultMaxDepthLineColor() {
  Color color;
  color.set_r(255);
  color.set_g(255);
  color.set_b(255);
  return color;
}

inline Color MixColors(const Color& color1, const Color& color2,
                       float color1_weight) {
  Color color;
  color.set_r(static_cast<int>(color1.r() * color1_weight +
                               color2.r() * (1.f - color1_weight)));
  color.set_g(static_cast<int>(color1.g() * color1_weight +
                               color2.g() * (1.f - color1_weight)));
  color.set_b(static_cast<int>(color1.b() * color1_weight +
                               color2.b() * (1.f - color1_weight)));
  return color;
}

inline void SetColor(RenderAnnotation* annotation, const Color& color) {
  annotation->mutable_color()->set_r(color.r());
  annotation->mutable_color()->set_g(color.g());
  annotation->mutable_color()->set_b(color.b());
}

// Remap x from range [lo hi] to range [0 1] then multiply by scale.
inline float Remap(float x, float lo, float hi, float scale) {
  return (x - lo) / (hi - lo + 1e-6) * scale;
}

template <class LandmarkListType, class LandmarkType>
inline void GetMinMaxZ(const LandmarkListType& landmarks, float* z_min,
                       float* z_max) {
  *z_min = std::numeric_limits<float>::max();
  *z_max = std::numeric_limits<float>::min();
  for (int i = 0; i < landmarks.landmark_size(); ++i) {
    const LandmarkType& landmark = landmarks.landmark(i);
    *z_min = std::min(landmark.z(), *z_min);
    *z_max = std::max(landmark.z(), *z_max);
  }
}

template <class LandmarkType>
bool IsLandmarkVisibleAndPresent(const LandmarkType& landmark,
                                 bool utilize_visibility,
                                 float visibility_threshold,
                                 bool utilize_presence,
                                 float presence_threshold) {
  if (utilize_visibility && landmark.has_visibility() &&
      landmark.visibility() < visibility_threshold) {
    return false;
  }
  if (utilize_presence && landmark.has_presence() &&
      landmark.presence() < presence_threshold) {
    return false;
  }
  return true;
}

void SetColorSizeValueFromZ(float z, float z_min, float z_max,
                            RenderAnnotation* render_annotation,
                            float min_depth_circle_thickness,
                            float max_depth_circle_thickness) {
  const int color_value = 255 - static_cast<int>(Remap(z, z_min, z_max, 255));
  ::mediapipe::Color color;
  color.set_r(color_value);
  color.set_g(color_value);
  color.set_b(color_value);
  SetColor(render_annotation, color);
  const float scale = max_depth_circle_thickness - min_depth_circle_thickness;
  const int thickness = static_cast<int>(
      min_depth_circle_thickness + (1.f - Remap(z, z_min, z_max, 1)) * scale);
  render_annotation->set_thickness(thickness);
}

template <class LandmarkType>
void AddConnectionToRenderData(const LandmarkType& start,
                               const LandmarkType& end,
                               const Color& color_start, const Color& color_end,
                               float thickness, bool normalized,
                               RenderData* render_data) {
  auto* connection_annotation = render_data->add_render_annotations();
  RenderAnnotation::GradientLine* line =
      connection_annotation->mutable_gradient_line();
  line->set_x_start(start.x());
  line->set_y_start(start.y());
  line->set_x_end(end.x());
  line->set_y_end(end.y());
  line->set_normalized(normalized);
  line->mutable_color1()->set_r(color_start.r());
  line->mutable_color1()->set_g(color_start.g());
  line->mutable_color1()->set_b(color_start.b());
  line->mutable_color2()->set_r(color_end.r());
  line->mutable_color2()->set_g(color_end.g());
  line->mutable_color2()->set_b(color_end.b());

  connection_annotation->set_thickness(thickness);
}

template <class LandmarkListType, class LandmarkType>
void AddConnectionsWithDepth(const LandmarkListType& landmarks,
                             const std::vector<int>& landmark_connections,
                             bool utilize_visibility,
                             float visibility_threshold, bool utilize_presence,
                             float presence_threshold, float thickness,
                             bool normalized, float min_z, float max_z,
                             const Color& min_depth_line_color,
                             const Color& max_depth_line_color,
                             RenderData* render_data) {
  for (int i = 0; i < landmark_connections.size(); i += 2) {
    if (landmark_connections[i] >= landmarks.landmark_size() ||
        landmark_connections[i + 1] >= landmarks.landmark_size()) {
      continue;
    }
    const auto& ld0 = landmarks.landmark(landmark_connections[i]);
    const auto& ld1 = landmarks.landmark(landmark_connections[i + 1]);
    if (!IsLandmarkVisibleAndPresent<LandmarkType>(
            ld0, utilize_visibility, visibility_threshold, utilize_presence,
            presence_threshold) ||
        !IsLandmarkVisibleAndPresent<LandmarkType>(
            ld1, utilize_visibility, visibility_threshold, utilize_presence,
            presence_threshold)) {
      continue;
    }
    const Color color0 = MixColors(min_depth_line_color, max_depth_line_color,
                                   Remap(ld0.z(), min_z, max_z, 1.f));
    const Color color1 = MixColors(min_depth_line_color, max_depth_line_color,
                                   Remap(ld1.z(), min_z, max_z, 1.f));
    AddConnectionToRenderData<LandmarkType>(ld0, ld1, color0, color1, thickness,
                                            normalized, render_data);
  }
}

template <class LandmarkType>
void AddConnectionToRenderData(const LandmarkType& start,
                               const LandmarkType& end,
                               const Color& connection_color, float thickness,
                               bool normalized, RenderData* render_data) {
  auto* connection_annotation = render_data->add_render_annotations();
  RenderAnnotation::Line* line = connection_annotation->mutable_line();
  line->set_x_start(start.x());
  line->set_y_start(start.y());
  line->set_x_end(end.x());
  line->set_y_end(end.y());
  line->set_normalized(normalized);
  SetColor(connection_annotation, connection_color);
  connection_annotation->set_thickness(thickness);
}

template <class LandmarkListType, class LandmarkType>
void AddConnections(const LandmarkListType& landmarks,
                    const std::vector<int>& landmark_connections,
                    bool utilize_visibility, float visibility_threshold,
                    bool utilize_presence, float presence_threshold,
                    const Color& connection_color, float thickness,
                    bool normalized, RenderData* render_data) {
  for (int i = 0; i < landmark_connections.size(); i += 2) {
    if (landmark_connections[i] >= landmarks.landmark_size() ||
        landmark_connections[i + 1] >= landmarks.landmark_size()) {
      continue;
    }
    const auto& ld0 = landmarks.landmark(landmark_connections[i]);
    const auto& ld1 = landmarks.landmark(landmark_connections[i + 1]);
    if (!IsLandmarkVisibleAndPresent<LandmarkType>(
            ld0, utilize_visibility, visibility_threshold, utilize_presence,
            presence_threshold) ||
        !IsLandmarkVisibleAndPresent<LandmarkType>(
            ld1, utilize_visibility, visibility_threshold, utilize_presence,
            presence_threshold)) {
      continue;
    }
    AddConnectionToRenderData<LandmarkType>(ld0, ld1, connection_color,
                                            thickness, normalized, render_data);
  }
}

RenderAnnotation* AddPointRenderData(const Color& landmark_color,
                                     float thickness, RenderData* render_data) {
  auto* landmark_data_annotation = render_data->add_render_annotations();
  landmark_data_annotation->set_scene_tag(kLandmarkLabel);
  SetColor(landmark_data_annotation, landmark_color);
  landmark_data_annotation->set_thickness(thickness);
  return landmark_data_annotation;
}

}  // namespace

absl::Status LandmarksToRenderDataCalculator::GetContract(
    CalculatorContract* cc) {
  RET_CHECK(cc->Inputs().HasTag(kLandmarksTag) ||
            cc->Inputs().HasTag(kNormLandmarksTag))
      << "None of the input streams are provided.";
  RET_CHECK(!(cc->Inputs().HasTag(kLandmarksTag) &&
              cc->Inputs().HasTag(kNormLandmarksTag)))
      << "Can only one type of landmark can be taken. Either absolute or "
         "normalized landmarks.";

  if (cc->Inputs().HasTag(kLandmarksTag)) {
    cc->Inputs().Tag(kLandmarksTag).Set<LandmarkList>();
  }
  if (cc->Inputs().HasTag(kNormLandmarksTag)) {
    cc->Inputs().Tag(kNormLandmarksTag).Set<NormalizedLandmarkList>();
  }
  if (cc->Inputs().HasTag(kRenderScaleTag)) {
    cc->Inputs().Tag(kRenderScaleTag).Set<float>();
  }
  cc->Outputs().Tag(kRenderDataTag).Set<RenderData>();
  return absl::OkStatus();
}

absl::Status LandmarksToRenderDataCalculator::Open(CalculatorContext* cc) {
  cc->SetOffset(TimestampDiff(0));
  options_ = cc->Options<LandmarksToRenderDataCalculatorOptions>();

  // Parse landmarks connections to a vector.
  RET_CHECK_EQ(options_.landmark_connections_size() % 2, 0)
      << "Number of entries in landmark connections must be a multiple of 2";

  for (int i = 0; i < options_.landmark_connections_size(); ++i) {
    landmark_connections_.push_back(options_.landmark_connections(i));
  }

  return absl::OkStatus();
}

absl::Status LandmarksToRenderDataCalculator::Process(CalculatorContext* cc) {
  // Check that landmarks are not empty and skip rendering if so.
  // Don't emit an empty packet for this timestamp.
  if (cc->Inputs().HasTag(kLandmarksTag) &&
      cc->Inputs().Tag(kLandmarksTag).IsEmpty()) {
    return absl::OkStatus();
  }
  if (cc->Inputs().HasTag(kNormLandmarksTag) &&
      cc->Inputs().Tag(kNormLandmarksTag).IsEmpty()) {
    return absl::OkStatus();
  }

  auto render_data = absl::make_unique<RenderData>();
  bool visualize_depth = options_.visualize_landmark_depth();
  float z_min = 0.f;
  float z_max = 0.f;

  const Color min_depth_line_color = options_.has_min_depth_line_color()
                                         ? options_.min_depth_line_color()
                                         : DefaultMinDepthLineColor();
  const Color max_depth_line_color = options_.has_max_depth_line_color()
                                         ? options_.max_depth_line_color()
                                         : DefaultMaxDepthLineColor();

  // Apply scale to `thickness` of rendered landmarks and connections to make
  // them bigger when object (e.g. pose, hand or face) is closer/bigger and
  // snaller when object is further/smaller.
  float thickness = options_.thickness();
  if (cc->Inputs().HasTag(kRenderScaleTag)) {
    const float render_scale = cc->Inputs().Tag(kRenderScaleTag).Get<float>();
    thickness *= render_scale;
  }

  if (cc->Inputs().HasTag(kLandmarksTag)) {
    const LandmarkList& landmarks =
        cc->Inputs().Tag(kLandmarksTag).Get<LandmarkList>();
    if (visualize_depth) {
      GetMinMaxZ<LandmarkList, Landmark>(landmarks, &z_min, &z_max);
    }
    // Only change rendering if there are actually z values other than 0.
    visualize_depth &= ((z_max - z_min) > 1e-3);
    if (visualize_depth) {
      AddConnectionsWithDepth<LandmarkList, Landmark>(
          landmarks, landmark_connections_, options_.utilize_visibility(),
          options_.visibility_threshold(), options_.utilize_presence(),
          options_.presence_threshold(), thickness, /*normalized=*/false, z_min,
          z_max, min_depth_line_color, max_depth_line_color, render_data.get());
    } else {
      AddConnections<LandmarkList, Landmark>(
          landmarks, landmark_connections_, options_.utilize_visibility(),
          options_.visibility_threshold(), options_.utilize_presence(),
          options_.presence_threshold(), options_.connection_color(), thickness,
          /*normalized=*/false, render_data.get());
    }
    if (options_.render_landmarks()) {
      for (int i = 0; i < landmarks.landmark_size(); ++i) {
        const Landmark& landmark = landmarks.landmark(i);

        if (!IsLandmarkVisibleAndPresent<Landmark>(
                landmark, options_.utilize_visibility(),
                options_.visibility_threshold(), options_.utilize_presence(),
                options_.presence_threshold())) {
          continue;
        }

        auto* landmark_data_render = AddPointRenderData(
            options_.landmark_color(), thickness, render_data.get());
        if (visualize_depth) {
          SetColorSizeValueFromZ(landmark.z(), z_min, z_max,
                                 landmark_data_render,
                                 options_.min_depth_circle_thickness(),
                                 options_.max_depth_circle_thickness());
        }
        auto* landmark_data = landmark_data_render->mutable_point();
        landmark_data->set_normalized(false);
        landmark_data->set_x(landmark.x());
        landmark_data->set_y(landmark.y());
      }
    }
  }

  if (cc->Inputs().HasTag(kNormLandmarksTag)) {
    const NormalizedLandmarkList& landmarks =
        cc->Inputs().Tag(kNormLandmarksTag).Get<NormalizedLandmarkList>();
    if (visualize_depth) {
      GetMinMaxZ<NormalizedLandmarkList, NormalizedLandmark>(landmarks, &z_min,
                                                             &z_max);
    }
    // Only change rendering if there are actually z values other than 0.
    visualize_depth &= ((z_max - z_min) > 1e-3);
    if (visualize_depth) {
      AddConnectionsWithDepth<NormalizedLandmarkList, NormalizedLandmark>(
          landmarks, landmark_connections_, options_.utilize_visibility(),
          options_.visibility_threshold(), options_.utilize_presence(),
          options_.presence_threshold(), thickness, /*normalized=*/true, z_min,
          z_max, min_depth_line_color, max_depth_line_color, render_data.get());
    } else {
      AddConnections<NormalizedLandmarkList, NormalizedLandmark>(
          landmarks, landmark_connections_, options_.utilize_visibility(),
          options_.visibility_threshold(), options_.utilize_presence(),
          options_.presence_threshold(), options_.connection_color(), thickness,
          /*normalized=*/true, render_data.get());
    }
    if (options_.render_landmarks()) {
      for (int i = 0; i < landmarks.landmark_size(); ++i) {
        const NormalizedLandmark& landmark = landmarks.landmark(i);

        if (!IsLandmarkVisibleAndPresent<NormalizedLandmark>(
                landmark, options_.utilize_visibility(),
                options_.visibility_threshold(), options_.utilize_presence(),
                options_.presence_threshold())) {
          continue;
        }

        auto* landmark_data_render = AddPointRenderData(
            options_.landmark_color(), thickness, render_data.get());
        if (visualize_depth) {
          SetColorSizeValueFromZ(landmark.z(), z_min, z_max,
                                 landmark_data_render,
                                 options_.min_depth_circle_thickness(),
                                 options_.max_depth_circle_thickness());
        }
        auto* landmark_data = landmark_data_render->mutable_point();
        landmark_data->set_normalized(true);
        landmark_data->set_x(landmark.x());
        landmark_data->set_y(landmark.y());
      }
    }
  }

  cc->Outputs()
      .Tag(kRenderDataTag)
      .Add(render_data.release(), cc->InputTimestamp());
  return absl::OkStatus();
}

REGISTER_CALCULATOR(LandmarksToRenderDataCalculator);
}  // namespace mediapipe
