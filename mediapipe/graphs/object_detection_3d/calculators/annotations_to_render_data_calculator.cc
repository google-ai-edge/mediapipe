// Copyright 2020 The MediaPipe Authors.
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

#include "absl/memory/memory.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/calculator_options.pb.h"
#include "mediapipe/framework/port/ret_check.h"
#include "mediapipe/graphs/object_detection_3d/calculators/annotation_data.pb.h"
#include "mediapipe/graphs/object_detection_3d/calculators/annotations_to_render_data_calculator.pb.h"
#include "mediapipe/util/color.pb.h"
#include "mediapipe/util/render_data.pb.h"

namespace mediapipe {

namespace {

constexpr char kAnnotationTag[] = "ANNOTATIONS";
constexpr char kRenderDataTag[] = "RENDER_DATA";
constexpr char kKeypointLabel[] = "KEYPOINT";
constexpr int kMaxLandmarkThickness = 18;

inline void SetColor(RenderAnnotation* annotation, const Color& color) {
  annotation->mutable_color()->set_r(color.r());
  annotation->mutable_color()->set_g(color.g());
  annotation->mutable_color()->set_b(color.b());
}

// Remap x from range [lo hi] to range [0 1] then multiply by scale.
inline float Remap(float x, float lo, float hi, float scale) {
  return (x - lo) / (hi - lo + 1e-6) * scale;
}

inline void GetMinMaxZ(const FrameAnnotation& annotations, float* z_min,
                       float* z_max) {
  *z_min = std::numeric_limits<float>::max();
  *z_max = std::numeric_limits<float>::min();
  // Use a global depth scale for all the objects in the scene
  for (const auto& object : annotations.annotations()) {
    for (const auto& keypoint : object.keypoints()) {
      *z_min = std::min(keypoint.point_2d().depth(), *z_min);
      *z_max = std::max(keypoint.point_2d().depth(), *z_max);
    }
  }
}

void SetColorSizeValueFromZ(float z, float z_min, float z_max,
                            RenderAnnotation* render_annotation) {
  const int color_value = 255 - static_cast<int>(Remap(z, z_min, z_max, 255));
  ::mediapipe::Color color;
  color.set_r(color_value);
  color.set_g(color_value);
  color.set_b(color_value);
  SetColor(render_annotation, color);
  const int thickness = static_cast<int>((1.f - Remap(z, z_min, z_max, 1)) *
                                         kMaxLandmarkThickness);
  render_annotation->set_thickness(thickness);
}

}  // namespace

// A calculator that converts FrameAnnotation proto to RenderData proto for
// visualization. The input should be the FrameAnnotation proto buffer. It is
// also possible to specify the connections between landmarks.
//
// Example config:
// node {
//   calculator: "AnnotationsToRenderDataCalculator"
//   input_stream: "ANNOTATIONS:annotations"
//   output_stream: "RENDER_DATA:render_data"
//   options {
//     [AnnotationsToRenderDataCalculator.ext] {
//       landmark_connections: [0, 1, 1, 2]
//       landmark_color { r: 0 g: 255 b: 0 }
//       connection_color { r: 0 g: 255 b: 0 }
//       thickness: 4.0
//     }
//   }
// }
class AnnotationsToRenderDataCalculator : public CalculatorBase {
 public:
  AnnotationsToRenderDataCalculator() {}
  ~AnnotationsToRenderDataCalculator() override {}
  AnnotationsToRenderDataCalculator(const AnnotationsToRenderDataCalculator&) =
      delete;
  AnnotationsToRenderDataCalculator& operator=(
      const AnnotationsToRenderDataCalculator&) = delete;

  static ::mediapipe::Status GetContract(CalculatorContract* cc);

  ::mediapipe::Status Open(CalculatorContext* cc) override;

  ::mediapipe::Status Process(CalculatorContext* cc) override;

 private:
  static void SetRenderAnnotationColorThickness(
      const AnnotationsToRenderDataCalculatorOptions& options,
      RenderAnnotation* render_annotation);
  static RenderAnnotation* AddPointRenderData(
      const AnnotationsToRenderDataCalculatorOptions& options,
      RenderData* render_data);

  // Add a command to draw a line in the rendering queue. The line is drawn from
  // (start_x, start_y) to (end_x, end_y). The input x,y can either be in pixel
  // or normalized coordinate [0, 1] as indicated by the normalized flag.
  static void AddConnectionToRenderData(
      float start_x, float start_y, float end_x, float end_y,
      const AnnotationsToRenderDataCalculatorOptions& options, bool normalized,
      RenderData* render_data);

  // Same as above function. Instead of using color data to render the line, it
  // re-colors the line according to the two depth value. gray_val1 is the color
  // of the starting point and gray_val2 is the color of the ending point. The
  // line is colored using gradient color from gray_val1 to gray_val2. The
  // gray_val ranges from [0 to 255] for black to white.
  static void AddConnectionToRenderData(
      float start_x, float start_y, float end_x, float end_y,
      const AnnotationsToRenderDataCalculatorOptions& options, bool normalized,
      int gray_val1, int gray_val2, RenderData* render_data);

  AnnotationsToRenderDataCalculatorOptions options_;
};
REGISTER_CALCULATOR(AnnotationsToRenderDataCalculator);

::mediapipe::Status AnnotationsToRenderDataCalculator::GetContract(
    CalculatorContract* cc) {
  RET_CHECK(cc->Inputs().HasTag(kAnnotationTag)) << "No input stream found.";
  if (cc->Inputs().HasTag(kAnnotationTag)) {
    cc->Inputs().Tag(kAnnotationTag).Set<FrameAnnotation>();
  }
  cc->Outputs().Tag(kRenderDataTag).Set<RenderData>();

  return ::mediapipe::OkStatus();
}

::mediapipe::Status AnnotationsToRenderDataCalculator::Open(
    CalculatorContext* cc) {
  cc->SetOffset(TimestampDiff(0));
  options_ = cc->Options<AnnotationsToRenderDataCalculatorOptions>();

  return ::mediapipe::OkStatus();
}

::mediapipe::Status AnnotationsToRenderDataCalculator::Process(
    CalculatorContext* cc) {
  auto render_data = absl::make_unique<RenderData>();
  bool visualize_depth = options_.visualize_landmark_depth();
  float z_min = 0.f;
  float z_max = 0.f;

  if (cc->Inputs().HasTag(kAnnotationTag)) {
    const auto& annotations =
        cc->Inputs().Tag(kAnnotationTag).Get<FrameAnnotation>();
    RET_CHECK_EQ(options_.landmark_connections_size() % 2, 0)
        << "Number of entries in landmark connections must be a multiple of 2";

    if (visualize_depth) {
      GetMinMaxZ(annotations, &z_min, &z_max);
      // Only change rendering if there are actually z values other than 0.
      visualize_depth &= ((z_max - z_min) > 1e-3);
    }

    for (const auto& object : annotations.annotations()) {
      for (const auto& keypoint : object.keypoints()) {
        auto* keypoint_data_render =
            AddPointRenderData(options_, render_data.get());
        auto* point = keypoint_data_render->mutable_point();
        if (visualize_depth) {
          SetColorSizeValueFromZ(keypoint.point_2d().depth(), z_min, z_max,
                                 keypoint_data_render);
        }

        point->set_normalized(true);
        point->set_x(keypoint.point_2d().x());
        point->set_y(keypoint.point_2d().y());
      }

      // Add edges
      for (int i = 0; i < options_.landmark_connections_size(); i += 2) {
        const auto& ld0 =
            object.keypoints(options_.landmark_connections(i)).point_2d();
        const auto& ld1 =
            object.keypoints(options_.landmark_connections(i + 1)).point_2d();
        const bool normalized = true;

        if (visualize_depth) {
          const int gray_val1 =
              255 - static_cast<int>(Remap(ld0.depth(), z_min, z_max, 255));
          const int gray_val2 =
              255 - static_cast<int>(Remap(ld1.depth(), z_min, z_max, 255));
          AddConnectionToRenderData(ld0.x(), ld0.y(), ld1.x(), ld1.y(),
                                    options_, normalized, gray_val1, gray_val2,
                                    render_data.get());
        } else {
          AddConnectionToRenderData(ld0.x(), ld0.y(), ld1.x(), ld1.y(),
                                    options_, normalized, render_data.get());
        }
      }
    }
  }

  cc->Outputs()
      .Tag(kRenderDataTag)
      .Add(render_data.release(), cc->InputTimestamp());

  return ::mediapipe::OkStatus();
}

void AnnotationsToRenderDataCalculator::AddConnectionToRenderData(
    float start_x, float start_y, float end_x, float end_y,
    const AnnotationsToRenderDataCalculatorOptions& options, bool normalized,
    int gray_val1, int gray_val2, RenderData* render_data) {
  auto* connection_annotation = render_data->add_render_annotations();
  RenderAnnotation::GradientLine* line =
      connection_annotation->mutable_gradient_line();
  line->set_x_start(start_x);
  line->set_y_start(start_y);
  line->set_x_end(end_x);
  line->set_y_end(end_y);
  line->set_normalized(normalized);
  line->mutable_color1()->set_r(gray_val1);
  line->mutable_color1()->set_g(gray_val1);
  line->mutable_color1()->set_b(gray_val1);
  line->mutable_color2()->set_r(gray_val2);
  line->mutable_color2()->set_g(gray_val2);
  line->mutable_color2()->set_b(gray_val2);
  connection_annotation->set_thickness(options.thickness());
}

void AnnotationsToRenderDataCalculator::AddConnectionToRenderData(
    float start_x, float start_y, float end_x, float end_y,
    const AnnotationsToRenderDataCalculatorOptions& options, bool normalized,
    RenderData* render_data) {
  auto* connection_annotation = render_data->add_render_annotations();
  RenderAnnotation::Line* line = connection_annotation->mutable_line();
  line->set_x_start(start_x);
  line->set_y_start(start_y);
  line->set_x_end(end_x);
  line->set_y_end(end_y);
  line->set_normalized(normalized);
  SetColor(connection_annotation, options.connection_color());
  connection_annotation->set_thickness(options.thickness());
}

RenderAnnotation* AnnotationsToRenderDataCalculator::AddPointRenderData(
    const AnnotationsToRenderDataCalculatorOptions& options,
    RenderData* render_data) {
  auto* landmark_data_annotation = render_data->add_render_annotations();
  landmark_data_annotation->set_scene_tag(kKeypointLabel);
  SetRenderAnnotationColorThickness(options, landmark_data_annotation);
  return landmark_data_annotation;
}

void AnnotationsToRenderDataCalculator::SetRenderAnnotationColorThickness(
    const AnnotationsToRenderDataCalculatorOptions& options,
    RenderAnnotation* render_annotation) {
  SetColor(render_annotation, options.landmark_color());
  render_annotation->set_thickness(options.thickness());
}

}  // namespace mediapipe
