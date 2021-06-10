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

#include "absl/memory/memory.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "mediapipe/calculators/util/detections_to_render_data_calculator.pb.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/calculator_options.pb.h"
#include "mediapipe/framework/formats/detection.pb.h"
#include "mediapipe/framework/formats/location_data.pb.h"
#include "mediapipe/framework/port/ret_check.h"
#include "mediapipe/util/color.pb.h"
#include "mediapipe/util/render_data.pb.h"
namespace mediapipe {

namespace {

constexpr char kDetectionTag[] = "DETECTION";
constexpr char kDetectionsTag[] = "DETECTIONS";
constexpr char kDetectionListTag[] = "DETECTION_LIST";
constexpr char kRenderDataTag[] = "RENDER_DATA";

constexpr char kSceneLabelLabel[] = "LABEL";
constexpr char kSceneFeatureLabel[] = "FEATURE";
constexpr char kSceneLocationLabel[] = "LOCATION";
constexpr char kKeypointLabel[] = "KEYPOINT";

// The ratio of detection label font height to the height of detection bounding
// box.
constexpr double kLabelToBoundingBoxRatio = 0.1;
// Perserve 2 decimal digits.
constexpr float kNumScoreDecimalDigitsMultipler = 100;

}  // namespace

// A calculator that converts Detection proto to RenderData proto for
// visualization.
//
// Detection is the format for encoding one or more detections in an image.
// The input can be std::vector<Detection> or DetectionList.
//
// Please note that only Location Data formats of BOUNDING_BOX and
// RELATIVE_BOUNDING_BOX are supported. Normalized coordinates for
// RELATIVE_BOUNDING_BOX must be between 0.0 and 1.0. Any incremental normalized
// coordinates calculation in this calculator is capped at 1.0.
//
// The text(s) for "label(_id),score" will be shown on top left
// corner of the bounding box. The text for "feature_tag" will be shown on
// bottom left corner of the bounding box.
//
// Example config:
// node {
//   calculator: "DetectionsToRenderDataCalculator"
//   input_stream: "DETECTION:detection"
//   input_stream: "DETECTIONS:detections"
//   input_stream: "DETECTION_LIST:detection_list"
//   output_stream: "RENDER_DATA:render_data"
//   options {
//     [DetectionsToRenderDataCalculatorOptions.ext] {
//       produce_empty_packet : false
//     }
//   }
// }
class DetectionsToRenderDataCalculator : public CalculatorBase {
 public:
  DetectionsToRenderDataCalculator() {}
  ~DetectionsToRenderDataCalculator() override {}
  DetectionsToRenderDataCalculator(const DetectionsToRenderDataCalculator&) =
      delete;
  DetectionsToRenderDataCalculator& operator=(
      const DetectionsToRenderDataCalculator&) = delete;

  static absl::Status GetContract(CalculatorContract* cc);

  absl::Status Open(CalculatorContext* cc) override;

  absl::Status Process(CalculatorContext* cc) override;

 private:
  // These utility methods are supposed to be used only by this class. No
  // external client should depend on them. Due to C++ style guide unnamed
  // namespace should not be used in header files. So, these has been defined
  // as private static methods.
  static void SetRenderAnnotationColorThickness(
      const DetectionsToRenderDataCalculatorOptions& options,
      RenderAnnotation* render_annotation);

  static void SetTextCoordinate(bool normalized, double left, double baseline,
                                RenderAnnotation::Text* text);

  static void SetRectCoordinate(bool normalized, double xmin, double ymin,
                                double width, double height,
                                RenderAnnotation::Rectangle* rect);

  static void AddLabels(const Detection& detection,
                        const DetectionsToRenderDataCalculatorOptions& options,
                        float text_line_height, RenderData* render_data);
  static void AddFeatureTag(
      const Detection& detection,
      const DetectionsToRenderDataCalculatorOptions& options,
      float text_line_height, RenderData* render_data);
  static void AddLocationData(
      const Detection& detection,
      const DetectionsToRenderDataCalculatorOptions& options,
      RenderData* render_data);
  static void AddDetectionToRenderData(
      const Detection& detection,
      const DetectionsToRenderDataCalculatorOptions& options,
      RenderData* render_data);
};
REGISTER_CALCULATOR(DetectionsToRenderDataCalculator);

absl::Status DetectionsToRenderDataCalculator::GetContract(
    CalculatorContract* cc) {
  RET_CHECK(cc->Inputs().HasTag(kDetectionListTag) ||
            cc->Inputs().HasTag(kDetectionsTag) ||
            cc->Inputs().HasTag(kDetectionTag))
      << "None of the input streams are provided.";

  if (cc->Inputs().HasTag(kDetectionTag)) {
    cc->Inputs().Tag(kDetectionTag).Set<Detection>();
  }
  if (cc->Inputs().HasTag(kDetectionListTag)) {
    cc->Inputs().Tag(kDetectionListTag).Set<DetectionList>();
  }
  if (cc->Inputs().HasTag(kDetectionsTag)) {
    cc->Inputs().Tag(kDetectionsTag).Set<std::vector<Detection>>();
  }
  cc->Outputs().Tag(kRenderDataTag).Set<RenderData>();
  return absl::OkStatus();
}

absl::Status DetectionsToRenderDataCalculator::Open(CalculatorContext* cc) {
  cc->SetOffset(TimestampDiff(0));

  return absl::OkStatus();
}

absl::Status DetectionsToRenderDataCalculator::Process(CalculatorContext* cc) {
  const auto& options = cc->Options<DetectionsToRenderDataCalculatorOptions>();
  const bool has_detection_from_list =
      cc->Inputs().HasTag(kDetectionListTag) && !cc->Inputs()
                                                     .Tag(kDetectionListTag)
                                                     .Get<DetectionList>()
                                                     .detection()
                                                     .empty();
  const bool has_detection_from_vector =
      cc->Inputs().HasTag(kDetectionsTag) &&
      !cc->Inputs().Tag(kDetectionsTag).Get<std::vector<Detection>>().empty();
  const bool has_single_detection = cc->Inputs().HasTag(kDetectionTag) &&
                                    !cc->Inputs().Tag(kDetectionTag).IsEmpty();
  if (!options.produce_empty_packet() && !has_detection_from_list &&
      !has_detection_from_vector && !has_single_detection) {
    return absl::OkStatus();
  }

  // TODO: Add score threshold to
  // DetectionsToRenderDataCalculatorOptions.
  auto render_data = absl::make_unique<RenderData>();
  render_data->set_scene_class(options.scene_class());
  if (has_detection_from_list) {
    for (const auto& detection :
         cc->Inputs().Tag(kDetectionListTag).Get<DetectionList>().detection()) {
      AddDetectionToRenderData(detection, options, render_data.get());
    }
  }
  if (has_detection_from_vector) {
    for (const auto& detection :
         cc->Inputs().Tag(kDetectionsTag).Get<std::vector<Detection>>()) {
      AddDetectionToRenderData(detection, options, render_data.get());
    }
  }
  if (has_single_detection) {
    AddDetectionToRenderData(cc->Inputs().Tag(kDetectionTag).Get<Detection>(),
                             options, render_data.get());
  }
  cc->Outputs()
      .Tag(kRenderDataTag)
      .Add(render_data.release(), cc->InputTimestamp());
  return absl::OkStatus();
}

void DetectionsToRenderDataCalculator::SetRenderAnnotationColorThickness(
    const DetectionsToRenderDataCalculatorOptions& options,
    RenderAnnotation* render_annotation) {
  render_annotation->mutable_color()->set_r(options.color().r());
  render_annotation->mutable_color()->set_g(options.color().g());
  render_annotation->mutable_color()->set_b(options.color().b());
  render_annotation->set_thickness(options.thickness());
}

void DetectionsToRenderDataCalculator::SetTextCoordinate(
    bool normalized, double left, double baseline,
    RenderAnnotation::Text* text) {
  text->set_normalized(normalized);
  text->set_left(normalized ? std::max(left, 0.0) : left);
  // Normalized coordinates must be between 0.0 and 1.0, if they are used.
  text->set_baseline(normalized ? std::min(baseline, 1.0) : baseline);
}

void DetectionsToRenderDataCalculator::SetRectCoordinate(
    bool normalized, double xmin, double ymin, double width, double height,
    RenderAnnotation::Rectangle* rect) {
  if (xmin + width < 0.0 || ymin + height < 0.0) return;
  if (normalized) {
    if (xmin > 1.0 || ymin > 1.0) return;
  }
  rect->set_normalized(normalized);
  rect->set_left(normalized ? std::max(xmin, 0.0) : xmin);
  rect->set_top(normalized ? std::max(ymin, 0.0) : ymin);
  // No "xmin + width -1" because the coordinates can be relative, i.e. [0,1],
  // and we don't know what 1 pixel means in term of double [0,1].
  // For consistency decided to not decrease by 1 also when it is not relative.
  // However, when the coordinate is normalized it has to be between 0.0 and
  // 1.0.
  rect->set_right(normalized ? std::min(xmin + width, 1.0) : xmin + width);
  rect->set_bottom(normalized ? std::min(ymin + height, 1.0) : ymin + height);
}

void DetectionsToRenderDataCalculator::AddLabels(
    const Detection& detection,
    const DetectionsToRenderDataCalculatorOptions& options,
    float text_line_height, RenderData* render_data) {
  CHECK(detection.label().empty() || detection.label_id().empty() ||
        detection.label_size() == detection.label_id_size())
      << "String or integer labels should be of same size. Or only one of them "
         "is present.";
  const auto num_labels =
      std::max(detection.label_size(), detection.label_id_size());
  CHECK_EQ(detection.score_size(), num_labels)
      << "Number of scores and labels should match for detection.";

  // Extracts all "label(_id),score" for the detection.
  std::vector<std::string> label_and_scores = {};
  for (int i = 0; i < num_labels; ++i) {
    std::string label_str = detection.label().empty()
                                ? absl::StrCat(detection.label_id(i))
                                : detection.label(i);
    const float rounded_score =
        std::round(detection.score(i) * kNumScoreDecimalDigitsMultipler) /
        kNumScoreDecimalDigitsMultipler;
    std::string label_and_score =
        absl::StrCat(label_str, options.text_delimiter(), rounded_score,
                     options.text_delimiter());
    label_and_scores.push_back(label_and_score);
  }
  std::vector<std::string> labels;
  if (options.render_detection_id()) {
    const std::string detection_id_str =
        absl::StrCat("Id: ", detection.detection_id());
    labels.push_back(detection_id_str);
  }
  if (options.one_label_per_line()) {
    labels.insert(labels.end(), label_and_scores.begin(),
                  label_and_scores.end());
  } else {
    labels.push_back(absl::StrJoin(label_and_scores, ""));
  }
  // Add the render annotations for "label(_id),score".
  for (int i = 0; i < labels.size(); ++i) {
    auto label = labels.at(i);
    auto* label_annotation = render_data->add_render_annotations();
    label_annotation->set_scene_tag(kSceneLabelLabel);
    SetRenderAnnotationColorThickness(options, label_annotation);
    auto* text = label_annotation->mutable_text();
    *text = options.text();
    text->set_display_text(label);
    if (detection.location_data().format() == LocationData::BOUNDING_BOX) {
      SetTextCoordinate(false, detection.location_data().bounding_box().xmin(),
                        detection.location_data().bounding_box().ymin() +
                            (i + 1) * text_line_height,
                        text);
    } else {
      text->set_font_height(text_line_height * 0.9);
      SetTextCoordinate(
          true, detection.location_data().relative_bounding_box().xmin(),
          detection.location_data().relative_bounding_box().ymin() +
              (i + 1) * text_line_height,
          text);
    }
  }
}

void DetectionsToRenderDataCalculator::AddFeatureTag(
    const Detection& detection,
    const DetectionsToRenderDataCalculatorOptions& options,
    float text_line_height, RenderData* render_data) {
  auto* feature_tag_annotation = render_data->add_render_annotations();
  feature_tag_annotation->set_scene_tag(kSceneFeatureLabel);
  SetRenderAnnotationColorThickness(options, feature_tag_annotation);
  auto* feature_tag_text = feature_tag_annotation->mutable_text();
  feature_tag_text->set_display_text(detection.feature_tag());
  if (detection.location_data().format() == LocationData::BOUNDING_BOX) {
    SetTextCoordinate(false, detection.location_data().bounding_box().xmin(),
                      detection.location_data().bounding_box().ymin() +
                          detection.location_data().bounding_box().height(),
                      feature_tag_text);
  } else {
    feature_tag_text->set_font_height(text_line_height * 0.9);
    SetTextCoordinate(
        true, detection.location_data().relative_bounding_box().xmin(),
        detection.location_data().relative_bounding_box().ymin() +
            detection.location_data().relative_bounding_box().height(),
        feature_tag_text);
  }
}

void DetectionsToRenderDataCalculator::AddLocationData(
    const Detection& detection,
    const DetectionsToRenderDataCalculatorOptions& options,
    RenderData* render_data) {
  auto* location_data_annotation = render_data->add_render_annotations();
  location_data_annotation->set_scene_tag(kSceneLocationLabel);
  SetRenderAnnotationColorThickness(options, location_data_annotation);
  auto* location_data_rect = location_data_annotation->mutable_rectangle();
  if (detection.location_data().format() == LocationData::BOUNDING_BOX) {
    SetRectCoordinate(false, detection.location_data().bounding_box().xmin(),
                      detection.location_data().bounding_box().ymin(),
                      detection.location_data().bounding_box().width(),
                      detection.location_data().bounding_box().height(),
                      location_data_rect);
  } else {
    SetRectCoordinate(
        true, detection.location_data().relative_bounding_box().xmin(),
        detection.location_data().relative_bounding_box().ymin(),
        detection.location_data().relative_bounding_box().width(),
        detection.location_data().relative_bounding_box().height(),
        location_data_rect);
    // Keypoints are only supported in normalized/relative coordinates.
    if (detection.location_data().relative_keypoints_size()) {
      for (int i = 0; i < detection.location_data().relative_keypoints_size();
           ++i) {
        auto* keypoint_data_annotation = render_data->add_render_annotations();
        keypoint_data_annotation->set_scene_tag(kKeypointLabel);
        SetRenderAnnotationColorThickness(options, keypoint_data_annotation);
        auto* keypoint_data = keypoint_data_annotation->mutable_point();
        keypoint_data->set_normalized(true);
        // See location_data.proto for detail.
        keypoint_data->set_x(
            detection.location_data().relative_keypoints(i).x());
        keypoint_data->set_y(
            detection.location_data().relative_keypoints(i).y());
      }
    }
  }
}

void DetectionsToRenderDataCalculator::AddDetectionToRenderData(
    const Detection& detection,
    const DetectionsToRenderDataCalculatorOptions& options,
    RenderData* render_data) {
  CHECK(detection.location_data().format() == LocationData::BOUNDING_BOX ||
        detection.location_data().format() ==
            LocationData::RELATIVE_BOUNDING_BOX)
      << "Only Detection with formats of BOUNDING_BOX or RELATIVE_BOUNDING_BOX "
         "are supported.";
  double text_line_height;
  if (detection.location_data().format() == LocationData::BOUNDING_BOX) {
    text_line_height = options.text().font_height();
  } else {
    // Determine the text line height based on the default label to bounding box
    // ratio and the number of labels.
    text_line_height =
        detection.location_data().relative_bounding_box().height() *
        std::min(kLabelToBoundingBoxRatio,
                 1 / (double)(std::max(detection.label_size(),
                                       detection.label_id_size()) +
                              1 /* for feature_tag */));
  }
  AddLabels(detection, options, text_line_height, render_data);
  AddFeatureTag(detection, options, text_line_height, render_data);
  AddLocationData(detection, options, render_data);
}
}  // namespace mediapipe
