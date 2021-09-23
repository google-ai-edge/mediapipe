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

#include <memory>

#include "Eigen/Core"
#include "Eigen/Dense"
#include "Eigen/Geometry"
#include "absl/memory/memory.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/calculator_options.pb.h"
#include "mediapipe/framework/port/ret_check.h"
#include "mediapipe/framework/port/status.h"
#include "mediapipe/graphs/object_detection_3d/calculators/annotations_to_model_matrices_calculator.pb.h"
#include "mediapipe/graphs/object_detection_3d/calculators/model_matrix.pb.h"
#include "mediapipe/modules/objectron/calculators/annotation_data.pb.h"
#include "mediapipe/modules/objectron/calculators/box.h"
#include "mediapipe/util/color.pb.h"

namespace mediapipe {

namespace {

constexpr char kAnnotationTag[] = "ANNOTATIONS";
constexpr char kModelMatricesTag[] = "MODEL_MATRICES";

using Matrix3fRM = Eigen::Matrix<float, 3, 3, Eigen::RowMajor>;
using Matrix4fRM = Eigen::Matrix<float, 4, 4, Eigen::RowMajor>;

}  // namespace

// Converts the box prediction from Objectron Model to the Model matrices
// to be rendered.
//
// Input:
//  ANNOTATIONS - Frame annotations with lifted 3D points, the points are in
//     Objectron coordinate system.
// Output:
//  MODEL_MATRICES - Result ModelMatrices, in OpenGL coordinate system.
//
// Usage example:
// node {
//  calculator: "AnnotationsToModelMatricesCalculator"
//  input_stream: "ANNOTATIONS:objects"
//  output_stream: "MODEL_MATRICES:model_matrices"
//}

class AnnotationsToModelMatricesCalculator : public CalculatorBase {
 public:
  AnnotationsToModelMatricesCalculator() {}
  ~AnnotationsToModelMatricesCalculator() override {}
  AnnotationsToModelMatricesCalculator(
      const AnnotationsToModelMatricesCalculator&) = delete;
  AnnotationsToModelMatricesCalculator& operator=(
      const AnnotationsToModelMatricesCalculator&) = delete;

  static absl::Status GetContract(CalculatorContract* cc);

  absl::Status Open(CalculatorContext* cc) override;

  absl::Status Process(CalculatorContext* cc) override;

 private:
  absl::Status GetModelMatricesForAnnotations(
      const FrameAnnotation& annotations,
      TimedModelMatrixProtoList* model_matrix_list);

  AnnotationsToModelMatricesCalculatorOptions options_;
  Eigen::Vector3f model_scale_;
  Matrix4fRM model_transformation_;
};
REGISTER_CALCULATOR(AnnotationsToModelMatricesCalculator);

absl::Status AnnotationsToModelMatricesCalculator::GetContract(
    CalculatorContract* cc) {
  RET_CHECK(cc->Inputs().HasTag(kAnnotationTag)) << "No input stream found.";
  if (cc->Inputs().HasTag(kAnnotationTag)) {
    cc->Inputs().Tag(kAnnotationTag).Set<FrameAnnotation>();
  }

  if (cc->Outputs().HasTag(kModelMatricesTag)) {
    cc->Outputs().Tag(kModelMatricesTag).Set<TimedModelMatrixProtoList>();
  }

  if (cc->InputSidePackets().HasTag("MODEL_SCALE")) {
    cc->InputSidePackets().Tag("MODEL_SCALE").Set<float[]>();
  }

  if (cc->InputSidePackets().HasTag("MODEL_TRANSFORMATION")) {
    cc->InputSidePackets().Tag("MODEL_TRANSFORMATION").Set<float[]>();
  }
  return absl::OkStatus();
}

absl::Status AnnotationsToModelMatricesCalculator::Open(CalculatorContext* cc) {
  RET_CHECK(cc->Inputs().HasTag(kAnnotationTag));

  cc->SetOffset(TimestampDiff(0));
  options_ = cc->Options<AnnotationsToModelMatricesCalculatorOptions>();

  if (cc->InputSidePackets().HasTag("MODEL_SCALE")) {
    model_scale_ = Eigen::Map<const Eigen::Vector3f>(
        cc->InputSidePackets().Tag("MODEL_SCALE").Get<float[]>());
  } else if (options_.model_scale_size() == 3) {
    model_scale_ =
        Eigen::Map<const Eigen::Vector3f>(options_.model_scale().data());
  } else {
    model_scale_.setOnes();
  }

  if (cc->InputSidePackets().HasTag("MODEL_TRANSFORMATION")) {
    model_transformation_ = Eigen::Map<const Matrix4fRM>(
        cc->InputSidePackets().Tag("MODEL_TRANSFORMATION").Get<float[]>());
  } else if (options_.model_transformation_size() == 16) {
    model_transformation_ =
        Eigen::Map<const Matrix4fRM>(options_.model_transformation().data());
  } else {
    model_transformation_.setIdentity();
  }

  return absl::OkStatus();
}

absl::Status AnnotationsToModelMatricesCalculator::Process(
    CalculatorContext* cc) {
  auto model_matrices = std::make_unique<TimedModelMatrixProtoList>();

  const FrameAnnotation& annotations =
      cc->Inputs().Tag(kAnnotationTag).Get<FrameAnnotation>();

  if (!GetModelMatricesForAnnotations(annotations, model_matrices.get()).ok()) {
    return absl::InvalidArgumentError("Error in GetModelMatricesForBoxes");
  }
  cc->Outputs()
      .Tag(kModelMatricesTag)
      .Add(model_matrices.release(), cc->InputTimestamp());

  return absl::OkStatus();
}

absl::Status
AnnotationsToModelMatricesCalculator::GetModelMatricesForAnnotations(
    const FrameAnnotation& annotations,
    TimedModelMatrixProtoList* model_matrix_list) {
  if (model_matrix_list == nullptr) {
    return absl::InvalidArgumentError("model_matrix_list is nullptr");
  }
  model_matrix_list->clear_model_matrix();

  for (const auto& object : annotations.annotations()) {
    TimedModelMatrixProto* model_matrix = model_matrix_list->add_model_matrix();
    model_matrix->set_id(object.object_id());

    // Get object rotation, translation and scale.
    const auto object_rotation =
        Eigen::Map<const Matrix3fRM>(object.rotation().data());
    const auto object_translation =
        Eigen::Map<const Eigen::Vector3f>(object.translation().data());
    const auto object_scale =
        Eigen::Map<const Eigen::Vector3f>(object.scale().data());

    // Compose object transformation matrix.
    Matrix4fRM object_transformation;
    object_transformation.setIdentity();
    object_transformation.topLeftCorner<3, 3>() = object_rotation;
    object_transformation.topRightCorner<3, 1>() = object_translation;

    Matrix4fRM model_view;
    Matrix4fRM objectron_model;
    // The reference view is
    //
    // ref <<  0.,  0.,  1.,  0.,
    //        -1.,  0., 0.,  0.,
    //         0.,  -1.,  0.,  0.,
    //         0.,  0.,  0.,  1.;
    // We have objectron_model * model = model_view, to get objectron_model:
    // objectron_model = model_view * model^-1
    // clang-format off
    objectron_model << 1.0, 0.0, 0.0, 0.0,
                       0.0, -1., 0.0, 0.0,
                       0.0, 0.0, 1.0, 0.0,
                       0.0, 0.0, 0.0, 1.0;
    // clang-format on

    // Re-scale the CAD model to the scale of the estimated bounding box.
    const Eigen::Vector3f scale = model_scale_.cwiseProduct(object_scale);
    const Matrix4fRM model =
        model_transformation_.array().colwise() * scale.homogeneous().array();

    // Finally compute the model_view matrix.
    model_view = objectron_model * object_transformation * model;

    for (int i = 0; i < model_view.rows(); ++i) {
      for (int j = 0; j < model_view.cols(); ++j) {
        model_matrix->add_matrix_entries(model_view(i, j));
      }
    }
  }
  return absl::OkStatus();
}

}  // namespace mediapipe
