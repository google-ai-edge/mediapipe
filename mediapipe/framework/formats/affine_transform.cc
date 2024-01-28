// Copyright 2021 The MediaPipe Authors.
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

#include "mediapipe/framework/formats/affine_transform.h"

#include <algorithm>
#include <cmath>
#include <memory>
#include <vector>

#include "absl/memory/memory.h"
#include "absl/strings/substitute.h"
#include "mediapipe/framework/port/canonical_errors.h"
#include "mediapipe/framework/port/logging.h"
#include "mediapipe/framework/port/point2.h"
#include "mediapipe/framework/port/ret_check.h"
#include "mediapipe/framework/port/status.h"
#include "mediapipe/framework/port/statusor.h"
#include "mediapipe/framework/tool/status_util.h"
#include "mediapipe/framework/type_map.h"

namespace mediapipe {
using ::mediapipe::AffineTransformData;

AffineTransform::AffineTransform() { SetScale(Point2_f(1, 1)); }

AffineTransform::AffineTransform(
    const AffineTransformData& affine_transform_data)
    : affine_transform_data_(affine_transform_data), is_dirty_(true) {
  // make sure scale is set to default (1, 1) when none provided
  if (!affine_transform_data_.has_scale()) {
    SetScale(Point2_f(1, 1));
  }
}

AffineTransform AffineTransform::Create(const Point2_f& translation,
                                        const Point2_f& scale, float rotation,
                                        const Point2_f& shear) {
  AffineTransformData affine_transform_data;

  auto* t = affine_transform_data.mutable_translation();
  t->set_x(translation.x());
  t->set_y(translation.y());

  auto* s = affine_transform_data.mutable_scale();
  s->set_x(scale.x());
  s->set_y(scale.y());

  s = affine_transform_data.mutable_shear();
  s->set_x(shear.x());
  s->set_y(shear.y());

  affine_transform_data.set_rotation(rotation);

  return AffineTransform(affine_transform_data);
}

// Accessor for the composition matrix
std::vector<float> AffineTransform::GetCompositionMatrix() {
  float r = affine_transform_data_.rotation();
  const auto t = affine_transform_data_.translation();
  const auto sc = affine_transform_data_.scale();
  const auto sh = affine_transform_data_.shear();

  if (is_dirty_) {
    // Composition matrix M = T*R*Sh*Sc
    // Column based to match GL matrix store order
    float cos_r = std::cos(r);
    float sin_r = std::sin(r);
    matrix_[0] = (cos_r + sin_r * -sh.y()) * sc.x();
    matrix_[1] = (-sin_r + cos_r * -sh.y()) * sc.x();
    matrix_[2] = 0;
    matrix_[3] = (cos_r * -sh.x() + sin_r) * sc.y();
    matrix_[4] = (-sin_r * -sh.x() + cos_r) * sc.y();
    matrix_[5] = 0;
    matrix_[6] = t.x();
    matrix_[7] = -t.y();
    matrix_[8] = 1;
    is_dirty_ = false;
  }

  return matrix_;
}

Point2_f AffineTransform::GetScale() const {
  return Point2_f(affine_transform_data_.scale().x(),
                  affine_transform_data_.scale().y());
}

Point2_f AffineTransform::GetTranslation() const {
  return Point2_f(affine_transform_data_.translation().x(),
                  affine_transform_data_.translation().y());
}

Point2_f AffineTransform::GetShear() const {
  return Point2_f(affine_transform_data_.shear().x(),
                  affine_transform_data_.shear().y());
}

float AffineTransform::GetRotation() const {
  return affine_transform_data_.rotation();
}

void AffineTransform::SetScale(const Point2_f& scale) {
  auto* s = affine_transform_data_.mutable_scale();
  s->set_x(scale.x());
  s->set_y(scale.y());
  is_dirty_ = true;
}

void AffineTransform::SetTranslation(const Point2_f& translation) {
  auto* t = affine_transform_data_.mutable_translation();
  t->set_x(translation.x());
  t->set_y(translation.y());
  is_dirty_ = true;
}

void AffineTransform::SetShear(const Point2_f& shear) {
  auto* s = affine_transform_data_.mutable_shear();
  s->set_x(shear.x());
  s->set_y(shear.y());
  is_dirty_ = true;
}

void AffineTransform::SetRotation(float rotationInRadians) {
  affine_transform_data_.set_rotation(rotationInRadians);
  is_dirty_ = true;
}

void AffineTransform::AddScale(const Point2_f& scale) {
  auto* s = affine_transform_data_.mutable_scale();
  s->set_x(s->x() + scale.x());
  s->set_y(s->y() + scale.y());
  is_dirty_ = true;
}

void AffineTransform::AddTranslation(const Point2_f& translation) {
  auto* t = affine_transform_data_.mutable_translation();
  t->set_x(t->x() + translation.x());
  t->set_y(t->y() + translation.y());
  is_dirty_ = true;
}

void AffineTransform::AddShear(const Point2_f& shear) {
  auto* s = affine_transform_data_.mutable_shear();
  s->set_x(s->x() + shear.x());
  s->set_y(s->y() + shear.y());
  is_dirty_ = true;
}

void AffineTransform::AddRotation(float rotationInRadians) {
  affine_transform_data_.set_rotation(affine_transform_data_.rotation() +
                                      rotationInRadians);
  is_dirty_ = true;
}

void AffineTransform::SetFromProto(const AffineTransformData& proto) {
  affine_transform_data_ = proto;
}

void AffineTransform::ConvertToProto(AffineTransformData* proto) const {
  *proto = affine_transform_data_;
}

AffineTransformData AffineTransform::ConvertToProto() const {
  AffineTransformData affine_transform_data;
  ConvertToProto(&affine_transform_data);
  return affine_transform_data;
}

bool compare(float lhs, float rhs, float epsilon = 0.001f) {
  return std::fabs(lhs - rhs) < epsilon;
}

bool AffineTransform::Equals(const AffineTransform& other,
                             float epsilon) const {
  auto trans1 = GetTranslation();
  auto trans2 = other.GetTranslation();

  if (!(compare(trans1.x(), trans2.x(), epsilon) &&
        compare(trans1.y(), trans2.y(), epsilon)))
    return false;

  auto scale1 = GetScale();
  auto scale2 = other.GetScale();

  if (!(compare(scale1.x(), scale2.x(), epsilon) &&
        compare(scale1.y(), scale2.y(), epsilon)))
    return false;

  auto shear1 = GetShear();
  auto shear2 = other.GetShear();

  if (!(compare(shear1.x(), shear2.x(), epsilon) &&
        compare(shear1.y(), shear2.y(), epsilon)))
    return false;

  auto rot1 = GetRotation();
  auto rot2 = other.GetRotation();

  if (!compare(rot1, rot2, epsilon)) {
    return false;
  }

  return true;
}

bool AffineTransform::Equal(const AffineTransform& lhs,
                            const AffineTransform& rhs, float epsilon) {
  return lhs.Equals(rhs, epsilon);
}

MEDIAPIPE_REGISTER_TYPE(mediapipe::AffineTransform,
                        "::mediapipe::AffineTransform", nullptr, nullptr);

}  // namespace mediapipe
