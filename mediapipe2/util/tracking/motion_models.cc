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

#include "mediapipe/util/tracking/motion_models.h"

#include <stdlib.h>

#include <cmath>
#include <string>
#include <utility>

#include "Eigen/Core"
#include "Eigen/Dense"
#include "absl/strings/str_format.h"

// Set to true to use catmull rom mixture weights instead of Gaussian weights
// for homography mixture estimation.
bool flags_catmull_rom_mixture_weights = false;

namespace mediapipe {

std::string ModelAdapter<TranslationModel>::ToString(
    const TranslationModel& model) {
  return absl::StrFormat("%7f", model.dx()) + " " +
         absl::StrFormat("%7f", model.dy());
}

AffineModel ModelAdapter<TranslationModel>::ToAffine(
    const TranslationModel& model) {
  return AffineAdapter::FromArgs(model.dx(), model.dy(), 1, 0, 0, 1);
}

TranslationModel ModelAdapter<TranslationModel>::FromAffine(
    const AffineModel& model) {
  DCHECK_EQ(model.a(), 1);
  DCHECK_EQ(model.b(), 0);
  DCHECK_EQ(model.c(), 0);
  DCHECK_EQ(model.d(), 1);

  return TranslationAdapter::FromArgs(model.dx(), model.dy());
}

Homography ModelAdapter<TranslationModel>::ToHomography(
    const TranslationModel& model) {
  return HomographyAdapter::FromAffine(ToAffine(model));
}

TranslationModel ModelAdapter<TranslationModel>::FromHomography(
    const Homography& model) {
  return TranslationAdapter::FromAffine(HomographyAdapter::ToAffine(model));
}

void ModelAdapter<TranslationModel>::GetJacobianAtPoint(const Vector2_f& pt,
                                                        float* jacobian) {
  DCHECK(jacobian);
  jacobian[0] = 1;
  jacobian[1] = 0;
  jacobian[2] = 0;
  jacobian[3] = 1;
}

TranslationModel ModelAdapter<TranslationModel>::NormalizationTransform(
    float frame_width, float frame_height) {
  // Independent of frame size.
  return TranslationModel();
}

TranslationModel ModelAdapter<TranslationModel>::Maximum(
    const TranslationModel& lhs, const TranslationModel& rhs) {
  return FromArgs(std::max(lhs.dx(), rhs.dx()), std::max(lhs.dy(), rhs.dy()));
}

TranslationModel ModelAdapter<TranslationModel>::ProjectFrom(
    const LinearSimilarityModel& model, float frame_width, float frame_height) {
  return LinearSimilarityAdapter::ProjectToTranslation(model, frame_width,
                                                       frame_height);
}

TranslationModel ModelAdapter<TranslationModel>::ProjectFrom(
    const AffineModel& model, float frame_width, float frame_height) {
  return ProjectFrom(AffineAdapter::ProjectToLinearSimilarity(
                         model, frame_width, frame_height),
                     frame_width, frame_height);
}

TranslationModel ModelAdapter<TranslationModel>::ProjectFrom(
    const Homography& model, float frame_width, float frame_height) {
  return ProjectFrom(
      HomographyAdapter::ProjectToAffine(model, frame_width, frame_height),
      frame_width, frame_height);
}

// Similarity model.
SimilarityModel ModelAdapter<SimilarityModel>::FromArgs(float dx, float dy,
                                                        float scale,
                                                        float rotation) {
  SimilarityModel model;
  model.set_dx(dx);
  model.set_dy(dy);
  model.set_scale(scale);
  model.set_rotation(rotation);
  return model;
}

SimilarityModel ModelAdapter<SimilarityModel>::FromFloatPointer(
    const float* args, bool identity_parametrization) {
  DCHECK(args);
  SimilarityModel model;
  model.set_dx(args[0]);
  model.set_dy(args[1]);
  model.set_scale((identity_parametrization ? 1.f : 0.f) + args[2]);
  model.set_rotation(args[3]);
  return model;
}

SimilarityModel ModelAdapter<SimilarityModel>::FromDoublePointer(
    const double* args, bool identity_parametrization) {
  DCHECK(args);
  SimilarityModel model;
  model.set_dx(args[0]);
  model.set_dy(args[1]);
  model.set_scale((identity_parametrization ? 1.f : 0.f) + args[2]);
  model.set_rotation(args[3]);
  return model;
}

Vector2_f ModelAdapter<SimilarityModel>::TransformPoint(
    const SimilarityModel& model, const Vector2_f& pt) {
  // Setup rotation part.
  const float c_r = std::cos(model.rotation());
  const float c_s = std::sin(model.rotation());
  const Vector2_f pt_rot(c_r * pt.x() - c_s * pt.y(),
                         c_s * pt.x() + c_r * pt.y());

  return pt_rot * model.scale() + Vector2_f(model.dx(), model.dy());
}

SimilarityModel ModelAdapter<SimilarityModel>::Invert(
    const SimilarityModel& model) {
  bool success = true;
  const SimilarityModel result = InvertChecked(model, &success);
  if (!success) {
    LOG(ERROR) << "Model not invertible. Returning identity.";
    return SimilarityModel();
  } else {
    return result;
  }
}

SimilarityModel ModelAdapter<SimilarityModel>::InvertChecked(
    const SimilarityModel& model, bool* success) {
  SimilarityModel inv_model;
  inv_model.set_rotation(-model.rotation());

  if (fabs(model.scale()) > kDetInvertibleEps) {
    inv_model.set_scale(1.0 / model.scale());
    *success = true;
  } else {
    *success = false;
    VLOG(1) << "Model is not invertible.";
    return SimilarityModel();
  }

  // Rotate and scale [dx, dy] by inv_model.
  const float c_r = std::cos(inv_model.rotation());
  const float c_s = std::sin(inv_model.rotation());
  const Vector2_f pt_rot(c_r * model.dx() - c_s * model.dy(),
                         c_s * model.dx() + c_r * model.dy());

  const Vector2_f inv_trans = -pt_rot * inv_model.scale();

  inv_model.set_dx(inv_trans.x());
  inv_model.set_dy(inv_trans.y());

  return inv_model;
}

SimilarityModel ModelAdapter<SimilarityModel>::Compose(
    const SimilarityModel& lhs, const SimilarityModel& rhs) {
  SimilarityModel result;
  result.set_scale(lhs.scale() * rhs.scale());
  result.set_rotation(lhs.rotation() + rhs.rotation());

  // Apply lhs scale and rot to rhs translation.
  const float c_r = std::cos(lhs.rotation());
  const float c_s = std::sin(lhs.rotation());
  const Vector2_f trans_rot(c_r * rhs.dx() - c_s * rhs.dy(),
                            c_s * rhs.dx() + c_r * rhs.dy());

  const Vector2_f trans_concat =
      trans_rot * lhs.scale() + Vector2_f(lhs.dx(), lhs.dy());

  result.set_dx(trans_concat.x());
  result.set_dy(trans_concat.y());
  return result;
}

float ModelAdapter<SimilarityModel>::GetParameter(const SimilarityModel& model,
                                                  int id) {
  switch (id) {
    case 0:
      return model.dx();
    case 1:
      return model.dy();
    case 2:
      return model.scale();
    case 3:
      return model.rotation();
    default:
      LOG(FATAL) << "Parameter id is out of bounds";
  }

  return 0;
}

std::string ModelAdapter<SimilarityModel>::ToString(
    const SimilarityModel& model) {
  return absl::StrFormat("%7f", model.dx()) + " " +
         absl::StrFormat("%7f", model.dy()) + " " +
         absl::StrFormat("%7f", model.scale()) + " " +
         absl::StrFormat("%7f", model.rotation());
}

SimilarityModel ModelAdapter<SimilarityModel>::NormalizationTransform(
    float frame_width, float frame_height) {
  const float scale = std::hypot(frame_width, frame_height);
  DCHECK_NE(scale, 0);
  return SimilarityAdapter::FromArgs(0, 0, 1.0 / scale, 0);
}

TranslationModel ModelAdapter<SimilarityModel>::ProjectToTranslation(
    const SimilarityModel& model, float frame_width, float frame_height) {
  return LinearSimilarityAdapter::ProjectToTranslation(
      LinearSimilarityAdapter::FromSimilarity(model), frame_width,
      frame_height);
}

std::string ModelAdapter<LinearSimilarityModel>::ToString(
    const LinearSimilarityModel& model) {
  return absl::StrFormat("%7f", model.dx()) + " " +
         absl::StrFormat("%7f", model.dy()) + " " +
         absl::StrFormat("%7f", model.a()) + " " +
         absl::StrFormat("%7f", model.b());
}

AffineModel ModelAdapter<LinearSimilarityModel>::ToAffine(
    const LinearSimilarityModel& model) {
  return AffineAdapter::FromArgs(model.dx(), model.dy(), model.a(), -model.b(),
                                 model.b(), model.a());
}

LinearSimilarityModel ModelAdapter<LinearSimilarityModel>::FromAffine(
    const AffineModel& model) {
  DCHECK_EQ(model.a(), model.d());
  DCHECK_EQ(model.b(), -model.c());

  return LinearSimilarityAdapter::FromArgs(model.dx(), model.dy(), model.a(),
                                           -model.b());
}

Homography ModelAdapter<LinearSimilarityModel>::ToHomography(
    const LinearSimilarityModel& model) {
  return HomographyAdapter::FromAffine(ToAffine(model));
}

LinearSimilarityModel ModelAdapter<LinearSimilarityModel>::FromHomography(
    const Homography& model) {
  return LinearSimilarityAdapter::FromAffine(
      HomographyAdapter::ToAffine(model));
}

SimilarityModel ModelAdapter<LinearSimilarityModel>::ToSimilarity(
    const LinearSimilarityModel& model) {
  const float scale = std::hypot(model.a(), model.b());
  return SimilarityAdapter::FromArgs(model.dx(), model.dy(), scale,
                                     std::atan2(model.b(), model.a()));
}

LinearSimilarityModel ModelAdapter<LinearSimilarityModel>::FromSimilarity(
    const SimilarityModel& model) {
  return LinearSimilarityAdapter::FromArgs(
      model.dx(), model.dy(), model.scale() * std::cos(model.rotation()),
      model.scale() * std::sin(model.rotation()));
}

LinearSimilarityModel ModelAdapter<LinearSimilarityModel>::ScaleParameters(
    const LinearSimilarityModel& model_in, float scale) {
  LinearSimilarityModel model = model_in;
  model.set_a(model.a() * scale);
  model.set_b(model.b() * scale);
  model.set_dx(model.dx() * scale);
  model.set_dy(model.dy() * scale);
  return model;
}

LinearSimilarityModel ModelAdapter<LinearSimilarityModel>::AddIdentity(
    const LinearSimilarityModel& model_in) {
  LinearSimilarityModel model = model_in;
  model.set_a(model.a() + 1);
  return model;
}

void ModelAdapter<LinearSimilarityModel>::GetJacobianAtPoint(
    const Vector2_f& pt, float* jacobian) {
  DCHECK(jacobian);
  // First row.
  jacobian[0] = 1;
  jacobian[1] = 0;
  jacobian[2] = pt.x();
  jacobian[3] = -pt.y();
  // Second row.
  jacobian[4] = 0;
  jacobian[5] = 1;
  jacobian[6] = pt.y();
  jacobian[7] = pt.x();
}

LinearSimilarityModel
ModelAdapter<LinearSimilarityModel>::NormalizationTransform(
    float frame_width, float frame_height) {
  const float scale = std::hypot(frame_width, frame_height);
  DCHECK_NE(scale, 0);
  return LinearSimilarityAdapter::FromArgs(0, 0, 1.0 / scale, 0);
}

LinearSimilarityModel ModelAdapter<LinearSimilarityModel>::Maximum(
    const LinearSimilarityModel& lhs, const LinearSimilarityModel& rhs) {
  return FromArgs(std::max(lhs.dx(), rhs.dx()), std::max(lhs.dy(), rhs.dy()),
                  std::max(lhs.a(), rhs.a()), std::max(lhs.b(), rhs.b()));
}

TranslationModel ModelAdapter<LinearSimilarityModel>::ProjectToTranslation(
    const LinearSimilarityModel& model, float frame_width, float frame_height) {
  LinearSimilarityModel center_trans = LinearSimilarityAdapter::FromArgs(
      frame_width * 0.5f, frame_height * 0.5f, 1, 0);
  LinearSimilarityModel inv_center_trans = LinearSimilarityAdapter::FromArgs(
      -frame_width * 0.5f, -frame_height * 0.5f, 1, 0);

  // Express model w.r.t. center.
  LinearSimilarityModel center_model =
      ModelCompose3(inv_center_trans, model, center_trans);

  // No need to shift back to top-left after decomposition, as translations
  // are independent from coordinate origin.
  return TranslationAdapter::FromArgs(center_model.dx(), center_model.dy());
}

std::string ModelAdapter<AffineModel>::ToString(const AffineModel& model) {
  return absl::StrFormat("%7f", model.dx()) + " " +
         absl::StrFormat("%7f", model.dy()) + " " +
         absl::StrFormat("%7f", model.a()) + " " +
         absl::StrFormat("%7f", model.b()) + " " +
         absl::StrFormat("%7f", model.c()) + " " +
         absl::StrFormat("%7f", model.d());
}

AffineModel ModelAdapter<AffineModel>::NormalizationTransform(
    float frame_width, float frame_height) {
  const float scale = std::hypot(frame_width, frame_height);
  DCHECK_NE(scale, 0);
  return AffineAdapter::FromArgs(0, 0, 1.0f / scale, 0, 0, 1.0f / scale);
}

Homography ModelAdapter<AffineModel>::ToHomography(const AffineModel& model) {
  float params[8] = {model.a(), model.b(),  model.dx(), model.c(),
                     model.d(), model.dy(), 0,          0};
  return HomographyAdapter::FromFloatPointer(params, false);
}

AffineModel ModelAdapter<AffineModel>::FromHomography(const Homography& model) {
  DCHECK_EQ(model.h_20(), 0);
  DCHECK_EQ(model.h_21(), 0);

  float params[6] = {model.h_02(), model.h_12(),   // dx, dy
                     model.h_00(), model.h_01(),   // a, b
                     model.h_10(), model.h_11()};  // c, d

  return FromFloatPointer(params, false);
}

AffineModel ModelAdapter<AffineModel>::ScaleParameters(
    const AffineModel& model_in, float scale) {
  AffineModel model = model_in;
  model.set_a(model.a() * scale);
  model.set_b(model.b() * scale);
  model.set_c(model.c() * scale);
  model.set_d(model.d() * scale);
  model.set_dx(model.dx() * scale);
  model.set_dy(model.dy() * scale);
  return model;
}

AffineModel ModelAdapter<AffineModel>::AddIdentity(
    const AffineModel& model_in) {
  AffineModel model = model_in;
  model.set_a(model.a() + 1);
  model.set_d(model.d() + 1);
  return model;
}

void ModelAdapter<AffineModel>::GetJacobianAtPoint(const Vector2_f& pt,
                                                   float* jacobian) {
  DCHECK(jacobian);
  // First row.
  jacobian[0] = 1;
  jacobian[1] = 0;
  jacobian[2] = pt.x();
  jacobian[3] = pt.y();
  jacobian[4] = 0;
  jacobian[5] = 0;

  // Second row.
  jacobian[6] = 0;
  jacobian[7] = 1;
  jacobian[8] = 0;
  jacobian[9] = 0;
  jacobian[10] = pt.x();
  jacobian[11] = pt.y();
}

AffineModel ModelAdapter<AffineModel>::Maximum(const AffineModel& lhs,
                                               const AffineModel& rhs) {
  return FromArgs(std::max(lhs.dx(), rhs.dx()), std::max(lhs.dy(), rhs.dy()),
                  std::max(lhs.a(), rhs.a()), std::max(lhs.b(), rhs.b()),
                  std::max(lhs.c(), rhs.c()), std::max(lhs.d(), rhs.d()));
}

LinearSimilarityModel ModelAdapter<LinearSimilarityModel>::ProjectFrom(
    const AffineModel& model, float frame_width, float frame_height) {
  return AffineAdapter::ProjectToLinearSimilarity(model, frame_width,
                                                  frame_height);
}

LinearSimilarityModel ModelAdapter<LinearSimilarityModel>::ProjectFrom(
    const Homography& model, float frame_width, float frame_height) {
  return ProjectFrom(
      AffineAdapter::ProjectFrom(model, frame_width, frame_height), frame_width,
      frame_height);
}

LinearSimilarityModel ModelAdapter<AffineModel>::ProjectToLinearSimilarity(
    const AffineModel& model, float frame_width, float frame_height) {
  AffineModel center_trans = AffineAdapter::FromArgs(
      frame_width * 0.5f, frame_height * 0.5f, 1, 0, 0, 1);
  AffineModel inv_center_trans = AffineAdapter::FromArgs(
      -frame_width * 0.5f, -frame_height * 0.5f, 1, 0, 0, 1);

  // Express model w.r.t. center.
  AffineModel center_model =
      ModelCompose3(inv_center_trans, model, center_trans);

  // Determine average scale.
  const float scale = std::sqrt(AffineAdapter::Determinant(center_model));

  // Goal is approximate matrix:
  // (a   b           (a' -b'
  //  c   d)  with     b'  a')
  //
  //  :=  :=
  //  v_1 v_2
  //  After normalization by the scale, a' = cos(u) and b' = sin(u)
  //  therefore, the columns on the right hand side have norm 1 and are
  //  orthogonal.
  //
  //  ==> Orthonormalize v_1 and v_2

  Vector2_f x_map(center_model.a(), center_model.c());  // == v_1
  Vector2_f y_map(center_model.b(), center_model.d());  // == v_2
  x_map.Normalize();
  y_map.Normalize();

  // Two approaches here
  // A) Perform Grahm Schmidt, i.e. QR decomp / polar decomposition, i.e.:
  // y_map' = y_map - <y_map, x_map> * x_map;
  //   ==> no error in x_map, error increasing with larger y
  // B) Compute  the middle vector between x_map and y_map and create
  //    orthogonal system from that
  //    (rotate by -45':  [ 1  1
  //                        -1 1]  * 1/sqrt(2)
  //    Error equally distributed between x and y.
  //
  // Comparing approach A and B:
  //
  //             video 1 (gleicher4)      video 2 (pool dance)
  // Method B
  // Max diff : dx: 1.6359973             4.600112
  //            dy: 2.1076841             11.51579
  //            a: 1.004498               1.01036
  //            b: 0.0047194548           0.027450036
  // Method A
  // Max diff : dx: 4.3549204             14.145205
  //            dy: 2.4496114             7.7804289
  //            a: 1.0136091              1.043335
  //            b: 0.0024313219           0.0065769218

  // Using method B:
  Vector2_f middle = (x_map + y_map).Normalize();

  Vector2_f a_b = Vector2_f(middle.x() + middle.y(),  // see above matrix.
                            middle.y() - middle.x())
                      .Normalize();
  AffineModel lin_approx = AffineAdapter::FromArgs(
      center_model.dx(), center_model.dy(), scale * a_b.x(), -scale * a_b.y(),
      scale * a_b.y(), scale * a_b.x());

  // Map back to top-left origin.
  return LinearSimilarityAdapter::FromAffine(
      ModelCompose3(center_trans, lin_approx, inv_center_trans));
}

AffineModel ModelAdapter<AffineModel>::ProjectFrom(const Homography& model,
                                                   float frame_width,
                                                   float frame_height) {
  return HomographyAdapter::ProjectToAffine(model, frame_width, frame_height);
}

Homography ModelAdapter<Homography>::InvertChecked(const Homography& model,
                                                   bool* success) {
  // Could do adjoint method and do it by hand. Use Eigen for now, not that
  // crucial at this point.

  Eigen::Matrix3d model_mat;
  model_mat(0, 0) = model.h_00();
  model_mat(0, 1) = model.h_01();
  model_mat(0, 2) = model.h_02();
  model_mat(1, 0) = model.h_10();
  model_mat(1, 1) = model.h_11();
  model_mat(1, 2) = model.h_12();
  model_mat(2, 0) = model.h_20();
  model_mat(2, 1) = model.h_21();
  model_mat(2, 2) = 1.0f;

  if (model_mat.determinant() < kDetInvertibleEps) {
    VLOG(1) << "Homography not invertible, det is zero.";
    *success = false;
    return Homography();
  }

  Eigen::Matrix3d inv_model_mat = model_mat.inverse();

  if (inv_model_mat(2, 2) == 0) {
    LOG(ERROR) << "Degenerate homography. See proto.";
    *success = false;
    return Homography();
  }

  *success = true;
  Homography inv_model;
  const float scale = 1.0f / inv_model_mat(2, 2);
  inv_model.set_h_00(inv_model_mat(0, 0) * scale);
  inv_model.set_h_01(inv_model_mat(0, 1) * scale);
  inv_model.set_h_02(inv_model_mat(0, 2) * scale);
  inv_model.set_h_10(inv_model_mat(1, 0) * scale);
  inv_model.set_h_11(inv_model_mat(1, 1) * scale);
  inv_model.set_h_12(inv_model_mat(1, 2) * scale);
  inv_model.set_h_20(inv_model_mat(2, 0) * scale);
  inv_model.set_h_21(inv_model_mat(2, 1) * scale);

  return inv_model;
}

std::string ModelAdapter<Homography>::ToString(const Homography& model) {
  return absl::StrFormat("%7f", model.h_00()) + " " +
         absl::StrFormat("%7f", model.h_01()) + " " +
         absl::StrFormat("%7f", model.h_02()) + " " +
         absl::StrFormat("%7f", model.h_10()) + " " +
         absl::StrFormat("%7f", model.h_11()) + " " +
         absl::StrFormat("%7f", model.h_12()) + " " +
         absl::StrFormat("%7f", model.h_20()) + " " +
         absl::StrFormat("%7f", model.h_21());
}

AffineModel ModelAdapter<Homography>::ToAffine(const Homography& model) {
  DCHECK_EQ(model.h_20(), 0);
  DCHECK_EQ(model.h_21(), 0);
  AffineModel affine_model;
  affine_model.set_a(model.h_00());
  affine_model.set_b(model.h_01());
  affine_model.set_dx(model.h_02());
  affine_model.set_c(model.h_10());
  affine_model.set_d(model.h_11());
  affine_model.set_dy(model.h_12());
  return affine_model;
}

Homography ModelAdapter<Homography>::FromAffine(const AffineModel& model) {
  return Embed(model);
}

bool ModelAdapter<Homography>::IsAffine(const Homography& model) {
  return model.h_20() == 0 && model.h_21() == 0;
}

void ModelAdapter<Homography>::GetJacobianAtPoint(const Vector2_f& pt,
                                                  float* jacobian) {
  DCHECK(jacobian);
  // First row.
  jacobian[0] = pt.x();
  jacobian[1] = pt.y();
  jacobian[2] = 1;
  jacobian[3] = 0;
  jacobian[4] = 0;
  jacobian[5] = 0;
  jacobian[6] = -pt.x() * pt.x();
  jacobian[7] = -pt.x() * pt.y();

  // Second row.
  jacobian[8] = 0;
  jacobian[9] = 0;
  jacobian[10] = 0;
  jacobian[11] = pt.x();
  jacobian[12] = pt.y();
  jacobian[13] = 1;
  jacobian[14] = -pt.x() * pt.y();
  jacobian[15] = -pt.y() * pt.y();
}

Homography ModelAdapter<Homography>::NormalizationTransform(
    float frame_width, float frame_height) {
  const float scale = std::hypot(frame_width, frame_height);
  DCHECK_NE(scale, 0);
  return HomographyAdapter::FromArgs(1.0f / scale, 0, 0, 0, 1.0f / scale, 0, 0,
                                     0);
}

AffineModel ModelAdapter<Homography>::ProjectToAffine(const Homography& model,
                                                      float frame_width,
                                                      float frame_height) {
  Homography center_trans;
  center_trans.set_h_02(frame_width * 0.5f);
  center_trans.set_h_12(frame_height * 0.5f);

  Homography inv_center_trans;
  inv_center_trans.set_h_02(-frame_width * 0.5f);
  inv_center_trans.set_h_12(-frame_height * 0.5f);

  // Express model w.r.t. center.
  Homography center_model =
      ModelCompose3(inv_center_trans, model, center_trans);

  // Zero out perspective.
  center_model.set_h_20(0);
  center_model.set_h_21(0);

  // Map back to top left and embed.
  return ToAffine(ModelCompose3(center_trans, center_model, inv_center_trans));
}

namespace {

// Returns true if non-empty intersection is present. In this case, start and
// end are clipped to rect, specifically after clipping the line
// [start, end] is inside the rectangle or incident to the boundary
// of the rectangle. If strict is set to true, [start, end] is strictly inside
// the rectangle, i.e. not incident to the boundary (minus start and end
// point which still can be incident).
// Implemented using Liang-Barsky algorithm.
// http://en.wikipedia.org/wiki/Liang%E2%80%93Barsky
inline bool ClipLine(const Vector2_f& rect, bool strict, Vector2_f* start,
                     Vector2_f* end) {
  Vector2_f diff = *end - *start;
  float p[4] = {-diff.x(), diff.x(), -diff.y(), diff.y()};

  // Bounds are (x_min, y_min) = (0, 0)
  //            (x_max, y_max) = rect
  float q[4] = {start->x() - 0, rect.x() - start->x(), start->y() - 0,
                rect.y() - start->y()};

  // Compute parametric intersection points.
  float near = -1e10;
  float far = 1e10;
  for (int k = 0; k < 4; ++k) {
    if (fabs(p[k]) < 1e-6f) {
      // Line is parallel to one axis of rectangle.
      if ((strict && q[k] <= 0.0f) || q[k] < 0.0f) {
        // Line is outside rectangle.
        return false;
      } else {
        // Possible intersection along other dimensions.
        continue;
      }
    } else {
      // Line is not parallel -> compute intersection.
      float intersect = q[k] / p[k];
      // Sign of p determines if near or far parameter.
      if (p[k] < 0.0f) {
        near = std::max(near, intersect);
      } else {
        far = std::min(far, intersect);
      }
    }
  }

  if (near > far) {
    // Line is outside of rectangle.
    return false;
  }

  // Clip near and far to valid line segment interval [0, 1].
  far = std::min(far, 1.0f);
  near = std::max(near, 0.0f);

  if (near <= far) {
    // Non-empty intersection. Single points are considered valid intersection.
    *end = *start + diff * far;
    *start = *start + diff * near;
    return true;
  } else {
    // Empty intersection.
    return false;
  }
}

}  // namespace.

template <class Model>
float ModelMethods<Model>::NormalizedIntersectionArea(const Model& model_1,
                                                      const Model& model_2,
                                                      const Vector2_f& rect) {
  const float rect_area = rect.x() * rect.y();
  if (rect_area <= 0) {
    LOG(WARNING) << "Empty rectangle passed -> empty intersection.";
    return 0.0f;
  }

  std::vector<std::pair<Vector2_f, Vector2_f>> lines(4);
  lines[0] = std::make_pair(Vector2_f(0, 0), Vector2_f(0, rect.y()));
  lines[1] =
      std::make_pair(Vector2_f(0, rect.y()), Vector2_f(rect.x(), rect.y()));
  lines[2] =
      std::make_pair(Vector2_f(rect.x(), rect.y()), Vector2_f(rect.x(), 0));
  lines[3] = std::make_pair(Vector2_f(rect.x(), 0), Vector2_f(0, 0));

  float model_1_area = 0.0f;
  float model_2_area = 0.0f;
  for (int k = 0; k < 4; ++k) {
    Vector2_f start_1 = Adapter::TransformPoint(model_1, lines[k].first);
    Vector2_f end_1 = Adapter::TransformPoint(model_1, lines[k].second);
    // Use trapezoidal rule for polygon area.
    model_1_area += 0.5 * (end_1.y() + start_1.y()) * (end_1.x() - start_1.x());
    Vector2_f start_2 = Adapter::TransformPoint(model_2, lines[k].first);
    Vector2_f end_2 = Adapter::TransformPoint(model_2, lines[k].second);
    model_2_area += 0.5 * (end_2.y() + start_2.y()) * (end_2.x() - start_2.x());
  }

  const float average_area = 0.5f * (model_1_area + model_2_area);
  if (average_area <= 0) {
    LOG(WARNING) << "Degenerative models passed -> empty intersection.";
    return 0.0f;
  }

  // First, clip transformed rectangle against origin defined by model_1.
  bool success = true;
  Model diff = ModelDiffChecked(model_2, model_1, &success);
  if (!success) {
    LOG(WARNING) << "Model difference is singular -> empty intersection.";
    return 0.0f;
  }

  float area = 0.0f;
  for (int k = 0; k < 4; ++k) {
    Vector2_f start_1 = Adapter::TransformPoint(diff, lines[k].first);
    Vector2_f end_1 = Adapter::TransformPoint(diff, lines[k].second);
    if (ClipLine(rect, false, &start_1, &end_1)) {
      // Non-empty intersection.
      // Transform intersection back to world coordinate system.
      const Vector2_f start = Adapter::TransformPoint(model_1, start_1);
      const Vector2_f end = Adapter::TransformPoint(model_1, end_1);
      // Use trapezoidal rule for polygon area without explicit ordering of
      // vertices.
      area += 0.5 * (end.y() + start.y()) * (end.x() - start.x());
    }
  }

  // Second, clip transformed rectangle against origin defined by model_2.
  Model inv_diff = Adapter::InvertChecked(diff, &success);
  if (!success) {
    LOG(WARNING) << "Model difference is singular -> empty intersection.";
    return 0.0f;
  }

  for (int k = 0; k < 4; ++k) {
    Vector2_f start_2 = Adapter::TransformPoint(inv_diff, lines[k].first);
    Vector2_f end_2 = Adapter::TransformPoint(inv_diff, lines[k].second);
    // Use strict comparison to address degenerate case of incident rectangles
    // in which case intersection would be counted twice if non-strict
    // comparison is employed.
    if (ClipLine(rect, true, &start_2, &end_2)) {
      // Transform start and end back to origin.
      const Vector2_f start = Adapter::TransformPoint(model_2, start_2);
      const Vector2_f end = Adapter::TransformPoint(model_2, end_2);
      area += 0.5 * (end.y() + start.y()) * (end.x() - start.x());
    }
  }

  // Normalize w.r.t. average rectangle area.
  return area / average_area;
}

MixtureRowWeights::MixtureRowWeights(int frame_height, int margin, float sigma,
                                     float y_scale, int num_models)
    : frame_height_(frame_height),
      y_scale_(y_scale),
      margin_(margin),
      sigma_(sigma),
      num_models_(num_models) {
  mid_points_.resize(num_models_);

  // Compute mid_point (row idx) for each model.
  if (flags_catmull_rom_mixture_weights) {
    const float model_height =
        static_cast<float>(frame_height_) / (num_models - 1);

    // Use Catmull-rom spline.
    // Compute weighting matrix.
    weights_.resize(frame_height * num_models);
    float spline_weights[4];

    // No margin support for splines.
    if (margin_ > 0) {
      LOG(WARNING) << "No margin support when flag catmull_rom_mixture_weights "
                   << "is set. Margin is reset to zero, it is recommended "
                   << "that RowWeightsBoundChecked is used to prevent "
                   << "segfaults.";
      margin_ = 0;
    }

    for (int i = 0; i < frame_height; ++i) {
      float* weight_ptr = &weights_[i * num_models];

      float float_pos = static_cast<float>(i) / model_height;
      int int_pos = float_pos;
      memset(weight_ptr, 0, sizeof(weight_ptr[0]) * num_models);

      float dy = float_pos - int_pos;

      // Weights sum to one, for all choices of dy.
      // For definition see
      // en.wikipedia.org/wiki/Cubic_Hermite_spline#Catmull.E2.80.93Rom_spline
      spline_weights[0] = 0.5f * (dy * ((2.0f - dy) * dy - 1.0f));
      spline_weights[1] = 0.5f * (dy * dy * (3.0f * dy - 5.0f) + 2.0f);
      spline_weights[2] = 0.5f * (dy * ((4.0f - 3.0f * dy) * dy + 1.0f));
      spline_weights[3] = 0.5f * (dy * dy * (dy - 1.0f));

      weight_ptr[int_pos] += spline_weights[1];
      if (int_pos > 0) {
        weight_ptr[int_pos - 1] += spline_weights[0];
      } else {
        weight_ptr[int_pos] += spline_weights[0];  // Double knot.
      }

      CHECK_LT(int_pos, num_models - 1);
      weight_ptr[int_pos + 1] += spline_weights[2];
      if (int_pos + 1 < num_models - 1) {
        weight_ptr[int_pos + 2] += spline_weights[3];
      } else {
        weight_ptr[int_pos + 1] += spline_weights[3];  // Double knot.
      }
    }
  } else {
    // Gaussian weights.
    const float model_height = static_cast<float>(frame_height_) / num_models;

    for (int i = 0; i < num_models; ++i) {
      mid_points_[i] = (i + 0.5f) * model_height;
    }

    // Compute gaussian weights.
    const int num_values = frame_height_ + 2 * margin_;
    std::vector<float> row_dist_weights(num_values);
    const float common = -0.5f / (sigma * sigma);
    for (int i = 0; i < num_values; ++i) {
      row_dist_weights[i] = std::exp(common * i * i);
    }

    // Compute weighting matrix.
    weights_.resize(num_values * num_models);
    for (int i = 0; i < num_values; ++i) {
      float* weight_ptr = &weights_[i * num_models];
      float weight_sum = 0;

      // Gaussian weights via lookup.
      for (int j = 0; j < num_models; ++j) {
        weight_ptr[j] = row_dist_weights[abs(i - margin_ - mid_points_[j])];
        weight_sum += weight_ptr[j];
      }

      // Normalize.
      DCHECK_GT(weight_sum, 0);
      const float inv_weight_sum = 1.0f / weight_sum;
      for (int j = 0; j < num_models; ++j) {
        weight_ptr[j] *= inv_weight_sum;
      }
    }
  }
}

float MixtureRowWeights::WeightThreshold(float frac_blocks) {
  const float model_height = static_cast<float>(frame_height_) / num_models_;

  const float y = model_height * frac_blocks + mid_points_[0];
  const float* row_weights = RowWeightsClamped(y / y_scale_);
  return row_weights[0];
}

// Explicit instantiations of ModelMethods.
template class ModelMethods<TranslationModel>;
template class ModelMethods<SimilarityModel>;
template class ModelMethods<LinearSimilarityModel>;
template class ModelMethods<AffineModel>;
template class ModelMethods<Homography>;

}  // namespace mediapipe
