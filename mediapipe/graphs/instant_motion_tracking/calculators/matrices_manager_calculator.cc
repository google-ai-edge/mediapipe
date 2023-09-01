// Copyright 2020 Google LLC
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

#include <cmath>
#include <memory>

#include "Eigen/Core"
#include "Eigen/Dense"
#include "Eigen/Geometry"
#include "absl/log/absl_log.h"
#include "absl/memory/memory.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/port/ret_check.h"
#include "mediapipe/framework/port/status.h"
#include "mediapipe/graphs/instant_motion_tracking/calculators/transformations.h"
#include "mediapipe/graphs/object_detection_3d/calculators/model_matrix.pb.h"
#include "mediapipe/modules/objectron/calculators/box.h"

namespace mediapipe {

namespace {
using Matrix4fCM = Eigen::Matrix<float, 4, 4, Eigen::ColMajor>;
using Vector3f = Eigen::Vector3f;
using Matrix3f = Eigen::Matrix3f;
using DiagonalMatrix3f = Eigen::DiagonalMatrix<float, 3>;
constexpr char kAnchorsTag[] = "ANCHORS";
constexpr char kIMUMatrixTag[] = "IMU_ROTATION";
constexpr char kUserRotationsTag[] = "USER_ROTATIONS";
constexpr char kUserScalingsTag[] = "USER_SCALINGS";
constexpr char kRendersTag[] = "RENDER_DATA";
constexpr char kGifAspectRatioTag[] = "GIF_ASPECT_RATIO";
constexpr char kFOVSidePacketTag[] = "FOV";
constexpr char kAspectRatioSidePacketTag[] = "ASPECT_RATIO";
// initial Z value (-10 is center point in visual range for OpenGL render)
constexpr float kInitialZ = -10.0f;
}  // namespace

// Intermediary for rotation and translation data to model matrix usable by
// gl_animation_overlay_calculator. For information on the construction of
// OpenGL objects and transformations (including a breakdown of model matrices),
// please visit: https://open.gl/transformations
//
// Input Side Packets:
//  FOV - Vertical field of view for device [REQUIRED - Defines perspective
//  matrix] ASPECT_RATIO - Aspect ratio of device [REQUIRED - Defines
//  perspective matrix]
//
// Input streams:
//  ANCHORS - Anchor data with x,y,z coordinates (x,y are in [0.0-1.0] range for
//    position on the device screen, while z is the scaling factor that changes
//    in proportion to the distance from the tracked region) [REQUIRED]
//  IMU_ROTATION - float[9] of row-major device rotation matrix [REQUIRED]
//  USER_ROTATIONS - UserRotations with corresponding radians of rotation
//    [REQUIRED]
//  USER_SCALINGS - UserScalings with corresponding scale factor [REQUIRED]
//  GIF_ASPECT_RATIO - Aspect ratio of GIF image used to dynamically scale
//    GIF asset defined as width / height [OPTIONAL]
// Output:
//  MATRICES - TimedModelMatrixProtoList of each object type to render
//    [REQUIRED]
//
// Example config:
// node{
//  calculator: "MatricesManagerCalculator"
//  input_stream: "ANCHORS:tracked_scaled_anchor_data"
//  input_stream: "IMU_ROTATION:imu_rotation_matrix"
//  input_stream: "USER_ROTATIONS:user_rotation_data"
//  input_stream: "USER_SCALINGS:user_scaling_data"
//  input_stream: "GIF_ASPECT_RATIO:gif_aspect_ratio"
//  output_stream: "MATRICES:0:first_render_matrices"
//  output_stream: "MATRICES:1:second_render_matrices" [unbounded input size]
//  input_side_packet: "FOV:vertical_fov_radians"
//  input_side_packet: "ASPECT_RATIO:aspect_ratio"
// }

class MatricesManagerCalculator : public CalculatorBase {
 public:
  static absl::Status GetContract(CalculatorContract* cc);
  absl::Status Open(CalculatorContext* cc) override;
  absl::Status Process(CalculatorContext* cc) override;

 private:
  // Device properties that will be preset by side packets
  float vertical_fov_radians_ = 0.0f;
  float aspect_ratio_ = 0.0f;
  float gif_aspect_ratio_ = 1.0f;

  const Matrix3f GenerateUserRotationMatrix(const float rotation_radians) const;
  const Matrix4fCM GenerateEigenModelMatrix(
      const Vector3f& translation_vector,
      const Matrix3f& rotation_submatrix) const;
  const Vector3f GenerateAnchorVector(const Anchor& tracked_anchor) const;
  DiagonalMatrix3f GetDefaultRenderScaleDiagonal(
      const int render_id, const float user_scale_factor,
      const float gif_aspect_ratio) const;

  // Returns a user scaling increment associated with the sticker_id
  // TODO: Adjust lookup function if total number of stickers is uncapped to
  // improve performance
  const float GetUserScaler(const std::vector<UserScaling>& scalings,
                            const int sticker_id) const {
    for (const UserScaling& user_scaling : scalings) {
      if (user_scaling.sticker_id == sticker_id) {
        return user_scaling.scale_factor;
      }
    }
    ABSL_LOG(WARNING) << "Cannot find sticker_id: " << sticker_id
                      << ", returning 1.0f scaling";
    return 1.0f;
  }

  // Returns a user rotation in radians associated with the sticker_id
  const float GetUserRotation(const std::vector<UserRotation>& rotations,
                              const int sticker_id) {
    for (const UserRotation& rotation : rotations) {
      if (rotation.sticker_id == sticker_id) {
        return rotation.rotation_radians;
      }
    }
    ABSL_LOG(WARNING) << "Cannot find sticker_id: " << sticker_id
                      << ", returning 0.0f rotation";
    return 0.0f;
  }
};

REGISTER_CALCULATOR(MatricesManagerCalculator);

absl::Status MatricesManagerCalculator::GetContract(CalculatorContract* cc) {
  RET_CHECK(cc->Inputs().HasTag(kAnchorsTag) &&
            cc->Inputs().HasTag(kIMUMatrixTag) &&
            cc->Inputs().HasTag(kUserRotationsTag) &&
            cc->Inputs().HasTag(kUserScalingsTag) &&
            cc->InputSidePackets().HasTag(kFOVSidePacketTag) &&
            cc->InputSidePackets().HasTag(kAspectRatioSidePacketTag));

  cc->Inputs().Tag(kAnchorsTag).Set<std::vector<Anchor>>();
  cc->Inputs().Tag(kIMUMatrixTag).Set<float[]>();
  cc->Inputs().Tag(kUserScalingsTag).Set<std::vector<UserScaling>>();
  cc->Inputs().Tag(kUserRotationsTag).Set<std::vector<UserRotation>>();
  cc->Inputs().Tag(kRendersTag).Set<std::vector<int>>();
  if (cc->Inputs().HasTag(kGifAspectRatioTag)) {
    cc->Inputs().Tag(kGifAspectRatioTag).Set<float>();
  }

  for (CollectionItemId id = cc->Outputs().BeginId("MATRICES");
       id < cc->Outputs().EndId("MATRICES"); ++id) {
    cc->Outputs().Get(id).Set<mediapipe::TimedModelMatrixProtoList>();
  }
  cc->InputSidePackets().Tag(kFOVSidePacketTag).Set<float>();
  cc->InputSidePackets().Tag(kAspectRatioSidePacketTag).Set<float>();

  return absl::OkStatus();
}

absl::Status MatricesManagerCalculator::Open(CalculatorContext* cc) {
  cc->SetOffset(TimestampDiff(0));
  // Set device properties from side packets
  vertical_fov_radians_ =
      cc->InputSidePackets().Tag(kFOVSidePacketTag).Get<float>();
  aspect_ratio_ =
      cc->InputSidePackets().Tag(kAspectRatioSidePacketTag).Get<float>();
  return absl::OkStatus();
}

absl::Status MatricesManagerCalculator::Process(CalculatorContext* cc) {
  // Define each object's model matrices
  auto asset_matrices_gif =
      std::make_unique<mediapipe::TimedModelMatrixProtoList>();
  auto asset_matrices_1 =
      std::make_unique<mediapipe::TimedModelMatrixProtoList>();
  // Clear all model matrices
  asset_matrices_gif->clear_model_matrix();
  asset_matrices_1->clear_model_matrix();

  const std::vector<UserRotation> user_rotation_data =
      cc->Inputs().Tag(kUserRotationsTag).Get<std::vector<UserRotation>>();

  const std::vector<UserScaling> user_scaling_data =
      cc->Inputs().Tag(kUserScalingsTag).Get<std::vector<UserScaling>>();

  const std::vector<int> render_data =
      cc->Inputs().Tag(kRendersTag).Get<std::vector<int>>();

  const std::vector<Anchor> anchor_data =
      cc->Inputs().Tag(kAnchorsTag).Get<std::vector<Anchor>>();
  if (cc->Inputs().HasTag(kGifAspectRatioTag) &&
      !cc->Inputs().Tag(kGifAspectRatioTag).IsEmpty()) {
    gif_aspect_ratio_ = cc->Inputs().Tag(kGifAspectRatioTag).Get<float>();
  }

  // Device IMU rotation submatrix
  const auto imu_matrix = cc->Inputs().Tag(kIMUMatrixTag).Get<float[]>();
  Matrix3f imu_rotation_submatrix;
  int idx = 0;
  for (int x = 0; x < 3; ++x) {
    for (int y = 0; y < 3; ++y) {
      // Input matrix is row-major matrix, it must be reformatted to
      // column-major via transpose procedure
      imu_rotation_submatrix(y, x) = imu_matrix[idx++];
    }
  }

  int render_idx = 0;
  for (const Anchor& anchor : anchor_data) {
    const int id = anchor.sticker_id;
    mediapipe::TimedModelMatrixProto* model_matrix;
    // Add model matrix to matrices list for defined object render ID
    if (render_data[render_idx] == 0) {  // GIF
      model_matrix = asset_matrices_gif->add_model_matrix();
    } else {  // Asset 3D
      if (render_data[render_idx] != 1) {
        ABSL_LOG(ERROR)
            << "render id: " << render_data[render_idx]
            << " is not supported. Fall back to using render_id = 1.";
      }
      model_matrix = asset_matrices_1->add_model_matrix();
    }

    model_matrix->set_id(id);

    // The user transformation data associated with this sticker must be defined
    const float user_rotation_radians = GetUserRotation(user_rotation_data, id);
    const float user_scale_factor = GetUserScaler(user_scaling_data, id);

    // A vector representative of a user's sticker rotation transformation can
    // be created
    const Matrix3f user_rotation_submatrix =
        GenerateUserRotationMatrix(user_rotation_radians);
    // Next, the diagonal representative of the combined scaling data
    const DiagonalMatrix3f scaling_diagonal = GetDefaultRenderScaleDiagonal(
        render_data[render_idx], user_scale_factor, gif_aspect_ratio_);
    // Increment to next render id from vector
    render_idx++;

    // The user transformation data can be concatenated into a final rotation
    // submatrix with the device IMU rotational data
    const Matrix3f user_transformed_rotation_submatrix =
        imu_rotation_submatrix * user_rotation_submatrix * scaling_diagonal;

    // A vector representative of the translation of the object in OpenGL
    // coordinate space must be generated
    const Vector3f translation_vector = GenerateAnchorVector(anchor);

    // Concatenate all model matrix data
    const Matrix4fCM final_model_matrix = GenerateEigenModelMatrix(
        translation_vector, user_transformed_rotation_submatrix);

    // The generated model matrix must be mapped to TimedModelMatrixProto
    // (col-wise)
    for (int x = 0; x < final_model_matrix.rows(); ++x) {
      for (int y = 0; y < final_model_matrix.cols(); ++y) {
        model_matrix->add_matrix_entries(final_model_matrix(x, y));
      }
    }
  }

  // Output all individual render matrices
  // TODO: Perform depth ordering with gl_animation_overlay_calculator to render
  // objects in order by depth to allow occlusion.
  cc->Outputs()
      .Get(cc->Outputs().GetId("MATRICES", 0))
      .Add(asset_matrices_gif.release(), cc->InputTimestamp());
  cc->Outputs()
      .Get(cc->Outputs().GetId("MATRICES", 1))
      .Add(asset_matrices_1.release(), cc->InputTimestamp());

  return absl::OkStatus();
}

// Using a specified rotation value in radians, generate a rotation matrix for
// use with base rotation submatrix
const Matrix3f MatricesManagerCalculator::GenerateUserRotationMatrix(
    const float rotation_radians) const {
  Eigen::Matrix3f user_rotation_submatrix;
  user_rotation_submatrix =
      // The rotation in radians must be inverted to rotate the object
      // with the direction of finger movement from the user (system dependent)
      Eigen::AngleAxisf(-rotation_radians, Eigen::Vector3f::UnitY()) *
      Eigen::AngleAxisf(0.0f, Eigen::Vector3f::UnitZ()) *
      // Model orientations all assume z-axis is up, but we need y-axis upwards,
      // therefore, a +(M_PI * 0.5f) transformation must be applied
      // TODO: Bring default rotations, translations, and scalings into
      // independent sticker configuration
      Eigen::AngleAxisf(M_PI * 0.5f, Eigen::Vector3f::UnitX());
  // Matrix must be transposed due to the method of submatrix generation in
  // Eigen
  return user_rotation_submatrix.transpose();
}

// TODO: Investigate possible differences in warping of tracking speed across
// screen Using the sticker anchor data, a translation vector can be generated
// in OpenGL coordinate space
const Vector3f MatricesManagerCalculator::GenerateAnchorVector(
    const Anchor& tracked_anchor) const {
  // Using an initial z-value in OpenGL space, generate a new base z-axis value
  // to mimic scaling by distance.
  const float z = kInitialZ * tracked_anchor.z;

  // Using triangle geometry, the minimum for a y-coordinate that will appear in
  // the view field for the given z value above can be found.
  const float y_half_range = z * (tan(vertical_fov_radians_ * 0.5f));

  // The aspect ratio of the device and y_minimum calculated above can be used
  // to find the minimum value for x that will appear in the view field of the
  // device screen.
  const float x_half_range = y_half_range * aspect_ratio_;

  // Given the minimum bounds of the screen in OpenGL space, the tracked anchor
  // coordinates can be converted to OpenGL coordinate space.
  //
  // (i.e: X and Y will be converted from [0.0-1.0] space to [x_minimum,
  // -x_minimum] space and [y_minimum, -y_minimum] space respectively)
  const float x = (-2.0f * tracked_anchor.x * x_half_range) + x_half_range;
  const float y = (-2.0f * tracked_anchor.y * y_half_range) + y_half_range;

  // A translation transformation vector can be generated via Eigen
  const Vector3f t_vector(x, y, z);
  return t_vector;
}

// Generates a model matrix via Eigen with appropriate transformations
const Matrix4fCM MatricesManagerCalculator::GenerateEigenModelMatrix(
    const Vector3f& translation_vector,
    const Matrix3f& rotation_submatrix) const {
  // Define basic empty model matrix
  Matrix4fCM mvp_matrix;

  // Set the translation vector
  mvp_matrix.topRightCorner<3, 1>() = translation_vector;

  // Set the rotation submatrix
  mvp_matrix.topLeftCorner<3, 3>() = rotation_submatrix;

  // Set trailing 1.0 required by OpenGL to define coordinate space
  mvp_matrix(3, 3) = 1.0f;

  return mvp_matrix;
}

// This returns a scaling matrix to alter the projection matrix for
// the specified render id in order to ensure all objects render at a similar
// size in the view screen upon initial placement
DiagonalMatrix3f MatricesManagerCalculator::GetDefaultRenderScaleDiagonal(
    const int render_id, const float user_scale_factor,
    const float gif_aspect_ratio) const {
  float scale_preset = 1.0f;
  float x_scalar = 1.0f;
  float y_scalar = 1.0f;

  switch (render_id) {
    case 0: {  // GIF
      // 160 is the scaling preset to make the GIF asset appear relatively
      // similar in size to all other assets
      scale_preset = 160.0f;
      if (gif_aspect_ratio >= 1.0f) {
        // GIF is wider horizontally (scale on x-axis)
        x_scalar = gif_aspect_ratio;
      } else {
        // GIF is wider vertically (scale on y-axis)
        y_scalar = 1.0f / gif_aspect_ratio;
      }
      break;
    }
    case 1: {  // 3D asset
      // 5 is the scaling preset to make the 3D asset appear relatively
      // similar in size to all other assets
      scale_preset = 5.0f;
      break;
    }
    default: {
      ABSL_LOG(INFO) << "Unsupported render_id: " << render_id
                     << ", returning default render_scale";
      break;
    }
  }

  DiagonalMatrix3f scaling(scale_preset * user_scale_factor * x_scalar,
                           scale_preset * user_scale_factor * y_scalar,
                           scale_preset * user_scale_factor);
  return scaling;
}
}  // namespace mediapipe
