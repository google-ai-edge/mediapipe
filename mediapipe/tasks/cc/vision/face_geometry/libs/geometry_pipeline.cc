// Copyright 2023 The MediaPipe Authors.
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

#include "mediapipe/tasks/cc/vision/face_geometry/libs/geometry_pipeline.h"

#include <cmath>
#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

#include "Eigen/Core"
#include "absl/memory/memory.h"
#include "mediapipe/framework/formats/landmark.pb.h"
#include "mediapipe/framework/formats/matrix.h"
#include "mediapipe/framework/formats/matrix_data.pb.h"
#include "mediapipe/framework/port/ret_check.h"
#include "mediapipe/framework/port/status.h"
#include "mediapipe/framework/port/status_macros.h"
#include "mediapipe/framework/port/statusor.h"
#include "mediapipe/tasks/cc/vision/face_geometry/libs/mesh_3d_utils.h"
#include "mediapipe/tasks/cc/vision/face_geometry/libs/procrustes_solver.h"
#include "mediapipe/tasks/cc/vision/face_geometry/libs/validation_utils.h"
#include "mediapipe/tasks/cc/vision/face_geometry/proto/environment.pb.h"
#include "mediapipe/tasks/cc/vision/face_geometry/proto/face_geometry.pb.h"
#include "mediapipe/tasks/cc/vision/face_geometry/proto/geometry_pipeline_metadata.pb.h"
#include "mediapipe/tasks/cc/vision/face_geometry/proto/mesh_3d.pb.h"

namespace mediapipe::tasks::vision::face_geometry {
namespace {

struct PerspectiveCameraFrustum {
  // NOTE: all arguments must be validated prior to calling this constructor.
  PerspectiveCameraFrustum(const proto::PerspectiveCamera& perspective_camera,
                           int frame_width, int frame_height) {
    static constexpr float kDegreesToRadians = 3.14159265358979323846f / 180.f;

    const float height_at_near =
        2.f * perspective_camera.near() *
        std::tan(0.5f * kDegreesToRadians *
                 perspective_camera.vertical_fov_degrees());

    const float width_at_near = frame_width * height_at_near / frame_height;

    left = -0.5f * width_at_near;
    right = 0.5f * width_at_near;
    bottom = -0.5f * height_at_near;
    top = 0.5f * height_at_near;
    near = perspective_camera.near();
    far = perspective_camera.far();
  }

  float left;
  float right;
  float bottom;
  float top;
  float near;
  float far;
};

class ScreenToMetricSpaceConverter {
 public:
  ScreenToMetricSpaceConverter(
      proto::OriginPointLocation origin_point_location,  //
      proto::InputSource input_source,                   //
      Eigen::Matrix3Xf&& canonical_metric_landmarks,     //
      Eigen::VectorXf&& landmark_weights,                //
      std::unique_ptr<ProcrustesSolver> procrustes_solver)
      : origin_point_location_(origin_point_location),
        input_source_(input_source),
        canonical_metric_landmarks_(std::move(canonical_metric_landmarks)),
        landmark_weights_(std::move(landmark_weights)),
        procrustes_solver_(std::move(procrustes_solver)) {}

  // Converts `screen_landmark_list` into `metric_landmark_list` and estimates
  // the `pose_transform_mat`.
  //
  // Here's the algorithm summary:
  //
  // (1) Project X- and Y- screen landmark coordinates at the Z near plane.
  //
  // (2) Estimate a canonical-to-runtime landmark set scale by running the
  //     Procrustes solver using the screen runtime landmarks.
  //
  //     On this iteration, screen landmarks are used instead of unprojected
  //     metric landmarks as it is not safe to unproject due to the relative
  //     nature of the input screen landmark Z coordinate.
  //
  // (3) Use the canonical-to-runtime scale from (2) to unproject the screen
  //     landmarks. The result is referenced as "intermediate landmarks" because
  //     they are the first estimation of the resulting metric landmarks,but are
  //     not quite there yet.
  //
  // (4) Estimate a canonical-to-runtime landmark set scale by running the
  //     Procrustes solver using the intermediate runtime landmarks.
  //
  // (5) Use the product of the scale factors from (2) and (4) to unproject
  //     the screen landmarks the second time. This is the second and the final
  //     estimation of the metric landmarks.
  //
  // (6) Multiply each of the metric landmarks by the inverse pose
  //     transformation matrix to align the runtime metric face landmarks with
  //     the canonical metric face landmarks.
  //
  // Note: the input screen landmarks are in the left-handed coordinate system,
  //       however any metric landmarks - including the canonical metric
  //       landmarks, the final runtime metric landmarks and any intermediate
  //       runtime metric landmarks - are in the right-handed coordinate system.
  //
  //       To keep the logic correct, the landmark set handedness is changed any
  //       time the screen-to-metric semantic barrier is passed.
  absl::Status Convert(
      const mediapipe::NormalizedLandmarkList& screen_landmark_list,  //
      const PerspectiveCameraFrustum& pcf,                            //
      mediapipe::LandmarkList& metric_landmark_list,                  //
      Eigen::Matrix4f& pose_transform_mat) const {
    RET_CHECK_EQ(screen_landmark_list.landmark_size(),
                 canonical_metric_landmarks_.cols())
        << "The number of landmarks doesn't match the number passed upon "
           "initialization!";

    Eigen::Matrix3Xf screen_landmarks;
    ConvertLandmarkListToEigenMatrix(screen_landmark_list, screen_landmarks);

    ProjectXY(pcf, screen_landmarks);
    const float depth_offset = screen_landmarks.row(2).mean();

    // 1st iteration: don't unproject XY because it's unsafe to do so due to
    //                the relative nature of the Z coordinate. Instead, run the
    //                first estimation on the projected XY and use that scale to
    //                unproject for the 2nd iteration.
    Eigen::Matrix3Xf intermediate_landmarks(screen_landmarks);
    ChangeHandedness(intermediate_landmarks);

    MP_ASSIGN_OR_RETURN(const float first_iteration_scale,
                        EstimateScale(intermediate_landmarks),
                        _ << "Failed to estimate first iteration scale!");

    // 2nd iteration: unproject XY using the scale from the 1st iteration.
    intermediate_landmarks = screen_landmarks;
    MoveAndRescaleZ(pcf, depth_offset, first_iteration_scale,
                    intermediate_landmarks);
    UnprojectXY(pcf, intermediate_landmarks);
    ChangeHandedness(intermediate_landmarks);

    // For face detection input landmarks, re-write Z-coord from the canonical
    // landmarks.
    if (input_source_ == proto::InputSource::FACE_DETECTION_PIPELINE) {
      Eigen::Matrix4f intermediate_pose_transform_mat;
      MP_RETURN_IF_ERROR(procrustes_solver_->SolveWeightedOrthogonalProblem(
          canonical_metric_landmarks_, intermediate_landmarks,
          landmark_weights_, intermediate_pose_transform_mat))
          << "Failed to estimate pose transform matrix!";

      intermediate_landmarks.row(2) =
          (intermediate_pose_transform_mat *
           canonical_metric_landmarks_.colwise().homogeneous())
              .row(2);
    }
    MP_ASSIGN_OR_RETURN(const float second_iteration_scale,
                        EstimateScale(intermediate_landmarks),
                        _ << "Failed to estimate second iteration scale!");

    // Use the total scale to unproject the screen landmarks.
    const float total_scale = first_iteration_scale * second_iteration_scale;
    MoveAndRescaleZ(pcf, depth_offset, total_scale, screen_landmarks);
    UnprojectXY(pcf, screen_landmarks);
    ChangeHandedness(screen_landmarks);

    // At this point, screen landmarks are converted into metric landmarks.
    Eigen::Matrix3Xf& metric_landmarks = screen_landmarks;

    MP_RETURN_IF_ERROR(procrustes_solver_->SolveWeightedOrthogonalProblem(
        canonical_metric_landmarks_, metric_landmarks, landmark_weights_,
        pose_transform_mat))
        << "Failed to estimate pose transform matrix!";

    // For face detection input landmarks, re-write Z-coord from the canonical
    // landmarks and run the pose transform estimation again.
    if (input_source_ == proto::InputSource::FACE_DETECTION_PIPELINE) {
      metric_landmarks.row(2) =
          (pose_transform_mat *
           canonical_metric_landmarks_.colwise().homogeneous())
              .row(2);

      MP_RETURN_IF_ERROR(procrustes_solver_->SolveWeightedOrthogonalProblem(
          canonical_metric_landmarks_, metric_landmarks, landmark_weights_,
          pose_transform_mat))
          << "Failed to estimate pose transform matrix!";
    }

    // Multiply each of the metric landmarks by the inverse pose
    // transformation matrix to align the runtime metric face landmarks with
    // the canonical metric face landmarks.
    metric_landmarks = (pose_transform_mat.inverse() *
                        metric_landmarks.colwise().homogeneous())
                           .topRows(3);

    ConvertEigenMatrixToLandmarkList(metric_landmarks, metric_landmark_list);

    return absl::OkStatus();
  }

 private:
  void ProjectXY(const PerspectiveCameraFrustum& pcf,
                 Eigen::Matrix3Xf& landmarks) const {
    float x_scale = pcf.right - pcf.left;
    float y_scale = pcf.top - pcf.bottom;
    float x_translation = pcf.left;
    float y_translation = pcf.bottom;

    if (origin_point_location_ == proto::OriginPointLocation::TOP_LEFT_CORNER) {
      landmarks.row(1) = 1.f - landmarks.row(1).array();
    }

    landmarks =
        landmarks.array().colwise() * Eigen::Array3f(x_scale, y_scale, x_scale);
    landmarks.colwise() += Eigen::Vector3f(x_translation, y_translation, 0.f);
  }

  absl::StatusOr<float> EstimateScale(Eigen::Matrix3Xf& landmarks) const {
    Eigen::Matrix4f transform_mat;
    MP_RETURN_IF_ERROR(procrustes_solver_->SolveWeightedOrthogonalProblem(
        canonical_metric_landmarks_, landmarks, landmark_weights_,
        transform_mat))
        << "Failed to estimate canonical-to-runtime landmark set transform!";

    return transform_mat.col(0).norm();
  }

  static void MoveAndRescaleZ(const PerspectiveCameraFrustum& pcf,
                              float depth_offset, float scale,
                              Eigen::Matrix3Xf& landmarks) {
    landmarks.row(2) =
        (landmarks.array().row(2) - depth_offset + pcf.near) / scale;
  }

  static void UnprojectXY(const PerspectiveCameraFrustum& pcf,
                          Eigen::Matrix3Xf& landmarks) {
    landmarks.row(0) =
        landmarks.row(0).cwiseProduct(landmarks.row(2)) / pcf.near;
    landmarks.row(1) =
        landmarks.row(1).cwiseProduct(landmarks.row(2)) / pcf.near;
  }

  static void ChangeHandedness(Eigen::Matrix3Xf& landmarks) {
    landmarks.row(2) *= -1.f;
  }

  static void ConvertLandmarkListToEigenMatrix(
      const mediapipe::NormalizedLandmarkList& landmark_list,
      Eigen::Matrix3Xf& eigen_matrix) {
    eigen_matrix = Eigen::Matrix3Xf(3, landmark_list.landmark_size());
    for (int i = 0; i < landmark_list.landmark_size(); ++i) {
      const auto& landmark = landmark_list.landmark(i);
      eigen_matrix(0, i) = landmark.x();
      eigen_matrix(1, i) = landmark.y();
      eigen_matrix(2, i) = landmark.z();
    }
  }

  static void ConvertEigenMatrixToLandmarkList(
      const Eigen::Matrix3Xf& eigen_matrix,
      mediapipe::LandmarkList& landmark_list) {
    landmark_list.Clear();

    for (int i = 0; i < eigen_matrix.cols(); ++i) {
      auto& landmark = *landmark_list.add_landmark();
      landmark.set_x(eigen_matrix(0, i));
      landmark.set_y(eigen_matrix(1, i));
      landmark.set_z(eigen_matrix(2, i));
    }
  }

  const proto::OriginPointLocation origin_point_location_;
  const proto::InputSource input_source_;
  Eigen::Matrix3Xf canonical_metric_landmarks_;
  Eigen::VectorXf landmark_weights_;

  std::unique_ptr<ProcrustesSolver> procrustes_solver_;
};

class GeometryPipelineImpl : public GeometryPipeline {
 public:
  GeometryPipelineImpl(
      const proto::PerspectiveCamera& perspective_camera,  //
      const proto::Mesh3d& canonical_mesh,                 //
      uint32_t canonical_mesh_vertex_size,                 //
      uint32_t canonical_mesh_num_vertices,
      uint32_t canonical_mesh_vertex_position_offset,
      std::unique_ptr<ScreenToMetricSpaceConverter> space_converter)
      : perspective_camera_(perspective_camera),
        canonical_mesh_(canonical_mesh),
        canonical_mesh_vertex_size_(canonical_mesh_vertex_size),
        canonical_mesh_num_vertices_(canonical_mesh_num_vertices),
        canonical_mesh_vertex_position_offset_(
            canonical_mesh_vertex_position_offset),
        space_converter_(std::move(space_converter)) {}

  absl::StatusOr<std::vector<proto::FaceGeometry>> EstimateFaceGeometry(
      const std::vector<mediapipe::NormalizedLandmarkList>&
          multi_face_landmarks,
      int frame_width, int frame_height) const override {
    MP_RETURN_IF_ERROR(ValidateFrameDimensions(frame_width, frame_height))
        << "Invalid frame dimensions!";

    // Create a perspective camera frustum to be shared for geometry estimation
    // per each face.
    PerspectiveCameraFrustum pcf(perspective_camera_, frame_width,
                                 frame_height);

    std::vector<proto::FaceGeometry> multi_face_geometry;

    // From this point, the meaning of "face landmarks" is clarified further as
    // "screen face landmarks". This is done do distinguish from "metric face
    // landmarks" that are derived during the face geometry estimation process.
    for (const mediapipe::NormalizedLandmarkList& screen_face_landmarks :
         multi_face_landmarks) {
      // Having a too compact screen landmark list will result in numerical
      // instabilities, therefore such faces are filtered.
      if (IsScreenLandmarkListTooCompact(screen_face_landmarks)) {
        continue;
      }

      // Convert the screen landmarks into the metric landmarks and get the pose
      // transformation matrix.
      mediapipe::LandmarkList metric_face_landmarks;
      Eigen::Matrix4f pose_transform_mat;
      MP_RETURN_IF_ERROR(space_converter_->Convert(screen_face_landmarks, pcf,
                                                   metric_face_landmarks,
                                                   pose_transform_mat))
          << "Failed to convert landmarks from the screen to the metric space!";

      // Pack geometry data for this face.
      proto::FaceGeometry face_geometry;
      proto::Mesh3d* mutable_mesh = face_geometry.mutable_mesh();
      // Copy the canonical face mesh as the face geometry mesh.
      mutable_mesh->CopyFrom(canonical_mesh_);
      // Replace XYZ vertex mesh coordinates with the metric landmark positions.
      for (int i = 0; i < canonical_mesh_num_vertices_; ++i) {
        uint32_t vertex_buffer_offset = canonical_mesh_vertex_size_ * i +
                                        canonical_mesh_vertex_position_offset_;

        mutable_mesh->set_vertex_buffer(vertex_buffer_offset,
                                        metric_face_landmarks.landmark(i).x());
        mutable_mesh->set_vertex_buffer(vertex_buffer_offset + 1,
                                        metric_face_landmarks.landmark(i).y());
        mutable_mesh->set_vertex_buffer(vertex_buffer_offset + 2,
                                        metric_face_landmarks.landmark(i).z());
      }
      // Populate the face pose transformation matrix.
      mediapipe::MatrixDataProtoFromMatrix(
          pose_transform_mat, face_geometry.mutable_pose_transform_matrix());

      multi_face_geometry.push_back(face_geometry);
    }

    return multi_face_geometry;
  }

 private:
  static bool IsScreenLandmarkListTooCompact(
      const mediapipe::NormalizedLandmarkList& screen_landmarks) {
    float mean_x = 0.f;
    float mean_y = 0.f;
    for (int i = 0; i < screen_landmarks.landmark_size(); ++i) {
      const auto& landmark = screen_landmarks.landmark(i);
      mean_x += (landmark.x() - mean_x) / static_cast<float>(i + 1);
      mean_y += (landmark.y() - mean_y) / static_cast<float>(i + 1);
    }

    float max_sq_dist = 0.f;
    for (const auto& landmark : screen_landmarks.landmark()) {
      const float d_x = landmark.x() - mean_x;
      const float d_y = landmark.y() - mean_y;
      max_sq_dist = std::max(max_sq_dist, d_x * d_x + d_y * d_y);
    }

    static constexpr float kIsScreenLandmarkListTooCompactThreshold = 1e-3f;
    return std::sqrt(max_sq_dist) <= kIsScreenLandmarkListTooCompactThreshold;
  }

  const proto::PerspectiveCamera perspective_camera_;
  const proto::Mesh3d canonical_mesh_;
  const uint32_t canonical_mesh_vertex_size_;
  const uint32_t canonical_mesh_num_vertices_;
  const uint32_t canonical_mesh_vertex_position_offset_;

  std::unique_ptr<ScreenToMetricSpaceConverter> space_converter_;
};

}  // namespace

absl::StatusOr<std::unique_ptr<GeometryPipeline>> CreateGeometryPipeline(
    const proto::Environment& environment,
    const proto::GeometryPipelineMetadata& metadata) {
  MP_RETURN_IF_ERROR(ValidateEnvironment(environment))
      << "Invalid environment!";
  MP_RETURN_IF_ERROR(ValidateGeometryPipelineMetadata(metadata))
      << "Invalid geometry pipeline metadata!";

  const auto& canonical_mesh = metadata.canonical_mesh();
  RET_CHECK(HasVertexComponent(canonical_mesh.vertex_type(),
                               VertexComponent::POSITION))
      << "Canonical face mesh must have the `POSITION` vertex component!";
  RET_CHECK(HasVertexComponent(canonical_mesh.vertex_type(),
                               VertexComponent::TEX_COORD))
      << "Canonical face mesh must have the `TEX_COORD` vertex component!";

  uint32_t canonical_mesh_vertex_size =
      GetVertexSize(canonical_mesh.vertex_type());
  uint32_t canonical_mesh_num_vertices =
      canonical_mesh.vertex_buffer_size() / canonical_mesh_vertex_size;
  uint32_t canonical_mesh_vertex_position_offset =
      GetVertexComponentOffset(canonical_mesh.vertex_type(),
                               VertexComponent::POSITION)
          .value();

  // Put the Procrustes landmark basis into Eigen matrices for an easier access.
  Eigen::Matrix3Xf canonical_metric_landmarks =
      Eigen::Matrix3Xf::Zero(3, canonical_mesh_num_vertices);
  Eigen::VectorXf landmark_weights =
      Eigen::VectorXf::Zero(canonical_mesh_num_vertices);

  for (int i = 0; i < canonical_mesh_num_vertices; ++i) {
    uint32_t vertex_buffer_offset =
        canonical_mesh_vertex_size * i + canonical_mesh_vertex_position_offset;

    canonical_metric_landmarks(0, i) =
        canonical_mesh.vertex_buffer(vertex_buffer_offset);
    canonical_metric_landmarks(1, i) =
        canonical_mesh.vertex_buffer(vertex_buffer_offset + 1);
    canonical_metric_landmarks(2, i) =
        canonical_mesh.vertex_buffer(vertex_buffer_offset + 2);
  }

  for (const proto::WeightedLandmarkRef& wlr :
       metadata.procrustes_landmark_basis()) {
    uint32_t landmark_id = wlr.landmark_id();
    landmark_weights(landmark_id) = wlr.weight();
  }

  std::unique_ptr<GeometryPipeline> result =
      absl::make_unique<GeometryPipelineImpl>(
          environment.perspective_camera(), canonical_mesh,
          canonical_mesh_vertex_size, canonical_mesh_num_vertices,
          canonical_mesh_vertex_position_offset,
          absl::make_unique<ScreenToMetricSpaceConverter>(
              environment.origin_point_location(),
              metadata.input_source() == proto::InputSource::DEFAULT
                  ? proto::InputSource::FACE_LANDMARK_PIPELINE
                  : metadata.input_source(),
              std::move(canonical_metric_landmarks),
              std::move(landmark_weights),
              CreateFloatPrecisionProcrustesSolver()));

  return result;
}

}  // namespace mediapipe::tasks::vision::face_geometry
