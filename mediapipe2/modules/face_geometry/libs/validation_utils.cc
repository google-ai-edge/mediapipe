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

#include "mediapipe/modules/face_geometry/libs/validation_utils.h"

#include <cstdint>
#include <cstdlib>

#include "mediapipe/framework/formats/matrix_data.pb.h"
#include "mediapipe/framework/port/ret_check.h"
#include "mediapipe/framework/port/status.h"
#include "mediapipe/framework/port/status_macros.h"
#include "mediapipe/modules/face_geometry/libs/mesh_3d_utils.h"
#include "mediapipe/modules/face_geometry/protos/environment.pb.h"
#include "mediapipe/modules/face_geometry/protos/geometry_pipeline_metadata.pb.h"
#include "mediapipe/modules/face_geometry/protos/mesh_3d.pb.h"

namespace mediapipe::face_geometry {

absl::Status ValidatePerspectiveCamera(
    const PerspectiveCamera& perspective_camera) {
  static constexpr float kAbsoluteErrorEps = 1e-9f;

  RET_CHECK_GT(perspective_camera.near(), kAbsoluteErrorEps)
      << "Near Z must be greater than 0 with a margin of 10^{-9}!";

  RET_CHECK_GT(perspective_camera.far(),
               perspective_camera.near() + kAbsoluteErrorEps)
      << "Far Z must be greater than Near Z with a margin of 10^{-9}!";

  RET_CHECK_GT(perspective_camera.vertical_fov_degrees(), kAbsoluteErrorEps)
      << "Vertical FOV must be positive with a margin of 10^{-9}!";

  RET_CHECK_LT(perspective_camera.vertical_fov_degrees() + kAbsoluteErrorEps,
               180.f)
      << "Vertical FOV must be less than 180 degrees with a margin of 10^{-9}";

  return absl::OkStatus();
}

absl::Status ValidateEnvironment(const Environment& environment) {
  MP_RETURN_IF_ERROR(
      ValidatePerspectiveCamera(environment.perspective_camera()))
      << "Invalid perspective camera!";

  return absl::OkStatus();
}

absl::Status ValidateMesh3d(const Mesh3d& mesh_3d) {
  const std::size_t vertex_size = GetVertexSize(mesh_3d.vertex_type());
  const std::size_t primitive_type = GetPrimitiveSize(mesh_3d.primitive_type());

  RET_CHECK_EQ(mesh_3d.vertex_buffer_size() % vertex_size, 0)
      << "Vertex buffer size must a multiple of the vertex size!";

  RET_CHECK_EQ(mesh_3d.index_buffer_size() % primitive_type, 0)
      << "Index buffer size must a multiple of the primitive size!";

  const int num_vertices = mesh_3d.vertex_buffer_size() / vertex_size;
  for (uint32_t idx : mesh_3d.index_buffer()) {
    RET_CHECK_LT(idx, num_vertices)
        << "All mesh indices must refer to an existing vertex!";
  }

  return absl::OkStatus();
}

absl::Status ValidateFaceGeometry(const FaceGeometry& face_geometry) {
  MP_RETURN_IF_ERROR(ValidateMesh3d(face_geometry.mesh())) << "Invalid mesh!";

  static constexpr char kInvalid4x4MatrixMessage[] =
      "Pose transformation matrix must be a 4x4 matrix!";

  const MatrixData& pose_transform_matrix =
      face_geometry.pose_transform_matrix();
  RET_CHECK_EQ(pose_transform_matrix.rows(), 4) << kInvalid4x4MatrixMessage;
  RET_CHECK_EQ(pose_transform_matrix.rows(), 4) << kInvalid4x4MatrixMessage;
  RET_CHECK_EQ(pose_transform_matrix.packed_data_size(), 16)
      << kInvalid4x4MatrixMessage;

  return absl::OkStatus();
}

absl::Status ValidateGeometryPipelineMetadata(
    const GeometryPipelineMetadata& metadata) {
  MP_RETURN_IF_ERROR(ValidateMesh3d(metadata.canonical_mesh()))
      << "Invalid canonical mesh!";

  RET_CHECK_GT(metadata.procrustes_landmark_basis_size(), 0)

      << "Procrustes landmark basis must be non-empty!";

  const int num_vertices =
      metadata.canonical_mesh().vertex_buffer_size() /
      GetVertexSize(metadata.canonical_mesh().vertex_type());
  for (const WeightedLandmarkRef& wlr : metadata.procrustes_landmark_basis()) {
    RET_CHECK_LT(wlr.landmark_id(), num_vertices)
        << "All Procrustes basis indices must refer to an existing canonical "
           "mesh vertex!";

    RET_CHECK_GE(wlr.weight(), 0.f)
        << "All Procrustes basis landmarks must have a non-negative weight!";
  }

  return absl::OkStatus();
}

absl::Status ValidateFrameDimensions(int frame_width, int frame_height) {
  RET_CHECK_GT(frame_width, 0) << "Frame width must be positive!";
  RET_CHECK_GT(frame_height, 0) << "Frame height must be positive!";

  return absl::OkStatus();
}

}  // namespace mediapipe::face_geometry
