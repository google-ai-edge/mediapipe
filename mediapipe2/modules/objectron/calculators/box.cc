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

#include "mediapipe/modules/objectron/calculators/box.h"

#include "Eigen/Core"
#include "mediapipe/framework/port/logging.h"

namespace mediapipe {

namespace {
constexpr int kFrontFaceId = 4;
constexpr int kTopFaceId = 2;
constexpr int kNumKeypoints = 8 + 1;
constexpr int kNumberOfAxis = 3;
constexpr int kEdgesPerAxis = 4;

}  // namespace

Box::Box(const std::string& category)
    : Model(kBoundingBox, kNumKeypoints, category),
      bounding_box_(kNumKeypoints) {
  transformation_.setIdentity();

  scale_ << 0.1, 0.1, 0.1;

  // The vertices are ordered according to the left-hand rule, so the normal
  // vector of each face will point inward the box.
  faces_.push_back({5, 6, 8, 7});  // +x on yz plane
  faces_.push_back({1, 3, 4, 2});  // -x on yz plane

  faces_.push_back({3, 7, 8, 4});  // +y on xz plane = top
  faces_.push_back({1, 2, 6, 5});  // -y on xz plane

  faces_.push_back({2, 4, 8, 6});  // +z on xy plane = front
  faces_.push_back({1, 5, 7, 3});  // -z on xy plane

  // Add the edges in the cube, they are sorted according to axis (x-y-z).
  edges_.push_back({1, 5});
  edges_.push_back({2, 6});
  edges_.push_back({3, 7});
  edges_.push_back({4, 8});

  edges_.push_back({1, 3});
  edges_.push_back({5, 7});
  edges_.push_back({2, 4});
  edges_.push_back({6, 8});

  edges_.push_back({1, 2});
  edges_.push_back({3, 4});
  edges_.push_back({5, 6});
  edges_.push_back({7, 8});
  Update();
}

void Box::Update() {
  // Compute the eight vertices of the bounding box from Box's parameters
  auto w = scale_[0] / 2.f;
  auto h = scale_[1] / 2.f;
  auto d = scale_[2] / 2.f;

  // Define the local coordinate system, w.r.t. the center of the boxs
  bounding_box_[0] << 0., 0., 0.;
  bounding_box_[1] << -w, -h, -d;
  bounding_box_[2] << -w, -h, +d;
  bounding_box_[3] << -w, +h, -d;
  bounding_box_[4] << -w, +h, +d;
  bounding_box_[5] << +w, -h, -d;
  bounding_box_[6] << +w, -h, +d;
  bounding_box_[7] << +w, +h, -d;
  bounding_box_[8] << +w, +h, +d;

  // Convert to world coordinate system
  for (int i = 0; i < kNumKeypoints; ++i) {
    bounding_box_[i] =
        transformation_.topLeftCorner<3, 3>() * bounding_box_[i] +
        transformation_.col(3).head<3>();
  }
}

void Box::Adjust(const std::vector<float>& variables) {
  Eigen::Vector3f translation;
  translation << variables[0], variables[1], variables[2];
  SetTranslation(translation);

  const float roll = variables[3];
  const float pitch = variables[4];
  const float yaw = variables[5];
  SetRotation(roll, pitch, yaw);

  Eigen::Vector3f scale;
  scale << variables[6], variables[7], variables[8];

  SetScale(scale);
  Update();
}

float* Box::GetVertex(size_t vertex_id) {
  CHECK_LT(vertex_id, kNumKeypoints);
  return bounding_box_[vertex_id].data();
}

const float* Box::GetVertex(size_t vertex_id) const {
  CHECK_LT(vertex_id, kNumKeypoints);
  return bounding_box_[vertex_id].data();
}

bool Box::InsideTest(const Eigen::Vector3f& point, int check_axis) const {
  const float* v0 = GetVertex(1);
  const float* v1 = GetVertex(2);
  const float* v2 = GetVertex(3);
  const float* v4 = GetVertex(5);

  switch (check_axis) {
    case 1:
      return (v0[0] <= point[0] && point[0] <= v1[0]);  // X-axis
    case 2:
      return (v0[1] <= point[1] && point[1] <= v2[1]);  // Y-axis
    case 3:
      return (v0[2] <= point[2] && point[2] <= v4[2]);  // Z-axis
    default:
      return false;
  }
}

void Box::Deserialize(const Object& obj) {
  CHECK_EQ(obj.keypoints_size(), kNumKeypoints);
  Model::Deserialize(obj);
}

void Box::Serialize(Object* obj) {
  Model::Serialize(obj);
  obj->set_type(Object::BOUNDING_BOX);
  std::vector<Vector3f> local_bounding_box(9);
  // Define the local coordinate system, w.r.t. the center of the boxs
  local_bounding_box[0] << 0., 0., 0.;
  local_bounding_box[1] << -0.5, -0.5, -0.5;
  local_bounding_box[2] << -0.5, -0.5, +0.5;
  local_bounding_box[3] << -0.5, +0.5, -0.5;
  local_bounding_box[4] << -0.5, +0.5, +0.5;
  local_bounding_box[5] << +0.5, -0.5, -0.5;
  local_bounding_box[6] << +0.5, -0.5, +0.5;
  local_bounding_box[7] << +0.5, +0.5, -0.5;
  local_bounding_box[8] << +0.5, +0.5, +0.5;
  for (int i = 0; i < kNumKeypoints; ++i) {
    KeyPoint* keypoint = obj->add_keypoints();
    keypoint->set_x(local_bounding_box[i][0]);
    keypoint->set_y(local_bounding_box[i][1]);
    keypoint->set_z(local_bounding_box[i][2]);
    keypoint->set_confidence_radius(0.);
  }
}

const Face& Box::GetFrontFace() const { return faces_[kFrontFaceId]; }

const Face& Box::GetTopFace() const { return faces_[kTopFaceId]; }

std::pair<Vector3f, Vector3f> Box::GetGroundPlane() const {
  const Vector3f gravity = Vector3f(0., 1., 0.);
  int ground_plane_id = 0;
  float ground_plane_error = 10.0;

  auto get_face_center = [&](const Face& face) {
    Vector3f center = Vector3f::Zero();
    for (const int vertex_id : face) {
      center += Map<const Vector3f>(GetVertex(vertex_id));
    }
    center /= face.size();
    return center;
  };

  auto get_face_normal = [&](const Face& face, const Vector3f& center) {
    Vector3f v1 = Map<const Vector3f>(GetVertex(face[0])) - center;
    Vector3f v2 = Map<const Vector3f>(GetVertex(face[1])) - center;
    Vector3f normal = v1.cross(v2);
    return normal;
  };

  // The ground plane is defined as a plane aligned with gravity.
  // gravity is the (0, 1, 0) vector in the world coordinate system.
  const auto& faces = GetFaces();
  for (int face_id = 0; face_id < faces.size(); face_id += 2) {
    const auto& face = faces[face_id];
    Vector3f center = get_face_center(face);
    Vector3f normal = get_face_normal(face, center);
    Vector3f w = gravity.cross(normal);
    const float w_sq_norm = w.squaredNorm();
    if (w_sq_norm < ground_plane_error) {
      ground_plane_error = w_sq_norm;
      ground_plane_id = face_id;
    }
  }

  Vector3f center = get_face_center(faces[ground_plane_id]);
  Vector3f normal = get_face_normal(faces[ground_plane_id], center);

  // For each face, we also have a parallel face that it's normal is also
  // aligned with gravity vector. We pick the face with lower height (y-value).
  // The parallel to face 0 is 1, face 2 is 3, and face 4 is 5.
  int parallel_face_id = ground_plane_id + 1;
  const auto& parallel_face = faces[parallel_face_id];
  Vector3f parallel_face_center = get_face_center(parallel_face);
  Vector3f parallel_face_normal =
      get_face_normal(parallel_face, parallel_face_center);
  if (parallel_face_center[1] < center[1]) {
    center = parallel_face_center;
    normal = parallel_face_normal;
  }
  return {center, normal};
}

template <typename T>
void Box::Fit(const std::vector<T>& vertices) {
  CHECK_EQ(vertices.size(), kNumKeypoints);
  scale_.setZero();
  // The scale would remain invariant under rotation and translation.
  // We can safely estimate the scale from the oriented box.
  for (int axis = 0; axis < kNumberOfAxis; ++axis) {
    for (int edge_id = 0; edge_id < kEdgesPerAxis; ++edge_id) {
      // The edges are stored in quadruples according to each axis
      const std::array<int, 2>& edge = edges_[axis * kEdgesPerAxis + edge_id];
      scale_[axis] += (vertices[edge[0]] - vertices[edge[1]]).norm();
    }
    scale_[axis] /= kEdgesPerAxis;
  }
  // Create a scaled axis-aligned box
  transformation_.setIdentity();
  Update();

  using MatrixN3_RM = Eigen::Matrix<float, kNumKeypoints, 3, Eigen::RowMajor>;
  Eigen::Map<const MatrixN3_RM> v(vertices[0].data());
  Eigen::Map<const MatrixN3_RM> system(bounding_box_[0].data());
  auto system_h = system.rowwise().homogeneous().eval();
  auto system_g = system_h.colPivHouseholderQr();
  auto solution = system_g.solve(v).eval();
  transformation_.topLeftCorner<3, 4>() = solution.transpose();
  Update();
}

template void Box::Fit<Vector3f>(const std::vector<Vector3f>&);
template void Box::Fit<Map<Vector3f>>(const std::vector<Map<Vector3f>>&);
template void Box::Fit<Map<const Vector3f>>(
    const std::vector<Map<const Vector3f>>&);
}  // namespace mediapipe
