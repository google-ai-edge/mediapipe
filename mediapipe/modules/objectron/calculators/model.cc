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

#include "mediapipe/modules/objectron/calculators/model.h"

#include "mediapipe/framework/port/logging.h"

namespace mediapipe {

void Model::SetTransformation(const Eigen::Matrix4f& transform) {
  transformation_ = transform;
}

void Model::SetTranslation(const Eigen::Vector3f& translation) {
  transformation_.col(3).template head<3>() = translation;
}

void Model::SetRotation(float roll, float pitch, float yaw) {
  // In our coordinate system, Y is up. We first rotate the object around Y
  // (yaw), then around Z (pitch), and finally around X (roll).
  Eigen::Matrix3f r;
  r = Eigen::AngleAxisf(yaw, Eigen::Vector3f::UnitY()) *
      Eigen::AngleAxisf(pitch, Eigen::Vector3f::UnitZ()) *
      Eigen::AngleAxisf(roll, Eigen::Vector3f::UnitX());
  transformation_.topLeftCorner<3, 3>() = r;
}

void Model::SetRotation(const Eigen::Matrix3f& rotation) {
  transformation_.topLeftCorner<3, 3>() = rotation;
}

void Model::SetScale(const Eigen::Vector3f& scale) { scale_ = scale; }

void Model::SetCategory(const std::string& category) { category_ = category; }

const Eigen::Vector3f Model::GetRotationAngles() const {
  Vector3f ypr = transformation_.topLeftCorner<3, 3>().eulerAngles(1, 2, 0);
  return Vector3f(ypr(2), ypr(1), ypr(0));  // swap YPR with RPY
}

const Eigen::Matrix4f& Model::GetTransformation() const {
  return transformation_;
}

const Eigen::Vector3f& Model::GetScale() const { return scale_; }

const Eigen::Ref<const Eigen::Vector3f> Model::GetTranslation() const {
  return transformation_.col(3).template head<3>();
}

const Eigen::Ref<const Eigen::Matrix3f> Model::GetRotation() const {
  return transformation_.template topLeftCorner<3, 3>();
}

const std::string& Model::GetCategory() const { return category_; }

void Model::Deserialize(const Object& obj) {
  CHECK_EQ(obj.rotation_size(), 9);
  CHECK_EQ(obj.translation_size(), 3);
  CHECK_EQ(obj.scale_size(), 3);
  category_ = obj.category();

  using RotationMatrix = Eigen::Matrix<float, 3, 3, Eigen::RowMajor>;
  transformation_.setIdentity();
  transformation_.topLeftCorner<3, 3>() =
      Eigen::Map<const RotationMatrix>(obj.rotation().data());
  transformation_.col(3).head<3>() =
      Eigen::Map<const Eigen::Vector3f>(obj.translation().data());
  scale_ = Eigen::Map<const Eigen::Vector3f>(obj.scale().data());
  Update();
}

void Model::Serialize(Object* obj) {
  obj->set_category(category_);
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      obj->add_rotation(transformation_(i, j));
    }
  }

  for (int i = 0; i < 3; ++i) {
    obj->add_translation(transformation_(i, 3));
  }

  for (int i = 0; i < 3; ++i) {
    obj->add_scale(scale_[i]);
  }
}

}  // namespace mediapipe
