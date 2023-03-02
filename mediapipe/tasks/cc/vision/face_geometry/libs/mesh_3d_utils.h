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

#ifndef MEDIAPIPE_TASKS_CC_VISION_FACE_GEOMETRY_LIBS_MESH_3D_UTILS_H_
#define MEDIAPIPE_TASKS_CC_VISION_FACE_GEOMETRY_LIBS_MESH_3D_UTILS_H_

#include <cstdint>
#include <cstdlib>

#include "mediapipe/framework/port/statusor.h"
#include "mediapipe/tasks/cc/vision/face_geometry/proto/mesh_3d.pb.h"

namespace mediapipe::tasks::vision::face_geometry {

enum class VertexComponent { POSITION, TEX_COORD };

std::size_t GetVertexSize(proto::Mesh3d::VertexType vertex_type);

std::size_t GetPrimitiveSize(proto::Mesh3d::PrimitiveType primitive_type);

bool HasVertexComponent(proto::Mesh3d::VertexType vertex_type,
                        VertexComponent vertex_component);

// Computes the vertex component offset.
//
// Returns an error status if a given vertex type doesn't have the requested
// component.
absl::StatusOr<uint32_t> GetVertexComponentOffset(
    proto::Mesh3d::VertexType vertex_type, VertexComponent vertex_component);

// Computes the vertex component size.
//
// Returns an error status if a given vertex type doesn't have the requested
// component.
absl::StatusOr<uint32_t> GetVertexComponentSize(
    proto::Mesh3d::VertexType vertex_type, VertexComponent vertex_component);

}  // namespace mediapipe::tasks::vision::face_geometry

#endif  // MEDIAPIPE_TASKS_CC_VISION_FACE_GEOMETRY_LIBS_MESH_3D_UTILS_H_
