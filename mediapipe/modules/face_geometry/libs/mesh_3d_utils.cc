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

#include "mediapipe/modules/face_geometry/libs/mesh_3d_utils.h"

#include <cstdint>
#include <cstdlib>

#include "mediapipe/framework/port/ret_check.h"
#include "mediapipe/framework/port/statusor.h"
#include "mediapipe/modules/face_geometry/protos/mesh_3d.pb.h"

namespace mediapipe::face_geometry {
namespace {

bool HasVertexComponentVertexPT(VertexComponent vertex_component) {
  switch (vertex_component) {
    case VertexComponent::POSITION:
    case VertexComponent::TEX_COORD:
      return true;

    default:
      return false;
  }
}

uint32_t GetVertexComponentSizeVertexPT(VertexComponent vertex_component) {
  switch (vertex_component) {
    case VertexComponent::POSITION:
      return 3;
    case VertexComponent::TEX_COORD:
      return 2;
  }
}

uint32_t GetVertexComponentOffsetVertexPT(VertexComponent vertex_component) {
  switch (vertex_component) {
    case VertexComponent::POSITION:
      return 0;
    case VertexComponent::TEX_COORD:
      return GetVertexComponentSizeVertexPT(VertexComponent::POSITION);
  }
}

}  // namespace

std::size_t GetVertexSize(Mesh3d::VertexType vertex_type) {
  switch (vertex_type) {
    case Mesh3d::VERTEX_PT:
      return GetVertexComponentSizeVertexPT(VertexComponent::POSITION) +
             GetVertexComponentSizeVertexPT(VertexComponent::TEX_COORD);
  }
}

std::size_t GetPrimitiveSize(Mesh3d::PrimitiveType primitive_type) {
  switch (primitive_type) {
    case Mesh3d::TRIANGLE:
      return 3;
  }
}

bool HasVertexComponent(Mesh3d::VertexType vertex_type,
                        VertexComponent vertex_component) {
  switch (vertex_type) {
    case Mesh3d::VERTEX_PT:
      return HasVertexComponentVertexPT(vertex_component);
  }
}

absl::StatusOr<uint32_t> GetVertexComponentOffset(
    Mesh3d::VertexType vertex_type, VertexComponent vertex_component) {
  RET_CHECK(HasVertexComponentVertexPT(vertex_component))
      << "A given vertex type doesn't have the requested component!";

  switch (vertex_type) {
    case Mesh3d::VERTEX_PT:
      return GetVertexComponentOffsetVertexPT(vertex_component);
  }
}

absl::StatusOr<uint32_t> GetVertexComponentSize(
    Mesh3d::VertexType vertex_type, VertexComponent vertex_component) {
  RET_CHECK(HasVertexComponentVertexPT(vertex_component))
      << "A given vertex type doesn't have the requested component!";

  switch (vertex_type) {
    case Mesh3d::VERTEX_PT:
      return GetVertexComponentSizeVertexPT(vertex_component);
  }
}

}  // namespace mediapipe::face_geometry
