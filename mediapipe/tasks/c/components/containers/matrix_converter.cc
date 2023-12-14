/* Copyright 2023 The MediaPipe Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "mediapipe/tasks/c/components/containers/matrix_converter.h"

#include <cstdlib>

#include "Eigen/Core"
#include "mediapipe/tasks/c/components/containers/matrix.h"

namespace mediapipe::tasks::c::components::containers {

void CppConvertToMatrix(const Eigen::MatrixXf& in, ::Matrix* out) {
  out->rows = in.rows();
  out->cols = in.cols();
  out->data = new float[out->rows * out->cols];

  // Copies data from an Eigen matrix (default column-major) to a C-style
  // matrix, preserving the sequence of elements as per the Eigen matrix's
  // internal storage (column-major order by default).
  if (!in.IsRowMajor) {
    // Safe to use memcpy when the Eigen matrix is in its default column-major
    // order.
    memcpy(out->data, in.data(), sizeof(float) * out->rows * out->cols);
  }
}

void CppCloseMatrix(::Matrix* in) {
  if (in->data) {
    delete[] in->data;
    in->data = nullptr;
  }
}

}  // namespace mediapipe::tasks::c::components::containers
