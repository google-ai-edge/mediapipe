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
#include "mediapipe/framework/formats/matrix.h"

#include <algorithm>

#include "absl/log/absl_check.h"
#include "mediapipe/framework/port/core_proto_inc.h"
#include "mediapipe/framework/port/logging.h"
#include "mediapipe/framework/port/proto_ns.h"
#include "mediapipe/framework/port/ret_check.h"

namespace mediapipe {

void MatrixDataProtoFromMatrix(const Matrix& matrix, MatrixData* matrix_data) {
  const int rows = matrix.rows();
  const int cols = matrix.cols();
  matrix_data->set_rows(rows);
  matrix_data->set_cols(cols);
  matrix_data->clear_layout();
  proto_ns::RepeatedField<float>(matrix.data(), matrix.data() + rows * cols)
      .Swap(matrix_data->mutable_packed_data());
}

void MatrixFromMatrixDataProto(const MatrixData& matrix_data, Matrix* matrix) {
  ABSL_CHECK_EQ(matrix_data.rows() * matrix_data.cols(),
                matrix_data.packed_data_size());
  if (matrix_data.layout() == MatrixData::ROW_MAJOR) {
    matrix->resize(matrix_data.cols(), matrix_data.rows());
  } else {
    matrix->resize(matrix_data.rows(), matrix_data.cols());
  }

  std::copy(matrix_data.packed_data().begin(), matrix_data.packed_data().end(),
            matrix->data());
  if (matrix_data.layout() == MatrixData::ROW_MAJOR) {
    matrix->transposeInPlace();
  }
}

#if !defined(MEDIAPIPE_MOBILE) && !defined(MEDIAPIPE_LITE)
std::string MatrixAsTextProto(const Matrix& matrix) {
  MatrixData matrix_data;
  MatrixDataProtoFromMatrix(matrix, &matrix_data);
  std::string text_proto;
  proto_ns::TextFormat::PrintToString(matrix_data, &text_proto);
  return text_proto;
}

void MatrixFromTextProto(const std::string& text_proto, Matrix* matrix) {
  ABSL_CHECK(matrix);
  MatrixData matrix_data;
  ABSL_CHECK(proto_ns::TextFormat::ParseFromString(text_proto, &matrix_data));
  MatrixFromMatrixDataProto(matrix_data, matrix);
}
#endif  // !defined(MEDIAPIPE_MOBILE) && !defined(MEDIAPIPE_LITE)
}  // namespace mediapipe
