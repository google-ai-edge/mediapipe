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

#include "absl/log/absl_check.h"
#include "mediapipe/calculators/tensorflow/matrix_to_tensor_calculator_options.pb.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/formats/matrix.h"
#include "mediapipe/framework/formats/time_series_header.pb.h"
#include "mediapipe/framework/port/ret_check.h"
#include "mediapipe/framework/port/status.h"
#include "mediapipe/framework/port/status_macros.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.h"

namespace mediapipe {

namespace {
absl::Status FillTimeSeriesHeaderIfValid(const Packet& header_packet,
                                         TimeSeriesHeader* header) {
  ABSL_CHECK(header);
  if (header_packet.IsEmpty()) {
    return absl::UnknownError("No header found.");
  }
  if (!header_packet.ValidateAsType<TimeSeriesHeader>().ok()) {
    return absl::UnknownError("Packet does not contain TimeSeriesHeader.");
  }
  *header = header_packet.Get<TimeSeriesHeader>();
  if (header->has_sample_rate() && header->sample_rate() >= 0 &&
      header->has_num_channels() && header->num_channels() >= 0) {
    return absl::OkStatus();
  } else {
    std::string error_message =
        "TimeSeriesHeader is missing necessary fields: "
        "sample_rate or num_channels, or one of their values is negative. ";
#ifndef MEDIAPIPE_MOBILE
    absl::StrAppend(&error_message, "Got header:\n",
                    header->ShortDebugString());
#endif
    return absl::InvalidArgumentError(error_message);
  }
}
}  // namespace

namespace tf = tensorflow;

typedef Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
    RowMajorMatrixXf;
typedef Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>
    ColMajorMatrixXf;

// Converts an input Matrix into a 2D or 3D tf::Tensor.
//
// The calculator expects one input (a packet containing a Matrix) and
// generates one output (a packet containing a tf::Tensor containing the same
// data). The output tensor will be 2D with dimensions corresponding to the
// input matrix, while it will be 3D if add_trailing_dimension is set to true.
// The option for making the tensor be 3D is useful for using audio and image
// features for training multimodal models, so that the number of tensor
// dimensions match up. It will hold DT_FLOAT values.
//
// Example config:
// node {
//   calculator: "MatrixToTensorCalculator"
//   input_stream: "matrix_features"
//   output_stream: "tensor_features"
// }
class MatrixToTensorCalculator : public CalculatorBase {
 public:
  static absl::Status GetContract(CalculatorContract* cc);

  absl::Status Open(CalculatorContext* cc) override;
  absl::Status Process(CalculatorContext* cc) override;

 private:
  MatrixToTensorCalculatorOptions options_;
};
REGISTER_CALCULATOR(MatrixToTensorCalculator);

absl::Status MatrixToTensorCalculator::GetContract(CalculatorContract* cc) {
  RET_CHECK_EQ(cc->Inputs().NumEntries(), 1)
      << "Only one input stream is supported.";
  cc->Inputs().Index(0).Set<Matrix>(
      // Input Matrix stream with optional TimeSeriesHeader.
  );
  RET_CHECK_EQ(cc->Outputs().NumEntries(), 1)
      << "Only one output stream is supported.";
  cc->Outputs().Index(0).Set<tf::Tensor>(
      // Output stream with data as tf::Tensor and the same
      // TimeSeriesHeader as the input (or no header if the input has no
      // header).
  );
  return absl::OkStatus();
}

absl::Status MatrixToTensorCalculator::Open(CalculatorContext* cc) {
  // If the input is part of a time series, then preserve the header so that
  // downstream consumers can access the sample rate if needed.
  options_ = cc->Options<MatrixToTensorCalculatorOptions>();
  auto input_header = ::absl::make_unique<TimeSeriesHeader>();
  const absl::Status header_status = FillTimeSeriesHeaderIfValid(
      cc->Inputs().Index(0).Header(), input_header.get());
  if (header_status.ok()) {
    cc->Outputs().Index(0).SetHeader(Adopt(input_header.release()));
  }

  // Inform the framework that we always output at the same timestamp
  // as we receive a packet at.
  cc->SetOffset(TimestampDiff(0));
  return absl::OkStatus();
}

absl::Status MatrixToTensorCalculator::Process(CalculatorContext* cc) {
  const Matrix& matrix = cc->Inputs().Index(0).Get<Matrix>();
  tf::TensorShape tensor_shape;
  if (options_.transpose()) {
    tensor_shape = tf::TensorShape({matrix.cols(), matrix.rows()});
  } else {
    tensor_shape = tf::TensorShape({matrix.rows(), matrix.cols()});
  }
  auto tensor = ::absl::make_unique<tf::Tensor>(tf::DT_FLOAT, tensor_shape);

  float* tensor_data = tensor->flat<float>().data();
  if (options_.transpose()) {
    auto matrix_map =
        Eigen::Map<ColMajorMatrixXf>(tensor_data, matrix.rows(), matrix.cols());
    matrix_map = matrix;
  } else {
    auto matrix_map =
        Eigen::Map<RowMajorMatrixXf>(tensor_data, matrix.rows(), matrix.cols());
    matrix_map = matrix;
  }

  if (options_.add_trailing_dimension()) {
    tf::TensorShape new_shape(tensor_shape);
    new_shape.AddDim(1 /* size of dimension */);
    RET_CHECK(tensor->CopyFrom(*tensor, new_shape))
        << "Could not add dimension to tensor without changing its shape."
        << " Current shape: " << tensor->shape().DebugString();
  }
  cc->Outputs().Index(0).Add(tensor.release(), cc->InputTimestamp());
  return absl::OkStatus();
}

}  // namespace mediapipe
