// Copyright 2018 The MediaPipe Authors.
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
//
// Calculator converts from one-dimensional Tensor of DT_FLOAT to Matrix
// OR from (batched) two-dimensional Tensor of DT_FLOAT to Matrix.

#include "absl/log/absl_check.h"
#include "mediapipe/calculators/tensorflow/tensor_to_matrix_calculator.pb.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/formats/matrix.h"
#include "mediapipe/framework/formats/time_series_header.pb.h"
#include "mediapipe/framework/port/ret_check.h"
#include "mediapipe/framework/port/status.h"
#include "mediapipe/framework/port/status_macros.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/types.h"

namespace mediapipe {

namespace tf = ::tensorflow;
namespace {

constexpr char kMatrix[] = "MATRIX";
constexpr char kTensor[] = "TENSOR";
constexpr char kReference[] = "REFERENCE";

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

// Converts a 1-D or a 2-D Tensor into a 2 dimensional Matrix.
// Input:
// -- 1-D or 2-D Tensor
// Output:
// -- Matrix with the same values as the Tensor
// If input tensor is 1 dimensional, the output Matrix is of (nx1) shape.
// It is a 1-D column vector, with n rows and 1 column.
// If input tensor is 2 dimensional (mxn), the output Matrix is (nxm) shape.
// It has n rows and m columns.
//
// Example Config
// node: {
//   calculator: "TensorToMatrixCalculator"
//   input_stream: "TENSOR:tensor"
//   output_stream: "MATRIX:matrix"
// }
//
//
// This calculator produces a TimeSeriesHeader header on its output stream iff
// an input stream is supplied with the REFERENCE tag and that stream has a
// header of type TimeSeriesHeader. This header is modified in two ways:
//   - the sample_rate is set to the packet rate of the REFERENCE stream (which
//    must have a packet_rate defined in its header). This is under the
//    assumption that the packets on the reference stream, input stream, and
//    output stream are in a 1:1 correspondence, and that the output packets are
//    1-D column vectors that represent a single sample of output.
//  - the TimeSeriesHeader overrides specified in the calculator options are
//    then applied, which can override the sample_rate field.
// If the REFERENCE stream is supplied, then the TimeSeriesHeader is verified on
// the input data when it arrives in Process(). In particular, if the header
// states that we produce a 1xD column vector, the input tensor must also be 1xD
//
// Example Config
// node: {
//   calculator: "TensorToMatrixCalculator"
//   input_stream: "TENSOR:tensor"
//   input_stream: "REFERENCE:reference_matrix"
//   output_stream: "MATRIX:matrix"
//   options {
//     [mediapipe.TensorToMatrixCalculatorOptions.ext] {
//       time_series_header_overrides {
//         num_channels: 128
//       }
//     }
//   }
// }
class TensorToMatrixCalculator : public CalculatorBase {
 public:
  static absl::Status GetContract(CalculatorContract* cc);

  absl::Status Open(CalculatorContext* cc) override;
  absl::Status Process(CalculatorContext* cc) override;

  // Store header information so that we can verify the inputs in process().
  TimeSeriesHeader header_;
};
REGISTER_CALCULATOR(TensorToMatrixCalculator);

absl::Status TensorToMatrixCalculator::GetContract(CalculatorContract* cc) {
  RET_CHECK_LE(cc->Inputs().NumEntries(), 2)
      << "Only one or two input streams are supported.";
  RET_CHECK_GT(cc->Inputs().NumEntries(), 0)
      << "At least one input stream must be provided.";
  RET_CHECK(cc->Inputs().HasTag(kTensor))
      << "An input stream for tag: " << kTensor << " must be provided.";
  cc->Inputs().Tag(kTensor).Set<tf::Tensor>(
      // Input Tensor.
  );
  if (cc->Inputs().NumEntries() == 2) {
    RET_CHECK(cc->Inputs().HasTag(kReference))
        << "An input stream for tag: " << kReference
        << " must be provided when"
           " providing two inputs.";
    cc->Inputs()
        .Tag(kReference)
        .Set<Matrix>(
            // A reference stream for the header.
        );
  }
  RET_CHECK_EQ(cc->Outputs().NumEntries(), 1)
      << "Only one output stream is supported.";
  cc->Outputs().Tag(kMatrix).Set<Matrix>(
      // Output Matrix.
  );
  return absl::OkStatus();
}

absl::Status TensorToMatrixCalculator::Open(CalculatorContext* cc) {
  auto input_header = absl::make_unique<TimeSeriesHeader>();
  absl::Status header_status;
  if (cc->Inputs().HasTag(kReference)) {
    header_status = FillTimeSeriesHeaderIfValid(
        cc->Inputs().Tag(kReference).Header(), input_header.get());
  }
  if (header_status.ok()) {
    if (cc->Options<TensorToMatrixCalculatorOptions>()
            .has_time_series_header_overrides()) {
      // This only supports a single sample per packet for now, so we hardcode
      // the sample_rate based on the packet_rate of the REFERENCE and fail
      // if we cannot.
      const TimeSeriesHeader& override_header =
          cc->Options<TensorToMatrixCalculatorOptions>()
              .time_series_header_overrides();
      input_header->MergeFrom(override_header);
      RET_CHECK(input_header->has_packet_rate())
          << "The TimeSeriesHeader.packet_rate must be set.";
      if (!override_header.has_sample_rate()) {
        RET_CHECK_EQ(input_header->num_samples(), 1)
            << "Currently the time series can only output single samples.";
        input_header->set_sample_rate(input_header->packet_rate());
      }
    }
    header_ = *input_header;
    cc->Outputs().Tag(kMatrix).SetHeader(Adopt(input_header.release()));
  }
  cc->SetOffset(TimestampDiff(0));
  return absl::OkStatus();
}

absl::Status TensorToMatrixCalculator::Process(CalculatorContext* cc) {
  // Verify that each reference stream packet corresponds to a tensor packet
  // otherwise the header information is invalid. If we don't have a reference
  // stream, Process() is only called when we have an input tensor and this is
  // always True.
  RET_CHECK(cc->Inputs().HasTag(kTensor))
      << "Tensor stream not available at same timestamp as the reference "
         "stream.";
  RET_CHECK(!cc->Inputs().Tag(kTensor).IsEmpty()) << "Tensor stream is empty.";
  RET_CHECK_OK(cc->Inputs().Tag(kTensor).Value().ValidateAsType<tf::Tensor>())
      << "Tensor stream packet does not contain a Tensor.";

  const tf::Tensor& input_tensor = cc->Inputs().Tag(kTensor).Get<tf::Tensor>();
  ABSL_CHECK(1 == input_tensor.dims() || 2 == input_tensor.dims())
      << "Only 1-D or 2-D Tensors can be converted to matrices.";
  const int32_t length = input_tensor.dim_size(input_tensor.dims() - 1);
  const int32_t width =
      (1 == input_tensor.dims()) ? 1 : input_tensor.dim_size(0);
  if (header_.has_num_channels()) {
    RET_CHECK_EQ(length, header_.num_channels())
        << "The number of channels at runtime does not match the header.";
  }
  if (header_.has_num_samples()) {
    RET_CHECK_EQ(width, header_.num_samples())
        << "The number of samples at runtime does not match the header.";
  }
  auto output = absl::make_unique<Matrix>(width, length);
  *output =
      Eigen::MatrixXf::Map(input_tensor.flat<float>().data(), length, width);
  cc->Outputs().Tag(kMatrix).Add(output.release(), cc->InputTimestamp());
  return absl::OkStatus();
}

}  // namespace mediapipe
