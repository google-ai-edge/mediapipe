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

::mediapipe::Status FillTimeSeriesHeaderIfValid(const Packet& header_packet,
                                                TimeSeriesHeader* header) {
  CHECK(header);
  if (header_packet.IsEmpty()) {
    return ::mediapipe::UnknownError("No header found.");
  }
  if (!header_packet.ValidateAsType<TimeSeriesHeader>().ok()) {
    return ::mediapipe::UnknownError(
        "Packet does not contain TimeSeriesHeader.");
  }
  *header = header_packet.Get<TimeSeriesHeader>();
  if (header->has_sample_rate() && header->sample_rate() >= 0 &&
      header->has_num_channels() && header->num_channels() >= 0) {
    return ::mediapipe::OkStatus();
  } else {
    std::string error_message =
        "TimeSeriesHeader is missing necessary fields: "
        "sample_rate or num_channels, or one of their values is negative. ";
#ifndef MEDIAPIPE_MOBILE
    absl::StrAppend(&error_message, "Got header:\n",
                    header->ShortDebugString());
#endif
    return ::mediapipe::InvalidArgumentError(error_message);
  }
}

}  // namespace

// Converts a 1-D or a 2-D Tensor into a 2 dimensional Matrix.
// Input:
// -- 1-D or 2-D Tensor
// Output:
// -- Matrix with the same values as the Tensor
// If input tensor is 1 dimensional, the ouput Matrix is of (1xn) shape.
// If input tensor is 2 dimensional (batched), the ouput Matrix is (mxn) shape.
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
// This designed was discussed in http://g/speakeranalysis/4uyx7cNRwJY and
// http://g/daredevil-project/VB26tcseUy8.
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
  static ::mediapipe::Status GetContract(CalculatorContract* cc);

  ::mediapipe::Status Open(CalculatorContext* cc) override;
  ::mediapipe::Status Process(CalculatorContext* cc) override;

  // Store header information so that we can verify the inputs in process().
  TimeSeriesHeader header_;
};
REGISTER_CALCULATOR(TensorToMatrixCalculator);

::mediapipe::Status TensorToMatrixCalculator::GetContract(
    CalculatorContract* cc) {
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
  return ::mediapipe::OkStatus();
}

::mediapipe::Status TensorToMatrixCalculator::Open(CalculatorContext* cc) {
  auto input_header = absl::make_unique<TimeSeriesHeader>();
  ::mediapipe::Status header_status;
  if (cc->Inputs().HasTag(kReference)) {
    header_status = FillTimeSeriesHeaderIfValid(
        cc->Inputs().Tag(kReference).Header(), input_header.get());
  }
  if (header_status.ok()) {
    if (cc->Options<TensorToMatrixCalculatorOptions>()
            .has_time_series_header_overrides()) {
      // From design discussions with Daredevil, we only want to support single
      // sample per packet for now, so we hardcode the sample_rate based on the
      // packet_rate of the REFERENCE and fail noisily if we cannot. An
      // alternative would be to calculate the sample_rate from the reference
      // sample_rate and the change in num_samples between the reference and
      // override headers:
      // sample_rate_output = sample_rate_reference /
      //                      (num_samples_override / num_samples_reference)
      const TimeSeriesHeader& override_header =
          cc->Options<TensorToMatrixCalculatorOptions>()
              .time_series_header_overrides();
      input_header->MergeFrom(override_header);
      CHECK(input_header->has_packet_rate())
          << "The TimeSeriesHeader.packet_rate must be set.";
      if (!override_header.has_sample_rate()) {
        CHECK_EQ(input_header->num_samples(), 1)
            << "Currently the time series can only output single samples.";
        input_header->set_sample_rate(input_header->packet_rate());
      }
    }
    header_ = *input_header;
    cc->Outputs().Tag(kMatrix).SetHeader(Adopt(input_header.release()));
  }
  cc->SetOffset(TimestampDiff(0));
  return ::mediapipe::OkStatus();
}

::mediapipe::Status TensorToMatrixCalculator::Process(CalculatorContext* cc) {
  // Daredevil requested CHECK for noisy failures rather than quieter RET_CHECK
  // failures. These are absolute conditions of the graph for the graph to be
  // valid, and if it is violated by any input anywhere, the graph will be
  // invalid for all inputs. A hard CHECK will enable faster debugging by
  // immediately exiting and more prominently displaying error messages.
  // Do not replace with RET_CHECKs.

  // Verify that each reference stream packet corresponds to a tensor packet
  // otherwise the header information is invalid. If we don't have a reference
  // stream, Process() is only called when we have an input tensor and this is
  // always True.
  CHECK(cc->Inputs().HasTag(kTensor))
      << "Tensor stream not available at same timestamp as the reference "
         "stream.";

  const tf::Tensor& input_tensor = cc->Inputs().Tag(kTensor).Get<tf::Tensor>();
  CHECK(1 == input_tensor.dims() || 2 == input_tensor.dims())
      << "Only 1-D or 2-D Tensors can be converted to matrices.";
  const int32 length = input_tensor.dim_size(input_tensor.dims() - 1);
  const int32 width = (1 == input_tensor.dims()) ? 1 : input_tensor.dim_size(0);
  if (header_.has_num_channels()) {
    CHECK_EQ(length, header_.num_channels())
        << "The number of channels at runtime does not match the header.";
  }
  if (header_.has_num_samples()) {
    CHECK_EQ(width, header_.num_samples())
        << "The number of samples at runtime does not match the header.";
    ;
  }
  auto output = absl::make_unique<Matrix>(width, length);
  *output =
      Eigen::MatrixXf::Map(input_tensor.flat<float>().data(), length, width);
  cc->Outputs().Tag(kMatrix).Add(output.release(), cc->InputTimestamp());
  return ::mediapipe::OkStatus();
}

}  // namespace mediapipe
