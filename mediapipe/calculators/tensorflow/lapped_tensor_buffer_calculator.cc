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

#include <vector>

#include "absl/memory/memory.h"
#include "absl/types/span.h"
#include "mediapipe/calculators/tensorflow/lapped_tensor_buffer_calculator.pb.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/port/ret_check.h"
#include "mediapipe/framework/port/status.h"
#include "mediapipe/framework/profiler/circular_buffer.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/tensor_util.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/core/status.h"

namespace mediapipe {

const char kBufferSize[] = "BUFFER_SIZE";
const char kOverlap[] = "OVERLAP";
const char kTimestampOffset[] = "TIMESTAMP_OFFSET";
const char kCalculatorOptions[] = "CALCULATOR_OPTIONS";

namespace tf = tensorflow;

// Given an input stream of tensors, concatenates the tensors over timesteps.
// The concatenated output tensors can be specified to have overlap between
// output timesteps. The tensors are concatenated along the first dimension, and
// a flag controls whether a new first dimension is inserted before
// concatenation.
//
// Currently, the number of tensors output will be buffer_size less than the
// number of input tensors because no padding is implemented and only full
// buffers are output.
//
// The timestamp of the output batch will match the timestamp of the first
// tensor in that batch by default. (e.g. when buffer_size frames are added, the
// output tensor will have the timestamp of the first input.). This behavior can
// be adjusted by the timestamp_offset option.
//
// Example config:
// node {
//   calculator: "LappedTensorBufferCalculator"
//   input_stream: "input_tensor"
//   output_stream: "output_tensor"
//   options {
//     [mediapipe.LappedTensorBufferCalculatorOptions.ext] {
//       buffer_size: 2
//       overlap: 1
//       add_batch_dim_to_tensors: false
//     }
//   }
// }
class LappedTensorBufferCalculator : public CalculatorBase {
 public:
  static ::mediapipe::Status GetContract(CalculatorContract* cc);

  ::mediapipe::Status Open(CalculatorContext* cc) override;
  ::mediapipe::Status Process(CalculatorContext* cc) override;

 private:
  // Adds a batch dimension to the input tensor if specified in the calculator
  // options.
  ::mediapipe::Status AddBatchDimension(tf::Tensor* input_tensor);

  int steps_until_output_;
  int buffer_size_;
  int overlap_;
  int timestamp_offset_;
  std::unique_ptr<CircularBuffer<Timestamp>> timestamp_buffer_;
  std::unique_ptr<CircularBuffer<tf::Tensor>> buffer_;
  LappedTensorBufferCalculatorOptions options_;
};
REGISTER_CALCULATOR(LappedTensorBufferCalculator);

::mediapipe::Status LappedTensorBufferCalculator::GetContract(
    CalculatorContract* cc) {
  RET_CHECK_EQ(cc->Inputs().NumEntries(), 1)
      << "Only one input stream is supported.";
  cc->Inputs().Index(0).Set<tf::Tensor>(
      // tensorflow::Tensor stream.
  );
  RET_CHECK_EQ(cc->Inputs().NumEntries(), 1)
      << "Only one output stream is supported.";

  if (cc->InputSidePackets().HasTag(kBufferSize)) {
    cc->InputSidePackets().Tag(kBufferSize).Set<int>();
  }
  if (cc->InputSidePackets().HasTag(kOverlap)) {
    cc->InputSidePackets().Tag(kOverlap).Set<int>();
  }
  if (cc->InputSidePackets().HasTag(kTimestampOffset)) {
    cc->InputSidePackets().Tag(kTimestampOffset).Set<int>();
  }
  if (cc->InputSidePackets().HasTag(kCalculatorOptions)) {
    cc->InputSidePackets()
        .Tag(kCalculatorOptions)
        .Set<LappedTensorBufferCalculatorOptions>();
  }
  cc->Outputs().Index(0).Set<tf::Tensor>(
      // Output tensorflow::Tensor stream with possibly overlapping steps.
  );
  return ::mediapipe::OkStatus();
}

::mediapipe::Status LappedTensorBufferCalculator::Open(CalculatorContext* cc) {
  options_ = cc->Options<LappedTensorBufferCalculatorOptions>();
  if (cc->InputSidePackets().HasTag(kCalculatorOptions)) {
    options_ = cc->InputSidePackets()
                   .Tag(kCalculatorOptions)
                   .Get<LappedTensorBufferCalculatorOptions>();
  }
  buffer_size_ = options_.buffer_size();
  if (cc->InputSidePackets().HasTag(kBufferSize)) {
    buffer_size_ = cc->InputSidePackets().Tag(kBufferSize).Get<int>();
  }
  overlap_ = options_.overlap();
  if (cc->InputSidePackets().HasTag(kOverlap)) {
    overlap_ = cc->InputSidePackets().Tag(kOverlap).Get<int>();
  }
  timestamp_offset_ = options_.timestamp_offset();
  if (cc->InputSidePackets().HasTag(kTimestampOffset)) {
    timestamp_offset_ = cc->InputSidePackets().Tag(kTimestampOffset).Get<int>();
  }

  RET_CHECK_LT(overlap_, buffer_size_);
  RET_CHECK_GE(timestamp_offset_, 0)
      << "Negative timestamp_offset is not allowed.";
  RET_CHECK_LT(timestamp_offset_, buffer_size_)
      << "output_frame_num_offset has to be less than buffer_size.";
  timestamp_buffer_ =
      absl::make_unique<CircularBuffer<Timestamp>>(buffer_size_);
  buffer_ = absl::make_unique<CircularBuffer<tf::Tensor>>(buffer_size_);
  steps_until_output_ = buffer_size_;
  return ::mediapipe::OkStatus();
}

::mediapipe::Status LappedTensorBufferCalculator::Process(
    CalculatorContext* cc) {
  // These are cheap, shallow copies.
  tensorflow::Tensor input_tensor(
      cc->Inputs().Index(0).Get<tensorflow::Tensor>());
  if (options_.add_batch_dim_to_tensors()) {
    RET_CHECK_OK(AddBatchDimension(&input_tensor));
  }
  buffer_->push_back(input_tensor);
  timestamp_buffer_->push_back(cc->InputTimestamp());
  --steps_until_output_;

  if (steps_until_output_ <= 0) {
    auto concatenated = ::absl::make_unique<tf::Tensor>();

    const tf::Status concat_status = tf::tensor::Concat(
        std::vector<tf::Tensor>(buffer_->begin(), buffer_->end()),
        concatenated.get());
    RET_CHECK(concat_status.ok()) << concat_status.ToString();

    cc->Outputs().Index(0).Add(concatenated.release(),
                               timestamp_buffer_->Get(timestamp_offset_));

    steps_until_output_ = buffer_size_ - overlap_;
  }
  return ::mediapipe::OkStatus();
}

// Adds a batch dimension to the input tensor if specified in the calculator
// options.
::mediapipe::Status LappedTensorBufferCalculator::AddBatchDimension(
    tf::Tensor* input_tensor) {
  if (options_.add_batch_dim_to_tensors()) {
    tf::TensorShape new_shape(input_tensor->shape());
    new_shape.InsertDim(0, 1);
    RET_CHECK(input_tensor->CopyFrom(*input_tensor, new_shape))
        << "Could not add 0th dimension to tensor without changing its shape."
        << " Current shape: " << input_tensor->shape().DebugString();
  }
  return ::mediapipe::OkStatus();
}

}  // namespace mediapipe
