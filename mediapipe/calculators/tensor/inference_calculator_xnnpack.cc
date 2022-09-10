// Copyright 2022 The MediaPipe Authors.
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

#include <cstdint>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "mediapipe/calculators/tensor/inference_calculator.h"
#include "mediapipe/calculators/tensor/inference_calculator_utils.h"
#include "mediapipe/calculators/tensor/inference_interpreter_delegate_runner.h"
#include "mediapipe/calculators/tensor/inference_runner.h"
#include "tensorflow/lite/delegates/xnnpack/xnnpack_delegate.h"
#include "tensorflow/lite/interpreter.h"

namespace mediapipe {
namespace api2 {

class InferenceCalculatorXnnpackImpl
    : public NodeImpl<InferenceCalculatorXnnpack,
                      InferenceCalculatorXnnpackImpl> {
 public:
  static absl::Status UpdateContract(CalculatorContract* cc);

  absl::Status Open(CalculatorContext* cc) override;
  absl::Status Process(CalculatorContext* cc) override;
  absl::Status Close(CalculatorContext* cc) override;

 private:
  absl::StatusOr<std::unique_ptr<InferenceRunner>> CreateInferenceRunner(
      CalculatorContext* cc);
  absl::StatusOr<TfLiteDelegatePtr> CreateDelegate(CalculatorContext* cc);

  std::unique_ptr<InferenceRunner> inference_runner_;
};

absl::Status InferenceCalculatorXnnpackImpl::UpdateContract(
    CalculatorContract* cc) {
  const auto& options = cc->Options<mediapipe::InferenceCalculatorOptions>();
  RET_CHECK(!options.model_path().empty() ^ kSideInModel(cc).IsConnected())
      << "Either model as side packet or model path in options is required.";

  return absl::OkStatus();
}

absl::Status InferenceCalculatorXnnpackImpl::Open(CalculatorContext* cc) {
  ASSIGN_OR_RETURN(inference_runner_, CreateInferenceRunner(cc));
  return absl::OkStatus();
}

absl::Status InferenceCalculatorXnnpackImpl::Process(CalculatorContext* cc) {
  if (kInTensors(cc).IsEmpty()) {
    return absl::OkStatus();
  }
  const auto& input_tensors = *kInTensors(cc);
  RET_CHECK(!input_tensors.empty());

  ASSIGN_OR_RETURN(std::vector<Tensor> output_tensors,
                   inference_runner_->Run(input_tensors));
  kOutTensors(cc).Send(std::move(output_tensors));
  return absl::OkStatus();
}

absl::Status InferenceCalculatorXnnpackImpl::Close(CalculatorContext* cc) {
  inference_runner_ = nullptr;
  return absl::OkStatus();
}

absl::StatusOr<std::unique_ptr<InferenceRunner>>
InferenceCalculatorXnnpackImpl::CreateInferenceRunner(CalculatorContext* cc) {
  ASSIGN_OR_RETURN(auto model_packet, GetModelAsPacket(cc));
  ASSIGN_OR_RETURN(auto op_resolver_packet, GetOpResolverAsPacket(cc));
  const int interpreter_num_threads =
      cc->Options<mediapipe::InferenceCalculatorOptions>().cpu_num_thread();
  ASSIGN_OR_RETURN(TfLiteDelegatePtr delegate, CreateDelegate(cc));
  return CreateInferenceInterpreterDelegateRunner(
      std::move(model_packet), std::move(op_resolver_packet),
      std::move(delegate), interpreter_num_threads);
}

absl::StatusOr<TfLiteDelegatePtr>
InferenceCalculatorXnnpackImpl::CreateDelegate(CalculatorContext* cc) {
  const auto& calculator_opts =
      cc->Options<mediapipe::InferenceCalculatorOptions>();
  auto opts_delegate = calculator_opts.delegate();
  if (!kDelegate(cc).IsEmpty()) {
    const mediapipe::InferenceCalculatorOptions::Delegate&
        input_side_packet_delegate = kDelegate(cc).Get();
    RET_CHECK(
        input_side_packet_delegate.has_xnnpack() ||
        input_side_packet_delegate.delegate_case() ==
            mediapipe::InferenceCalculatorOptions::Delegate::DELEGATE_NOT_SET)
        << "inference_calculator_cpu only supports delegate input side packet "
        << "for TFLite, XNNPack";
    opts_delegate.MergeFrom(input_side_packet_delegate);
  }
  const bool opts_has_delegate =
      calculator_opts.has_delegate() || !kDelegate(cc).IsEmpty();

  auto xnnpack_opts = TfLiteXNNPackDelegateOptionsDefault();
  xnnpack_opts.num_threads =
      GetXnnpackNumThreads(opts_has_delegate, opts_delegate);
  return TfLiteDelegatePtr(TfLiteXNNPackDelegateCreate(&xnnpack_opts),
                           &TfLiteXNNPackDelegateDelete);
}

}  // namespace api2
}  // namespace mediapipe
