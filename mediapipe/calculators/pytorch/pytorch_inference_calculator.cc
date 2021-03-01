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

#include <cstring>
#include <memory>
#include <string>
#include <vector>

#include "mediapipe/calculators/pytorch/pytorch_inference_calculator.pb.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/port/ret_check.h"
#include "mediapipe/framework/port/status.h"
#include "mediapipe/util/resource_util.h"
#include "torch/script.h"
#include "torch/torch.h"

namespace mediapipe {

namespace {
constexpr char kTensorsTag[] = "TENSORS";

using Inputs = std::vector<torch::jit::IValue>;
using Outputs = torch::Tensor;
}  // namespace

// Calculator Header Section

// Runs inference on the provided input TFLite tensors and TFLite model.
//
// Creates an interpreter with given model and calls invoke().
// Optionally run inference on CPU/GPU.
//
// This calculator is designed to be used with the TfLiteConverterCalcualtor,
// to get the appropriate inputs.
//
// When the input tensors are on CPU, gpu inference is optional and can be
// specified in the calculator options.
//
// Input:
//  TENSORS - Vector of TfLiteTensor of type kTfLiteFloat32 or kTfLiteUInt8
//
// Output:
//  TENSORS - Vector of TfLiteTensor of type kTfLiteFloat32 or kTfLiteUInt8
//
// Example use:
// node {
//   calculator: "PyTorchInferenceCalculator"
//   input_stream: "TENSORS:tensor_image"
//   output_stream: "TENSORS:tensors"
//   options: {
//     [mediapipe.PyTorchInferenceCalculatorOptions.ext] {
//       model_path: "modelname.tflite"
//       delegate { gpu {} }
//     }
//   }
// }
//
// IMPORTANT Notes:
//  This calculator uses FixedSizeInputStreamHandler by default.
//
class PyTorchInferenceCalculator : public CalculatorBase {
 public:
  static ::mediapipe::Status GetContract(CalculatorContract* cc);

  ::mediapipe::Status Open(CalculatorContext* cc) override;
  ::mediapipe::Status Process(CalculatorContext* cc) override;
  ::mediapipe::Status Close(CalculatorContext* cc) override;

 private:
  ::mediapipe::PyTorchInferenceCalculatorOptions options_;
  torch::jit::script::Module module_;
  torch::jit::IValue hidden_state_;
};
REGISTER_CALCULATOR(PyTorchInferenceCalculator);

// Calculator Core Section

::mediapipe::Status PyTorchInferenceCalculator::GetContract(
    CalculatorContract* cc) {
  RET_CHECK(cc->Inputs().HasTag(kTensorsTag));
  RET_CHECK(cc->Outputs().HasTag(kTensorsTag));

  if (cc->Inputs().HasTag(kTensorsTag))
    cc->Inputs().Tag(kTensorsTag).Set<Inputs>();

  if (cc->Outputs().HasTag(kTensorsTag))
    cc->Outputs().Tag(kTensorsTag).Set<Outputs>();

  // Assign this calculator's default InputStreamHandler.
  cc->SetInputStreamHandler("FixedSizeInputStreamHandler");

  return ::mediapipe::OkStatus();
}

::mediapipe::Status PyTorchInferenceCalculator::Open(CalculatorContext* cc) {
  cc->SetOffset(TimestampDiff(0));

  options_ = cc->Options<::mediapipe::PyTorchInferenceCalculatorOptions>();

  std::string model_path = options_.model_path();
  ASSIGN_OR_RETURN(model_path, mediapipe::PathToResourceAsFile(model_path));
  try {
    // https://github.com/pytorch/ios-demo-app/issues/8#issuecomment-612996683
    auto qengines = at::globalContext().supportedQEngines();
    if (std::find(qengines.begin(), qengines.end(), at::QEngine::QNNPACK) !=
        qengines.end()) {
      LOG(INFO) << "Using QEngine at::QEngine::QNNPACK";
      at::globalContext().setQEngine(at::QEngine::QNNPACK);
    }
#if defined(MEDIAPIPE_IOS)
    else {
      RET_CHECK_FAIL() << "QEngine::QNNPACK is required for iOS";
    }
#endif

    module_ = torch::jit::load(model_path);
    module_.eval();
  } catch (const std::exception& e) {
    LOG(ERROR) << e.what();
    return ::mediapipe::UnknownError(e.what());
  }

  if (options_.model_has_hidden_state())
    hidden_state_ = torch::zeros({1, 1, 10});  // TODO: read options

  return ::mediapipe::OkStatus();
}

::mediapipe::Status PyTorchInferenceCalculator::Process(CalculatorContext* cc) {
  const auto inputs = cc->Inputs().Tag(kTensorsTag).Get<Inputs>();
  RET_CHECK_GT(inputs.size(), 0);

  // Disables autograd
  torch::autograd::AutoGradMode guard(false);
  // Disables autograd even more? https://github.com/pytorch/pytorch/pull/26477
  at::AutoNonVariableTypeMode non_var_type_mode(true);

  Outputs out_tensor;
  if (options_.model_has_hidden_state()) {
    RET_CHECK_EQ(inputs.size(), 1) << "Not sure how to forward() hidden state";
    // auto tuple = torch::ivalue::Tuple::create({inp,hidden_state_});
    const auto result = module_.forward({{inputs[0], hidden_state_}});
    const auto out = result.toTuple()->elements();
    out_tensor = out[0].toTensor();
    hidden_state_ = out[1].toTensor();
  } else {
    const auto result = module_.forward(std::move(inputs));
    out_tensor = result.toTensor();
  }

  auto out = absl::make_unique<Outputs>(out_tensor);
  cc->Outputs().Tag(kTensorsTag).Add(out.release(), cc->InputTimestamp());

  return ::mediapipe::OkStatus();
}

::mediapipe::Status PyTorchInferenceCalculator::Close(CalculatorContext* cc) {
  return ::mediapipe::OkStatus();
}

}  // namespace mediapipe
