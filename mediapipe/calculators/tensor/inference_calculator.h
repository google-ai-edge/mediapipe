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

#ifndef MEDIAPIPE_CALCULATORS_TENSOR_INFERENCE_CALCULATOR_H_
#define MEDIAPIPE_CALCULATORS_TENSOR_INFERENCE_CALCULATOR_H_

#include <cstring>
#include <memory>
#include <string>
#include <vector>

#include "absl/memory/memory.h"
#include "mediapipe/calculators/tensor/inference_calculator.pb.h"
#include "mediapipe/framework/api2/node.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/formats/tensor.h"
#include "mediapipe/framework/port/ret_check.h"
#include "mediapipe/util/tflite/tflite_model_loader.h"
#include "tensorflow/lite/core/api/op_resolver.h"
#include "tensorflow/lite/error_reporter.h"
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model.h"

namespace mediapipe {
namespace api2 {

// Runs inference on the provided input Tensors and TFLite model.
//
// Creates an interpreter with given model and calls invoke().
// Optionally run inference on CPU/GPU.
//
// This calculator can be used with TensorConverterCalculator to get the
// appropriate inputs.
//
// When the input tensors are on CPU, gpu inference is optional and can be
// specified in the calculator options.
// When the input tensors are on GPU, inference is GPU and output can be CPU or
// GPU.
//
// Input:
//  TENSORS - Vector of Tensors
//
// Output:
//  TENSORS - Vector of Tensors
//
// Input side packet:
//  DEPRECATED: Prefer to use the "OP_RESOLVER" input side packet instead.
//  CUSTOM_OP_RESOLVER (optional) - Use a custom op resolver,
//                                  instead of the builtin one.
//  OP_RESOLVER (optional) - Use to provide tflite op resolver
//                           (tflite::OpResolver)
//  MODEL (optional) - Use to specify TfLite model
//                     (std::unique_ptr<tflite::FlatBufferModel,
//                       std::function<void(tflite::FlatBufferModel*)>>)
//
// Example use:
// node {
//   calculator: "InferenceCalculator"
//   input_stream: "TENSORS:tensor_image"
//   output_stream: "TENSORS:tensors"
//   options: {
//     [mediapipe.InferenceCalculatorOptions.ext] {
//       model_path: "modelname.tflite"
//     }
//   }
// }
//
// or
//
// node {
//   calculator: "InferenceCalculator"
//   input_stream: "TENSORS:tensor_image"
//   input_side_packet: "MODEL:model"
//   output_stream: "TENSORS:tensors"
//   options: {
//     [mediapipe.InferenceCalculatorOptions.ext] {
//       model_path: "modelname.tflite"
//       delegate { gpu {} }
//     }
//   }
// }
//
// IMPORTANT Notes:
//  Tensors are assumed to be ordered correctly (sequentially added to model).
//  Input tensors are assumed to be of the correct size and already normalized.

class InferenceCalculator : public NodeIntf {
 public:
  static constexpr Input<std::vector<Tensor>> kInTensors{"TENSORS"};
  // Deprecated. Prefers to use "OP_RESOLVER" input side packet instead.
  // TODO: Removes the "CUSTOM_OP_RESOLVER" side input after the
  // migration.
  static constexpr SideInput<tflite::ops::builtin::BuiltinOpResolver>::Optional
      kSideInCustomOpResolver{"CUSTOM_OP_RESOLVER"};
  static constexpr SideInput<tflite::OpResolver>::Optional kSideInOpResolver{
      "OP_RESOLVER"};
  static constexpr SideInput<TfLiteModelPtr>::Optional kSideInModel{"MODEL"};
  static constexpr Output<std::vector<Tensor>> kOutTensors{"TENSORS"};
  static constexpr SideInput<
      mediapipe::InferenceCalculatorOptions::Delegate>::Optional kDelegate{
      "DELEGATE"};
  MEDIAPIPE_NODE_CONTRACT(kInTensors, kSideInCustomOpResolver,
                          kSideInOpResolver, kSideInModel, kOutTensors,
                          kDelegate);

 protected:
  using TfLiteDelegatePtr =
      std::unique_ptr<TfLiteDelegate, std::function<void(TfLiteDelegate*)>>;

  absl::StatusOr<Packet<TfLiteModelPtr>> GetModelAsPacket(
      CalculatorContext* cc);

  absl::StatusOr<Packet<tflite::OpResolver>> GetOpResolverAsPacket(
      CalculatorContext* cc);
};

struct InferenceCalculatorSelector : public InferenceCalculator {
  static constexpr char kCalculatorName[] = "InferenceCalculator";
};

struct InferenceCalculatorGl : public InferenceCalculator {
  static constexpr char kCalculatorName[] = "InferenceCalculatorGl";
};

struct InferenceCalculatorGlAdvanced : public InferenceCalculator {
  static constexpr char kCalculatorName[] = "InferenceCalculatorGlAdvanced";
};

struct InferenceCalculatorMetal : public InferenceCalculator {
  static constexpr char kCalculatorName[] = "InferenceCalculatorMetal";
};

struct InferenceCalculatorCpu : public InferenceCalculator {
  static constexpr char kCalculatorName[] = "InferenceCalculatorCpu";
};

}  // namespace api2
}  // namespace mediapipe

#endif  // MEDIAPIPE_CALCULATORS_TENSOR_INFERENCE_CALCULATOR_H_
