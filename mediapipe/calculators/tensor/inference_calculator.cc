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

#include "mediapipe/calculators/tensor/inference_calculator.h"

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "mediapipe/calculators/tensor/inference_calculator.pb.h"
#include "mediapipe/framework/api2/node.h"
#include "mediapipe/framework/api2/packet.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/port.h"
#include "mediapipe/framework/port/ret_check.h"
#include "mediapipe/framework/port/status_macros.h"
#include "mediapipe/framework/tool/subgraph_expansion.h"
#include "mediapipe/util/tflite/tflite_model_loader.h"
#include "tensorflow/lite/core/api/op_resolver.h"
#include "tensorflow/lite/kernels/register.h"

namespace mediapipe {
namespace api2 {

class InferenceCalculatorSelectorImpl
    : public SubgraphImpl<InferenceCalculatorSelector,
                          InferenceCalculatorSelectorImpl> {
 public:
  absl::StatusOr<CalculatorGraphConfig> GetConfig(
      const CalculatorGraphConfig::Node& subgraph_node) override {
    const auto& options =
        Subgraph::GetOptions<mediapipe::InferenceCalculatorOptions>(
            subgraph_node);
    std::vector<absl::string_view> impls;

    const bool should_use_gpu =
        !options.has_delegate() ||  // Use GPU delegate if not specified
        (options.has_delegate() && options.delegate().has_gpu());
    if (should_use_gpu) {
      const auto& api = options.delegate().gpu().api();
      using Gpu = ::mediapipe::InferenceCalculatorOptions::Delegate::Gpu;
#if MEDIAPIPE_METAL_ENABLED
      impls.emplace_back("Metal");
#endif

      const bool prefer_gl_advanced =
          options.delegate().gpu().use_advanced_gpu_api() &&
          (api == Gpu::ANY || api == Gpu::OPENGL || api == Gpu::OPENCL);
      if (prefer_gl_advanced) {
        impls.emplace_back("GlAdvanced");
        impls.emplace_back("Gl");
      } else {
        impls.emplace_back("Gl");
        impls.emplace_back("GlAdvanced");
      }
    }
    impls.emplace_back("Cpu");
    impls.emplace_back("Xnnpack");
    for (const auto& suffix : impls) {
      const auto impl = absl::StrCat("InferenceCalculator", suffix);
      if (!CalculatorBaseRegistry::IsRegistered(impl)) {
        ABSL_LOG(WARNING) << absl::StrFormat(
            "Missing InferenceCalculator registration for %s. Check if the "
            "build dependency is present.",
            impl);
        continue;
      };

      VLOG(1) << "Using " << suffix << " for InferenceCalculator with "
              << (options.has_model_path()
                      ? "model " + options.model_path()
                      : "output_stream " +
                            (subgraph_node.output_stream_size() > 0
                                 ? subgraph_node.output_stream(0)
                                 : "<none>"));
      CalculatorGraphConfig::Node impl_node = subgraph_node;
      impl_node.set_calculator(impl);
      return tool::MakeSingleNodeGraph(std::move(impl_node));
    }
    return absl::UnimplementedError("no implementation available");
  }
};

absl::Status InferenceCalculator::TensorContractCheck(CalculatorContract* cc) {
  RET_CHECK(kInTensors(cc).IsConnected() ^ (kInTensor(cc).Count() > 0))
      << "Exactly one of TENSORS and TENSOR must be used for input.";
  RET_CHECK(kOutTensors(cc).IsConnected() ^ (kOutTensor(cc).Count() > 0))
      << "Exactly one of TENSORS and TENSOR must be used for output.";
  return absl::OkStatus();
}

absl::StatusOr<Packet<TfLiteModelPtr>> InferenceCalculator::GetModelAsPacket(
    CalculatorContext* cc) {
  const auto& options = cc->Options<mediapipe::InferenceCalculatorOptions>();
  if (!options.model_path().empty()) {
    return TfLiteModelLoader::LoadFromPath(options.model_path(),
                                           options.try_mmap_model());
  }
  if (!kSideInModel(cc).IsEmpty()) return kSideInModel(cc);
  return absl::Status(absl::StatusCode::kNotFound,
                      "Must specify TFLite model as path or loaded model.");
}

absl::StatusOr<Packet<tflite::OpResolver>>
InferenceCalculator::GetOpResolverAsPacket(CalculatorContext* cc) {
  if (kSideInOpResolver(cc).IsConnected()) {
    return kSideInOpResolver(cc).As<tflite::OpResolver>();
  } else if (kSideInCustomOpResolver(cc).IsConnected()) {
    return kSideInCustomOpResolver(cc).As<tflite::OpResolver>();
  }
  return PacketAdopting<tflite::OpResolver>(
      std::make_unique<
          tflite::ops::builtin::BuiltinOpResolverWithoutDefaultDelegates>());
}

void InferenceCalculator::WarnFeedbackTensorsUnsupported(
    CalculatorContract* cc) {
  const auto& options = cc->Options<mediapipe::InferenceCalculatorOptions>();
  if (options.has_input_output_config() &&
      !options.input_output_config().feedback_tensor_links().empty()) {
    ABSL_LOG(WARNING)
        << "Feedback tensor support is only available for CPU and "
        << "XNNPACK inference. Ignoring "
           "input_output_config.feedback_tensor_links option.";
  }
}

}  // namespace api2
}  // namespace mediapipe
