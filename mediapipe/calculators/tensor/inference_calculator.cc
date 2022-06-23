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

#include <cstring>
#include <memory>
#include <string>
#include <vector>

#include "absl/strings/string_view.h"
#include "mediapipe/calculators/tensor/inference_calculator.pb.h"
#include "mediapipe/framework/api2/packet.h"
#include "mediapipe/framework/tool/subgraph_expansion.h"
#include "tensorflow/lite/core/api/op_resolver.h"

namespace mediapipe {
namespace api2 {

class InferenceCalculatorSelectorImpl
    : public SubgraphImpl<InferenceCalculatorSelector,
                          InferenceCalculatorSelectorImpl> {
 public:
  absl::StatusOr<CalculatorGraphConfig> GetConfig(
      const CalculatorGraphConfig::Node& subgraph_node) {
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
      impls.emplace_back("Metal");
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
    for (const auto& suffix : impls) {
      const auto impl = absl::StrCat("InferenceCalculator", suffix);
      if (!mediapipe::CalculatorBaseRegistry::IsRegistered(impl)) continue;
      CalculatorGraphConfig::Node impl_node = subgraph_node;
      impl_node.set_calculator(impl);
      return tool::MakeSingleNodeGraph(std::move(impl_node));
    }
    return absl::UnimplementedError("no implementation available");
  }
};

absl::StatusOr<Packet<TfLiteModelPtr>> InferenceCalculator::GetModelAsPacket(
    CalculatorContext* cc) {
  const auto& options = cc->Options<mediapipe::InferenceCalculatorOptions>();
  if (!options.model_path().empty()) {
    return TfLiteModelLoader::LoadFromPath(options.model_path());
  }
  if (!kSideInModel(cc).IsEmpty()) return kSideInModel(cc);
  return absl::Status(mediapipe::StatusCode::kNotFound,
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

}  // namespace api2
}  // namespace mediapipe
