// Copyright (c) 2023 Intel Corporation
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
// From benchmark_app code
// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <openvino/openvino.hpp>

#include "absl/base/thread_annotations.h"
#include "absl/memory/memory.h"
#include "absl/strings/str_split.h"
#include "absl/synchronization/mutex.h"
#include "mediapipe/calculators/openvino/openvino_inference_calculator.pb.h"
#include "mediapipe/framework/calculator_context.h"
#include "mediapipe/framework/calculator_framework.h"

#include "mediapipe/calculators/openvino/internal/infer_request_wrap.hpp"

namespace {
    constexpr char kTensorsTag[] = "TENSORS";
    constexpr char kRemoteTensorsTag[] = "TENSORS_REMOTE";
}  // namespace

namespace mediapipe {

class OpenVINOInferenceCalculator : public CalculatorBase {
public:
    OpenVINOInferenceCalculator() {
    }

    static absl::Status GetContract(CalculatorContract *cc) {
      RET_CHECK(cc->Inputs().HasTag(kTensorsTag) ^
                cc->Inputs().HasTag(kRemoteTensorsTag));
      RET_CHECK(cc->Outputs().HasTag(kTensorsTag) ^
                cc->Outputs().HasTag(kRemoteTensorsTag));

      const auto& options =
              cc->Options<::mediapipe::OpenVINOInferenceCalculatorOptions>();
      RET_CHECK(!options.model_path().empty())
//                cc->InputSidePackets().HasTag(kModelTag))
                << "Either model as side packet or model path in options is required.";

      if (cc->Inputs().HasTag(kTensorsTag))
        cc->Inputs().Tag(kTensorsTag).Set<std::vector<ov::Tensor>>();
      if (cc->Outputs().HasTag(kTensorsTag))
        cc->Outputs().Tag(kTensorsTag).Set<std::vector<ov::Tensor>>();

      if (cc->Inputs().HasTag(kRemoteTensorsTag))
        cc->Inputs().Tag(kRemoteTensorsTag).Set<std::vector<ov::RemoteTensor>>();
      if (cc->Outputs().HasTag(kRemoteTensorsTag))
        cc->Outputs().Tag(kRemoteTensorsTag).Set<std::vector<ov::RemoteTensor>>();

//      if (cc->InputSidePackets().HasTag(kModelTag)) {
//        cc->InputSidePackets().Tag(kModelTag).Set<TfLiteModelPtr>();
//      }

      // Assign this calculator's default InputStreamHandler.
      // TODO
//      cc->SetInputStreamHandler("FixedSizeInputStreamHandler");

      return absl::OkStatus();
    }

    absl::Status Open(CalculatorContext *cc) override {
      const auto& options =
              cc->Options<::mediapipe::OpenVINOInferenceCalculatorOptions>();

      RET_CHECK(!options.model_path().empty())
        << "Model path should be defined in options";

      // TODO: process device message

      ov::Core core;
      model_ = core.compile_model(options.model_path(), "CPU");

      // TODO: add perf hints
      // TODO: get nireq from perf hints
      size_t nireq = 4;
      infer_requests_queue_ = std::make_unique<InferRequestsQueue>(model_, nireq);
      return absl::OkStatus();
    }

    absl::Status Process(CalculatorContext *cc) override {
      // Receive tensor from input message
      if (cc->Inputs().Tag(kTensorsTag).IsEmpty()) {
        return absl::OkStatus();
      }

      // Get infer request
      auto infer_request = infer_requests_queue_->get_idle_request();
      if (!infer_request) {
        return absl::InternalError("No idle inference requests available");
      }

      // Read CPU input into tensors.
      const auto& input_tensors =
              cc->Inputs().Tag(kTensorsTag).Get<std::vector<ov::Tensor>>();
      // TODO: add support for models with >1 inputs
//      RET_CHECK_GT(input_tensors.size(), 0);
      RET_CHECK_EQ(input_tensors.size(), 1);
      for (int i = 0; i < input_tensors.size(); ++i) {
        infer_request->set_input_tensor(i, input_tensors[i]);
      }
//        RET_CHECK(input_tensor->data.raw);

      // TODO: use async inference
      infer_request->infer();

      // Process output tensors
      auto output_tensors = absl::make_unique<std::vector<ov::Tensor>>();
      for (int i = 0; i < model_.outputs().size(); ++i) {
        ov::Tensor out_tensor = infer_request->get_output_tensor(i);
        output_tensors->emplace_back(out_tensor);
      }

      // Prepare calculator output
      cc->Outputs()
        .Tag(kTensorsTag)
        .Add(output_tensors.release(), cc->InputTimestamp());

      return absl::OkStatus();
    }

    absl::Status Close(CalculatorContext *cc) override {
      return absl::OkStatus();
    }

private:
    ov::CompiledModel model_;
    std::unique_ptr<InferRequestsQueue> infer_requests_queue_;
};

REGISTER_CALCULATOR(OpenVINOInferenceCalculator);

} // namespace mediapipe
