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

#include "absl/memory/memory.h"
#include "mediapipe/calculators/tensor/inference_calculator.h"
#include "onnxruntime_cxx_api.h"
#include <cstring>
#include <memory>
#include <string>
#include <vector>

namespace mediapipe {
namespace api2 {

namespace {

int64_t value_size_of(const std::vector<int64_t>& dims) {
    if (dims.empty()) return 0;
    int64_t value_size = 1;
    for (const auto& size : dims) value_size *= size;
    return value_size;
}

}  // namespace

class InferenceCalculatorOnnxTensorRTImpl : public NodeImpl<InferenceCalculatorOnnxTensorRT, InferenceCalculatorOnnxTensorRTImpl> {
public:
    static absl::Status UpdateContract(CalculatorContract* cc);

    absl::Status Open(CalculatorContext* cc) override;
    absl::Status Process(CalculatorContext* cc) override;

private:
    absl::Status LoadModel(const std::string& path);

    Ort::Env env_;
    std::unique_ptr<Ort::Session> session_;
    Ort::AllocatorWithDefaultOptions allocator;
    Ort::MemoryInfo memory_info_handler = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    std::vector<const char*> m_input_names;
    std::vector<const char*> m_output_names;
};

absl::Status InferenceCalculatorOnnxTensorRTImpl::UpdateContract(CalculatorContract* cc) {
    const auto& options = cc->Options<::mediapipe::InferenceCalculatorOptions>();
    RET_CHECK(!options.model_path().empty() ^ kSideInModel(cc).IsConnected())
        << "Either model as side packet or model path in options is required.";
    return absl::OkStatus();
}

absl::Status InferenceCalculatorOnnxTensorRTImpl::LoadModel(const std::string& path) {
    auto model_path = std::wstring(path.begin(), path.end());
    Ort::SessionOptions session_options;
    OrtTensorRTProviderOptions trt_options{};
    trt_options.device_id = 0;
    trt_options.trt_max_workspace_size = 1073741824;
    trt_options.trt_max_partition_iterations = 1000;
    trt_options.trt_min_subgraph_size = 1;
    trt_options.trt_engine_cache_enable = 1;
    trt_options.trt_engine_cache_path = "D:/code/mediapipe/mediapipe/modules/tensorrt/";
    trt_options.trt_dump_subgraphs = 1;
    session_options.AppendExecutionProvider_TensorRT(trt_options);
    session_ = std::make_unique<Ort::Session>(env_, model_path.c_str(), session_options);
    size_t num_input_nodes = session_->GetInputCount();
    size_t num_output_nodes = session_->GetOutputCount();
    m_input_names.reserve(num_input_nodes);
    m_output_names.reserve(num_output_nodes);
    for (int i = 0; i < num_input_nodes; i++) {
        char* input_name = session_->GetInputName(i, allocator);
        m_input_names.push_back(input_name);
    }
    for (int i = 0; i < num_output_nodes; i++) {
        char* output_name = session_->GetOutputName(i, allocator);
        m_output_names.push_back(output_name);
    }
    return absl::OkStatus();
}

absl::Status InferenceCalculatorOnnxTensorRTImpl::Open(CalculatorContext* cc) {
    const auto& options = cc->Options<mediapipe::InferenceCalculatorOptions>();
    if (!options.model_path().empty()) {
        return LoadModel(options.model_path());
    }
    return absl::Status(mediapipe::StatusCode::kNotFound, "Must specify Onnx model path.");
}

absl::Status InferenceCalculatorOnnxTensorRTImpl::Process(CalculatorContext* cc) {
    if (kInTensors(cc).IsEmpty()) {
        return absl::OkStatus();
    }
    const auto& input_tensors = *kInTensors(cc);
    RET_CHECK(!input_tensors.empty());
    auto input_tensor_type = int(input_tensors[0].element_type());
    std::vector<Ort::Value> ort_input_tensors;
    ort_input_tensors.reserve(input_tensors.size());
    for (const auto& tensor : input_tensors) {
        auto& inputDims = tensor.shape().dims;
        std::vector<int64_t> src_dims{inputDims[0], inputDims[1], inputDims[2], inputDims[3]};
        auto src_value_size = value_size_of(src_dims);
        auto input_tensor_view = tensor.GetCpuReadView();
        auto input_tensor_buffer = const_cast<float*>(input_tensor_view.buffer<float>());
        auto tmp_tensor = Ort::Value::CreateTensor<float>(memory_info_handler, input_tensor_buffer, src_value_size, src_dims.data(), src_dims.size());
        ort_input_tensors.emplace_back(std::move(tmp_tensor));
    }
    auto output_tensors = absl::make_unique<std::vector<Tensor>>();
    std::vector<Ort::Value> onnx_output_tensors;
    try {
        onnx_output_tensors = session_->Run(
            Ort::RunOptions{nullptr}, m_input_names.data(),
            ort_input_tensors.data(), ort_input_tensors.size(), m_output_names.data(),
            m_output_names.size());
    } catch (Ort::Exception& e) {
        LOG(ERROR) << "Run error msg:" << e.what();
    }
    for (const auto& tensor : onnx_output_tensors) {
        auto info = tensor.GetTensorTypeAndShapeInfo();
        auto dims = info.GetShape();
        std::vector<int> tmp_dims;
        for (const auto& i : dims) {
            tmp_dims.push_back(i);
        }
        output_tensors->emplace_back(Tensor::ElementType::kFloat32, Tensor::Shape{tmp_dims});
        auto cpu_view = output_tensors->back().GetCpuWriteView();
        std::memcpy(cpu_view.buffer<float>(), tensor.GetTensorData<float>(), output_tensors->back().bytes());
    }
    kOutTensors(cc).Send(std::move(output_tensors));
    return absl::OkStatus();
}

}  // namespace api2
}  // namespace mediapipe
