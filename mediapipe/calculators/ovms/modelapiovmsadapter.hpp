#pragma once
//*****************************************************************************
// Copyright 2023 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//*****************************************************************************
#include <iostream>
#include <map>
#include <memory>
#include <sstream>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include <adapters/inference_adapter.h>  // model_api/model_api/cpp/adapters/include/adapters/inference_adapter.h
#include <openvino/openvino.hpp>

#include "ovms.h"  // NOLINT

// here we need to decide if we have several calculators (1 for OVMS repository, 1-N inside mediapipe)
// for the one inside OVMS repo it makes sense to reuse code from ovms lib

class OVMS_Server_;
typedef struct OVMS_Server_ OVMS_Server;

namespace mediapipe::ovms {

using InferenceOutput = std::map<std::string, ov::Tensor>;
using InferenceInput = std::map<std::string, ov::Tensor>;

using shape_border_t = std::vector<int64_t>;
using shape_min_max_t = std::pair<shape_border_t, shape_border_t>;
using shapes_min_max_t = std::unordered_map<std::string, shape_min_max_t>;
class OVMSInferenceAdapter : public ::InferenceAdapter {
    OVMS_Server* cserver{nullptr};
    const std::string servableName;
    uint32_t servableVersion;
    std::vector<std::string> inputNames;
    std::vector<std::string> outputNames;
    shapes_min_max_t inShapesMinMaxes;
    ov::AnyMap modelConfig;

public:
    // TODO Windows: Fix definition in header - does not compile in cpp.
    OVMSInferenceAdapter(const std::string& servableName, uint32_t servableVersion = 0, OVMS_Server* server = nullptr) :
        servableName(servableName),
        servableVersion(servableVersion) {
        if (nullptr != server) {
            this->cserver = server;
        } else {
            OVMS_ServerNew(&this->cserver);
        }
    }
    virtual ~OVMSInferenceAdapter();
    InferenceOutput infer(const InferenceInput& input) override;
    void loadModel(const std::shared_ptr<const ov::Model>& model, ov::Core& core,
        const std::string& device, const ov::AnyMap& compilationConfig, size_t max_num_requests = 1) override;
    void inferAsync(const InferenceInput& input, const CallbackData callback_args) override;
    void setCallback(std::function<void(ov::InferRequest, const CallbackData)> callback);
    bool isReady();
    void awaitAll();
    void awaitAny();
    size_t getNumAsyncExecutors() const;
    ov::PartialShape getInputShape(const std::string& inputName) const override;
    std::vector<std::string> getInputNames() const override;
    std::vector<std::string> getOutputNames() const override;
    const ov::AnyMap& getModelConfig() const override;
};
}  // namespace mediapipe
