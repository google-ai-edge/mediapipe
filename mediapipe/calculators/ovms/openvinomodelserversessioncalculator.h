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
#include <memory>
#include <mutex>
#include <thread>
#include <chrono>
#include <sstream>
#include <unordered_map>

#include <openvino/openvino.hpp>

#include "ovms.h"           // NOLINT
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/port/canonical_errors.h"
#include "modelapiovmsadapter.hpp"
#include "mediapipe/calculators/ovms/openvinomodelserversessioncalculator.pb.h"
#pragma GCC diagnostic pop
namespace mediapipe {

class OpenVINOModelServerSessionCalculator : public CalculatorBase {
    std::shared_ptr<::InferenceAdapter> adapter;
    std::unordered_map<std::string, std::string> outputNameToTag;
    OVMS_Server* cserver{nullptr};
    static bool triedToStartOVMS;
    static std::mutex loadingMtx;
public:
    static absl::Status GetContract(CalculatorContract* cc);
    absl::Status Close(CalculatorContext* cc) override final;
    absl::Status Open(CalculatorContext* cc) override final;

    absl::Status Process(CalculatorContext* cc) override final;
    static OVMS_LogLevel OvmsLogLevel;
    static const char* OvmsLogLevelEnv;
};

}  // namespace mediapipe
