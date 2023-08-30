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
#include "mediapipe/calculators/ovms/modelapiovmssessioncalculator.pb.h"
#pragma GCC diagnostic pop
// here we need to decide if we have several calculators (1 for OVMS repository, 1-N inside mediapipe)
// for the one inside OVMS repo it makes sense to reuse code from ovms lib
namespace mediapipe {

using ovms::OVMSInferenceAdapter;
using std::endl;

const std::string SESSION_TAG{"SESSION"};
ov::Core UNUSED_OV_CORE;

#define ASSERT_CAPI_STATUS_NULL(C_API_CALL)                                                 \
    {                                                                                       \
        auto* err = C_API_CALL;                                                             \
        if (err != nullptr) {                                                               \
            uint32_t code = 0;                                                              \
            const char* msg = nullptr;                                                      \
            OVMS_StatusGetCode(err, &code);                                                 \
            OVMS_StatusGetDetails(err, &msg);                                               \
            LOG(INFO) << "Error encountred in OVMSCalculator:" << msg << " code: " << code; \
            OVMS_StatusDelete(err);                                                         \
            RET_CHECK(nullptr == err);                                                      \
        }                                                                                   \
    }

#define REPORT_CAPI_STATUS_NULL(C_API_CALL)                                                 \
    {                                                                                       \
        auto* err = C_API_CALL;                                                             \
        if (err != nullptr) {                                                               \
            uint32_t code = 0;                                                              \
            const char* msg = nullptr;                                                      \
            OVMS_StatusGetCode(err, &code);                                                 \
            OVMS_StatusGetDetails(err, &msg);                                               \
            LOG(INFO) << "Error encountred in OVMSCalculator:" << msg << " code: " << code; \
            OVMS_StatusDelete(err);                                                         \
        }                                                                                   \
    }

// Function from ovms/src/string_utils.h
void erase_spaces(std::string& str) {
    str.erase(std::remove_if(str.begin(), str.end(),
                  [](char c) -> bool {
                      return std::isspace<char>(c, std::locale::classic());
                  }),
        str.end());
}

// Function from ovms/src/string_utils.h
std::optional<uint32_t> stou32(const std::string& input) {
    std::string str = input;
    erase_spaces(str);

    if (str.size() > 0 && str[0] == '-') {
        return std::nullopt;
    }

    try {
        uint64_t val = std::stoul(str);
        if (val > std::numeric_limits<uint32_t>::max()) {
            return std::nullopt;
        }
        return {static_cast<uint32_t>(val)};
    } catch (...) {
        return std::nullopt;
    }
}

class ModelAPISessionCalculator : public CalculatorBase {
    std::shared_ptr<::InferenceAdapter> adapter;
    std::unordered_map<std::string, std::string> outputNameToTag;
    // TODO where to place members
    OVMS_Server* cserver{nullptr};
    OVMS_ServerSettings* _serverSettings{nullptr};
    OVMS_ModelsSettings* _modelsSettings{nullptr};
public:
    static absl::Status GetContract(CalculatorContract* cc) {
        LOG(INFO) << "Session GetContract start";
        RET_CHECK(cc->Inputs().GetTags().empty());
        RET_CHECK(cc->Outputs().GetTags().empty());
        cc->OutputSidePackets().Tag(SESSION_TAG.c_str()).Set<std::shared_ptr<::InferenceAdapter>>();
        const auto& options = cc->Options<ModelAPIOVMSSessionCalculatorOptions>();
        RET_CHECK(!options.servable_name().empty());
        // TODO validate version from string
        // TODO validate service url format
        // this is for later support for remote server inference
        LOG(INFO) << "Session GetContract end";
        return absl::OkStatus();
    }

    absl::Status Close(CalculatorContext* cc) final {
        LOG(INFO) << "Session Close";
        return absl::OkStatus();
    }
    absl::Status Open(CalculatorContext* cc) final {
        LOG(INFO) << "Session Open start";
        for (CollectionItemId id = cc->Inputs().BeginId();
             id < cc->Inputs().EndId(); ++id) {
            if (!cc->Inputs().Get(id).Header().IsEmpty()) {
                cc->Outputs().Get(id).SetHeader(cc->Inputs().Get(id).Header());
            }
        }
        if (cc->OutputSidePackets().NumEntries() != 0) {
            for (CollectionItemId id = cc->InputSidePackets().BeginId();
                 id < cc->InputSidePackets().EndId(); ++id) {
                cc->OutputSidePackets().Get(id).Set(cc->InputSidePackets().Get(id));
            }
        }
        cc->SetOffset(TimestampDiff(0));

        const auto& options = cc->Options<ModelAPIOVMSSessionCalculatorOptions>();
        // if config is in calc then we start the server
        LOG(INFO) << "Will check if we want to start server";
        if (!options.server_config().empty()) {
            LOG(INFO) << "Will start new server";
            OVMS_ServerNew(&cserver);
            OVMS_ServerSettingsNew(&_serverSettings);
            OVMS_ModelsSettingsNew(&_modelsSettings);
            OVMS_ModelsSettingsSetConfigPath(_modelsSettings, options.server_config().c_str());
            LOG(INFO) << "state config file:" << options.server_config();
            OVMS_ServerSettingsSetLogLevel(_serverSettings, OVMS_LOG_DEBUG);
            bool isServerReady = false;
            ASSERT_CAPI_STATUS_NULL(OVMS_ServerReady(cserver, &isServerReady));
            if (!isServerReady) {
                REPORT_CAPI_STATUS_NULL(OVMS_ServerStartFromConfigurationFile(cserver, _serverSettings, _modelsSettings));
            }
            while (!isServerReady) {
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
                ASSERT_CAPI_STATUS_NULL(OVMS_ServerReady(cserver, &isServerReady));
            }
            LOG(INFO) << "Ensured server is ready";
        }

        const std::string& servableName = options.servable_name();
        const std::string& servableVersionStr = options.servable_version();
        auto servableVersionOpt = stou32(servableVersionStr);
        // 0 means default
        uint32_t servableVersion = servableVersionOpt.value_or(0);
        auto session = std::make_shared<OVMSInferenceAdapter>(servableName, servableVersion);
        try {
            session->loadModel(nullptr, UNUSED_OV_CORE, "UNUSED", {});
        } catch (const std::exception& e) {
            LOG(INFO) << "Catched exception with message: " << e.what();
            RET_CHECK(false);
        } catch (...) {
            LOG(INFO) << "Catched unknown exception";
            RET_CHECK(false);
        }

        LOG(INFO) << "Session create adapter";
        cc->OutputSidePackets().Tag(SESSION_TAG.c_str()).Set(MakePacket<std::shared_ptr<InferenceAdapter>>(session));
        LOG(INFO) << "Session Open end";
        return absl::OkStatus();
    }

    absl::Status Process(CalculatorContext* cc) final {
        LOG(INFO) << "Session Process";
        return absl::OkStatus();
    }
};

REGISTER_CALCULATOR(ModelAPISessionCalculator);
}  // namespace mediapipe
