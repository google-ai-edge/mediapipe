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
#include "openvinomodelserversessioncalculator.h"
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
#include "mediapipe/calculators/ovms/openvinomodelserversessioncalculator.pb.h"
#include "mediapipe/calculators/ovms/openvinoinferenceutils.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/port/canonical_errors.h"
#include "modelapiovmsadapter.hpp"
#pragma GCC diagnostic pop
namespace mediapipe {
using ovms::OVMSInferenceAdapter;

const std::string SESSION_TAG{"SESSION"};
ov::Core UNUSED_OV_CORE;

#define ASSERT_CIRCULAR_ERR(C_API_CALL) \
    {                                   \
        auto* fatalErr = C_API_CALL;    \
        RET_CHECK(fatalErr == nullptr); \
    }

#define ASSERT_CAPI_STATUS_NULL(C_API_CALL)                                                 \
    {                                                                                       \
        auto* err = C_API_CALL;                                                             \
        if (err != nullptr) {                                                               \
            uint32_t code = 0;                                                              \
            const char* msg = nullptr;                                                      \
            ASSERT_CIRCULAR_ERR(OVMS_StatusCode(err, &code));                               \
            ASSERT_CIRCULAR_ERR(OVMS_StatusDetails(err, &msg));                             \
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
            ASSERT_CIRCULAR_ERR(OVMS_StatusCode(err, &code));                               \
            ASSERT_CIRCULAR_ERR(OVMS_StatusDetails(err, &msg));                             \
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

class SettingsGuard {
public:
    OVMS_ServerSettings* serverSettings{nullptr};
    OVMS_ModelsSettings* modelsSettings{nullptr};
    SettingsGuard() {
        OVMS_ServerSettingsNew(&serverSettings);
        OVMS_ModelsSettingsNew(&modelsSettings);
    }
    ~SettingsGuard() {
        OVMS_ServerSettingsDelete(serverSettings);
        OVMS_ModelsSettingsDelete(modelsSettings);
    }
};

absl::Status OpenVINOModelServerSessionCalculator::GetContract(CalculatorContract* cc) {
    LOG(INFO) << "OpenVINOModelServerSessionCalculator GetContract start";
    RET_CHECK(cc->Inputs().GetTags().empty());
    RET_CHECK(cc->Outputs().GetTags().empty());
    cc->OutputSidePackets().Tag(SESSION_TAG.c_str()).Set<std::shared_ptr<::InferenceAdapter>>();
    const auto& options = cc->Options<OpenVINOModelServerSessionCalculatorOptions>();
    RET_CHECK(!options.servable_name().empty());
    
    OvmsLogLevel = StringToLogLevel(std::string(std::getenv(OvmsLogLevelEnv) == nullptr ? "" : std::getenv(OvmsLogLevelEnv)));
    LOG(INFO) << "OpenVINOModelServerSessionCalculator ovms log level setting: " << LogLevelToString(OvmsLogLevel);
    LOG(INFO) << "OpenVINOModelServerSessionCalculator GetContract end";
    return absl::OkStatus();
}

absl::Status OpenVINOModelServerSessionCalculator::Close(CalculatorContext* cc) {
    LOG(INFO) << "OpenVINOModelServerSessionCalculator Close";
    return absl::OkStatus();
}
absl::Status OpenVINOModelServerSessionCalculator::Open(CalculatorContext* cc) {
    LOG(INFO) << "OpenVINOModelServerSessionCalculator Open start";
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

    const auto& options = cc->Options<OpenVINOModelServerSessionCalculatorOptions>();
    // if config is in calc then we start the server
    LOG(INFO) << "Will check if we want to start server";
    if (!options.server_config().empty()) {
        // Lock access to server from multiple calculator instances during the model loading phase
        std::unique_lock<std::mutex> lk(OpenVINOModelServerSessionCalculator::loadingMtx);
        bool isServerReady = false;
        bool isServerLive = false;
        OVMS_ServerNew(&cserver);

        ASSERT_CAPI_STATUS_NULL(OVMS_ServerLive(cserver, &isServerLive));

        if (triedToStartOVMS) {
            RET_CHECK(isServerLive);
        } else if (!isServerLive) {
            LOG(INFO) << "Will start new server";
            triedToStartOVMS = true;
            SettingsGuard guard;
            OVMS_ServerSettingsNew(&guard.serverSettings);
            OVMS_ModelsSettingsNew(&guard.modelsSettings);
            OVMS_ModelsSettingsSetConfigPath(guard.modelsSettings, options.server_config().c_str());
            LOG(INFO) << "state config file:" << options.server_config();
            OVMS_ServerSettingsSetLogLevel(guard.serverSettings, OvmsLogLevel);

            ASSERT_CAPI_STATUS_NULL(OVMS_ServerStartFromConfigurationFile(cserver, guard.serverSettings, guard.modelsSettings));

            ASSERT_CAPI_STATUS_NULL(OVMS_ServerReady(cserver, &isServerReady));
            RET_CHECK(isServerReady);
            LOG(INFO) << "Server started";
        }
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
        LOG(INFO) << "Caught exception with message: " << e.what();
        return mediapipe::FailedPreconditionErrorBuilder(MEDIAPIPE_LOC)
             << "OpenVINOModelServerSessionCalculator failed to load the model";
    } catch (...) {
        LOG(INFO) << "Caught unknown exception";
        RET_CHECK(false);
    }

    LOG(INFO) << "OpenVINOModelServerSessionCalculator create adapter";
    cc->OutputSidePackets().Tag(SESSION_TAG.c_str()).Set(MakePacket<std::shared_ptr<InferenceAdapter>>(session));
    LOG(INFO) << "OpenVINOModelServerSessionCalculator Open end";
    return absl::OkStatus();
}

absl::Status OpenVINOModelServerSessionCalculator::Process(CalculatorContext* cc) {
    LOG(INFO) << "OpenVINOModelServerSessionCalculator Process";
    return absl::OkStatus();
}

bool OpenVINOModelServerSessionCalculator::triedToStartOVMS = false;
std::mutex OpenVINOModelServerSessionCalculator::loadingMtx;
const char* OpenVINOModelServerSessionCalculator::OvmsLogLevelEnv = "GLOG_minloglevel";
OVMS_LogLevel OpenVINOModelServerSessionCalculator::OvmsLogLevel = OVMS_LOG_INFO;

REGISTER_CALCULATOR(OpenVINOModelServerSessionCalculator);
}  // namespace mediapipe
