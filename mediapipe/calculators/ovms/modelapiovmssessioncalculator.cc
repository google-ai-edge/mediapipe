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
#include <sstream>
#include <unordered_map>

#include <openvino/openvino.hpp>

#include "ovms.h"           // NOLINT
#include "stringutils.hpp"  // TODO dispose
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/port/canonical_errors.h"
#include "modelapiovmsadapter.hpp"
#include "mediapipe/calculators/ovms/modelapiovmssessioncalculator.pb.h"
// here we need to decide if we have several calculators (1 for OVMS repository, 1-N inside mediapipe)
// for the one inside OVMS repo it makes sense to reuse code from ovms lib
namespace mediapipe {

using ovms::OVMSInferenceAdapter;
using std::endl;

const std::string SESSION_TAG{"SESSION"};
ov::Core UNUSED_OV_CORE;

class ModelAPISessionCalculator : public CalculatorBase {
    std::shared_ptr<::InferenceAdapter> adapter;
    std::unordered_map<std::string, std::string> outputNameToTag;

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
        const std::string& servableName = options.servable_name();
        const std::string& servableVersionStr = options.servable_version();
        auto servableVersionOpt = ::ovms::stou32(servableVersionStr);
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
