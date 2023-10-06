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
#include <sstream>
#include <unordered_map>

#include <openvino/openvino.hpp>

#include "../ovms.h"           // NOLINT
#include "../stringutils.hpp"  // TODO dispose

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/port/canonical_errors.h"
#pragma GCC diagnostic pop
#include "src/mediapipe_calculators/ovmscalculator.pb.h"
// here we need to decide if we have several calculators (1 for OVMS repository, 1-N inside mediapipe)
// for the one inside OVMS repo it makes sense to reuse code from ovms lib
namespace mediapipe {
using std::endl;

namespace {
#define ASSERT_CAPI_STATUS_NULL(C_API_CALL)                                                 \
    {                                                                                       \
        auto* err = C_API_CALL;                                                             \
        if (err != nullptr) {                                                               \
            uint32_t code = 0;                                                              \
            const char* msg = nullptr;                                                      \
            OVMS_StatusCode(err, &code);                                                    \
            OVMS_StatusDetails(err, &msg);                                                  \
            LOG(INFO) << "Error encountred in OVMSCalculator:" << msg << " code: " << code; \
            OVMS_StatusDelete(err);                                                         \
            RET_CHECK(err == nullptr);                                                      \
        }                                                                                   \
    }
#define CREATE_GUARD(GUARD_NAME, CAPI_TYPE, CAPI_PTR) \
    std::unique_ptr<CAPI_TYPE, decltype(&(CAPI_TYPE##Delete))> GUARD_NAME(CAPI_PTR, &(CAPI_TYPE##Delete));

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-function"
static ov::element::Type_t CAPI2OVPrecision(OVMS_DataType datatype) {
    static std::unordered_map<OVMS_DataType, ov::element::Type_t> precisionMap{
        {OVMS_DATATYPE_FP64, ov::element::Type_t::f64},
        {OVMS_DATATYPE_FP32, ov::element::Type_t::f32},
        {OVMS_DATATYPE_FP16, ov::element::Type_t::f16},
        {OVMS_DATATYPE_I64, ov::element::Type_t::i64},
        {OVMS_DATATYPE_I32, ov::element::Type_t::i32},
        {OVMS_DATATYPE_I16, ov::element::Type_t::i16},
        {OVMS_DATATYPE_I8, ov::element::Type_t::i8},
        {OVMS_DATATYPE_I4, ov::element::Type_t::i4},
        {OVMS_DATATYPE_U64, ov::element::Type_t::u64},
        {OVMS_DATATYPE_U32, ov::element::Type_t::u32},
        {OVMS_DATATYPE_U16, ov::element::Type_t::u16},
        {OVMS_DATATYPE_U8, ov::element::Type_t::u8},
        {OVMS_DATATYPE_U4, ov::element::Type_t::u4},
        {OVMS_DATATYPE_U1, ov::element::Type_t::u1},
        {OVMS_DATATYPE_BOOL, ov::element::Type_t::boolean},
        {OVMS_DATATYPE_BF16, ov::element::Type_t::bf16},
        {OVMS_DATATYPE_UNDEFINED, ov::element::Type_t::undefined},
        {OVMS_DATATYPE_DYNAMIC, ov::element::Type_t::dynamic}
        //    {OVMS_DATATYPE_MIXED, ov::element::Type_t::MIXED},
        //    {OVMS_DATATYPE_Q78, ov::element::Type_t::Q78},
        //    {OVMS_DATATYPE_BIN, ov::element::Type_t::BIN},
        //    {OVMS_DATATYPE_CUSTOM, ov::element::Type_t::CUSTOM
    };
    auto it = precisionMap.find(datatype);
    if (it == precisionMap.end()) {
        return ov::element::Type_t::undefined;
    }
    return it->second;
}

static OVMS_DataType OVPrecision2CAPI(ov::element::Type_t datatype) {
    static std::unordered_map<ov::element::Type_t, OVMS_DataType> precisionMap{
        {ov::element::Type_t::f64, OVMS_DATATYPE_FP64},
        {ov::element::Type_t::f32, OVMS_DATATYPE_FP32},
        {ov::element::Type_t::f16, OVMS_DATATYPE_FP16},
        {ov::element::Type_t::i64, OVMS_DATATYPE_I64},
        {ov::element::Type_t::i32, OVMS_DATATYPE_I32},
        {ov::element::Type_t::i16, OVMS_DATATYPE_I16},
        {ov::element::Type_t::i8, OVMS_DATATYPE_I8},
        {ov::element::Type_t::i4, OVMS_DATATYPE_I4},
        {ov::element::Type_t::u64, OVMS_DATATYPE_U64},
        {ov::element::Type_t::u32, OVMS_DATATYPE_U32},
        {ov::element::Type_t::u16, OVMS_DATATYPE_U16},
        {ov::element::Type_t::u8, OVMS_DATATYPE_U8},
        {ov::element::Type_t::u4, OVMS_DATATYPE_U4},
        {ov::element::Type_t::u1, OVMS_DATATYPE_U1},
        {ov::element::Type_t::boolean, OVMS_DATATYPE_BOOL},
        {ov::element::Type_t::bf16, OVMS_DATATYPE_BF16},
        {ov::element::Type_t::undefined, OVMS_DATATYPE_UNDEFINED},
        {ov::element::Type_t::dynamic, OVMS_DATATYPE_DYNAMIC}
        //    {ov::element::Type_t::, OVMS_DATATYPE_MIXEDMIXED},
        //    {ov::element::Type_t::, OVMS_DATATYPE_Q78Q78},
        //    {ov::element::Type_t::, OVMS_DATATYPE_BINBIN},
        //    {ov::element::Type_t::, OVMS_DATATYPE_CUSTOMCUSTOM
    };
    auto it = precisionMap.find(datatype);
    if (it == precisionMap.end()) {
        return OVMS_DATATYPE_UNDEFINED;
    }
    return it->second;
}

static ov::Tensor* makeOvTensor(OVMS_DataType datatype, const int64_t* shape, uint32_t dimCount, const void* voutputData, size_t bytesize) {
    ov::Shape ovShape;
    for (size_t i = 0; i < dimCount; ++i) {
        ovShape.push_back(shape[i]);
    }
    // here we make copy of underlying OVMS repsonse tensor
    ov::Tensor* output = new ov::Tensor(CAPI2OVPrecision(datatype), ovShape);
    std::memcpy(output->data(), voutputData, bytesize);
    return output;
}
static ov::Tensor makeOvTensorO(OVMS_DataType datatype, const int64_t* shape, uint32_t dimCount, const void* voutputData, size_t bytesize) {
    ov::Shape ovShape;
    for (size_t i = 0; i < dimCount; ++i) {
        ovShape.push_back(shape[i]);
    }
    // here we make copy of underlying OVMS repsonse tensor
    ov::Tensor output(CAPI2OVPrecision(datatype), ovShape);
    std::memcpy(output.data(), voutputData, bytesize);
    return output;
}
}  // namespace

class OVMSOVCalculator : public CalculatorBase {
    OVMS_Server* cserver{nullptr};
    OVMS_ServerSettings* _serverSettings{nullptr};
    OVMS_ModelsSettings* _modelsSettings{nullptr};
    std::unordered_map<std::string, std::string> outputNameToTag;

public:
    static absl::Status GetContract(CalculatorContract* cc) {
        RET_CHECK(!cc->Inputs().GetTags().empty());
        RET_CHECK(!cc->Outputs().GetTags().empty());
        for (const std::string& tag : cc->Inputs().GetTags()) {
            cc->Inputs().Tag(tag).Set<ov::Tensor>();
        }
        for (const std::string& tag : cc->Outputs().GetTags()) {
            cc->Outputs().Tag(tag).Set<ov::Tensor>();
        }
        const auto& options = cc->Options<OVMSCalculatorOptions>();
        RET_CHECK(!options.servable_name().empty());
        // TODO validate version from string
        // TODO validate service url format
        RET_CHECK(options.config_path().empty() ||
                  options.service_url().empty());
        // TODO validate tag_to_tensor maps so that key fulfill regex
        return absl::OkStatus();
    }

    absl::Status Close(CalculatorContext* cc) final {
        const auto& options = cc->Options<OVMSCalculatorOptions>();
        if (!options.config_path().empty()) {
            OVMS_ModelsSettingsDelete(_modelsSettings);
            OVMS_ServerSettingsDelete(_serverSettings);
            // Close is called on input node and output node in initial pipeline
            // Commented out since for now this happens twice in 2 nodes graph. Server will close
            // OVMS_ServerDelete(cserver); TODO this should happen onlif graph is used once
            // moreover we may need several ovms calculators use in graph each providing its own model? how to handle then different model inputs, as wel as config?
        }
        return absl::OkStatus();
    }
    absl::Status Open(CalculatorContext* cc) final {
        for (CollectionItemId id = cc->Inputs().BeginId(); id < cc->Inputs().EndId(); ++id) {
            if (!cc->Inputs().Get(id).Header().IsEmpty()) {
                cc->Outputs().Get(id).SetHeader(cc->Inputs().Get(id).Header());
            }
        }
        if (cc->OutputSidePackets().NumEntries() != 0) {
            for (CollectionItemId id = cc->InputSidePackets().BeginId(); id < cc->InputSidePackets().EndId(); ++id) {
                cc->OutputSidePackets().Get(id).Set(cc->InputSidePackets().Get(id));
            }
        }
        cc->SetOffset(TimestampDiff(0));

        const auto& options = cc->Options<OVMSCalculatorOptions>();
        OVMS_ServerNew(&cserver);
        if (!options.config_path().empty()) {
            OVMS_ServerSettingsNew(&_serverSettings);
            OVMS_ModelsSettingsNew(&_modelsSettings);
            OVMS_ModelsSettingsSetConfigPath(_modelsSettings, options.config_path().c_str());
            OVMS_ServerSettingsSetLogLevel(_serverSettings, OVMS_LOG_DEBUG);
            OVMS_ServerStartFromConfigurationFile(cserver, _serverSettings, _modelsSettings);
        }
        for (const auto& [key, value] : options.tag_to_output_tensor_names()) {
            outputNameToTag[value] = key;
        }
        return absl::OkStatus();
    }

    absl::Status Process(CalculatorContext* cc) final {
        const auto& options = cc->Options<OVMSCalculatorOptions>();
        /////////////////////
        // PREPARE REQUEST
        /////////////////////
        OVMS_InferenceRequest* request{nullptr};
        auto servableVersionOpt = ovms::stou32(options.servable_version().c_str());
        uint64_t servableVersion = servableVersionOpt.value_or(0);
        ASSERT_CAPI_STATUS_NULL(OVMS_InferenceRequestNew(&request, cserver, options.servable_name().c_str(), servableVersion));
        CREATE_GUARD(requestGuard, OVMS_InferenceRequest, request);

        // PREPARE EACH INPUT
        // extract single tensor
        const auto inputTagInputMap = options.tag_to_input_tensor_names();
        const auto inputTagOutputMap = options.tag_to_output_tensor_names();
        for (const std::string& tag : cc->Inputs().GetTags()) {
            // TODO validate existence of tag key in map
            const char* realInputName = inputTagInputMap.at(tag).c_str();

            auto& packet = cc->Inputs().Tag(tag).Get<ov::Tensor>();
            ov::Tensor input_tensor(packet);
            const float* input_tensor_access = reinterpret_cast<float*>(input_tensor.data());
            std::stringstream ss;
            ss << "Calculator received tensor: [ ";
            for (int x = 0; x < 10; ++x) {
                ss << input_tensor_access[x] << " ";
            }
            ss << " ] timestamp: " << cc->InputTimestamp().DebugString();
            LOG(INFO) << ss.str();
            const auto& ovInputShape = input_tensor.get_shape();
            std::vector<int64_t> inputShape(ovInputShape.begin(), ovInputShape.end());  // TODO ensure ov tensors shapes conversions return error in all calcs
            OVMS_DataType inputDataType = OVPrecision2CAPI(input_tensor.get_element_type());
            ASSERT_CAPI_STATUS_NULL(OVMS_InferenceRequestAddInput(request, realInputName, inputDataType, inputShape.data(), inputShape.size()));
            const uint32_t notUsedNum = 0;
            // TODO handle hardcoded buffertype, notUsedNum additional options? side packets?
            ASSERT_CAPI_STATUS_NULL(OVMS_InferenceRequestInputSetData(request,
                realInputName,
                reinterpret_cast<void*>(input_tensor.data()),
                input_tensor.get_byte_size(),
                OVMS_BUFFERTYPE_CPU,
                notUsedNum));
        }
        //////////////////
        //  INFERENCE
        //////////////////
        OVMS_InferenceResponse* response = nullptr;
        ASSERT_CAPI_STATUS_NULL(OVMS_Inference(cserver, request, &response));
        CREATE_GUARD(responseGuard, OVMS_InferenceResponse, response);
        // verify GetOutputCount
        uint32_t outputCount = 42;
        ASSERT_CAPI_STATUS_NULL(OVMS_InferenceResponseOutputCount(response, &outputCount));
        RET_CHECK(outputCount == cc->Outputs().GetTags().size());
        uint32_t parameterCount = 42;
        ASSERT_CAPI_STATUS_NULL(OVMS_InferenceResponseParameterCount(response, &parameterCount));
        // TODO handle output filtering. Graph definition could suggest
        // that we are not interested in all outputs from OVMS Inference
        const void* voutputData;
        size_t bytesize = 42;
        OVMS_DataType datatype = (OVMS_DataType)199;
        const int64_t* shape{nullptr};
        size_t dimCount = 42;
        OVMS_BufferType bufferType = (OVMS_BufferType)199;
        uint32_t deviceId = 42;
        const char* outputName{nullptr};
        for (size_t i = 0; i < outputCount; ++i) {
            ASSERT_CAPI_STATUS_NULL(OVMS_InferenceResponseOutput(response, i, &outputName, &datatype, &shape, &dimCount, &voutputData, &bytesize, &bufferType, &deviceId));
            ov::Tensor* outOvTensor = makeOvTensor(datatype, shape, dimCount, voutputData, bytesize);
            cc->Outputs().Tag(outputNameToTag.at(outputName)).Add(outOvTensor, cc->InputTimestamp());
        }
        return absl::OkStatus();
    }
};
#pragma GCC diagnostic pop
REGISTER_CALCULATOR(OVMSOVCalculator);
}  // namespace mediapipe
