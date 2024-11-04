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
#include "modelapiovmsadapter.hpp"
#if (OVMS_DUMP_TO_FILE == 1)
#include "openvinoinferencedumputils.h"
#endif
#include <iostream>
#include <map>
#include <memory>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

#include <openvino/openvino.hpp>

#include "ovms.h"  // NOLINT
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#include "mediapipe/framework/port/logging.h"
#pragma GCC diagnostic pop
// here we need to decide if we have several calculators (1 for OVMS repository, 1-N inside mediapipe)
// for the one inside OVMS repo it makes sense to reuse code from ovms lib
namespace mediapipe::ovms {
    
using std::endl;
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-function"
using InferenceOutput = std::map<std::string, ov::Tensor>;
using InferenceInput = std::map<std::string, ov::Tensor>;

#define THROW_IF_CIRCULAR_ERR(C_API_CALL) { auto* fatalErr = C_API_CALL;if (fatalErr != nullptr) {std::runtime_error exc("Getting status details circular error");throw exc; } }

#define ASSERT_CAPI_STATUS_NULL(C_API_CALL)                                                 \
    {                                                                                       \
        auto* err = C_API_CALL;                                                             \
        if (err != nullptr) {                                                               \
            uint32_t code = 0;                                                              \
            const char* msg = nullptr;                                                      \
            THROW_IF_CIRCULAR_ERR(OVMS_StatusCode(err, &code));                             \
            THROW_IF_CIRCULAR_ERR(OVMS_StatusDetails(err, &msg));                           \
            LOG(INFO) << "Error encountred in OVMSCalculator:" << msg << " code: " << code; \
            std::runtime_error exc(msg);                                                    \
            OVMS_StatusDelete(err);                                                         \
            throw exc;                                                                      \
        }                                                                                   \
    }

#define CREATE_GUARD(GUARD_NAME, CAPI_TYPE, CAPI_PTR) \
    std::unique_ptr<CAPI_TYPE, decltype(&(CAPI_TYPE##Delete))> GUARD_NAME(CAPI_PTR, &(CAPI_TYPE##Delete));

static OVMS_DataType OVPrecision2CAPI(ov::element::Type_t datatype);
static ov::element::Type_t CAPI2OVPrecision(OVMS_DataType datatype);
static ov::Tensor makeOvTensor(OVMS_DataType datatype, const int64_t* shape, size_t dimCount, const void* voutputData, size_t bytesize);

OVMSInferenceAdapter::~OVMSInferenceAdapter() {
    LOG(INFO) << "OVMSAdapter destr";
}

inline std::vector<int64_t> getShapeAcceptableByCAPI(const ov::Shape& shape) {
    if (std::any_of(shape.begin(), shape.end(), [](size_t dim) {
            return dim > std::numeric_limits<int64_t>::max();})) {
        throw std::runtime_error("Cannot use C-API with dimension size greater than int64_t max value");
    }
    return std::vector<int64_t>{shape.begin(), shape.end()};
}

void OVMSInferenceAdapter::infer(const InferenceInput& input, InferenceOutput& output) {
    /////////////////////
    // PREPARE REQUEST
    /////////////////////
    OVMS_InferenceRequest* request{nullptr};
    ASSERT_CAPI_STATUS_NULL(OVMS_InferenceRequestNew(&request, cserver, servableName.c_str(), servableVersion));
    CREATE_GUARD(requestGuard, OVMS_InferenceRequest, request);
    
    OVMS_Status* status{nullptr};
    std::vector<std::string> outputsSet;
    // PREPARE EACH INPUT
    // extract single tensor
    for (const auto& [name, input_tensor] : input) {
        const char* realName = name.c_str();
        const auto& ovShape = input_tensor.get_shape();
        std::vector<int64_t> capiShape = getShapeAcceptableByCAPI(ovShape);
        OVMS_DataType inputDataType = OVPrecision2CAPI(input_tensor.get_element_type());
        ASSERT_CAPI_STATUS_NULL(OVMS_InferenceRequestAddInput(request, realName, inputDataType, capiShape.data(), capiShape.size()));
        const uint32_t NOT_USED_NUM = 0;
        ASSERT_CAPI_STATUS_NULL(OVMS_InferenceRequestInputSetData(request,
            realName,
            reinterpret_cast<void*>(input_tensor.data()),
            input_tensor.get_byte_size(),
            OVMS_BUFFERTYPE_CPU,
            NOT_USED_NUM));
    }
    for (const auto& [name, output_tensor] : output) {
        outputsSet.emplace_back(name);
        const char* realName = name.c_str();
        const auto& ovShape = output_tensor.get_shape();
        std::vector<int64_t> capiShape = getShapeAcceptableByCAPI(ovShape);
        OVMS_DataType inputDataType = OVPrecision2CAPI(output_tensor.get_element_type());
        ASSERT_CAPI_STATUS_NULL(OVMS_InferenceRequestAddOutput(request, realName, inputDataType, capiShape.data(), capiShape.size()));
        const uint32_t NOT_USED_NUM = 0;
        ASSERT_CAPI_STATUS_NULL(OVMS_InferenceRequestOutputSetData(request,
            realName,
            reinterpret_cast<void*>(output_tensor.data()),
            output_tensor.get_byte_size(),
            OVMS_BUFFERTYPE_CPU,
            NOT_USED_NUM));

    }
#if (OVMS_DUMP_TO_FILE == 1)
    dumpOvTensorInput(input,"input");
#endif
    //////////////////
    //  INFERENCE
    //////////////////
    OVMS_InferenceResponse* response = nullptr;
    ASSERT_CAPI_STATUS_NULL(OVMS_Inference(cserver, request, &response));
    CREATE_GUARD(responseGuard, OVMS_InferenceResponse, response);
    uint32_t outputCount = 42;
    ASSERT_CAPI_STATUS_NULL(OVMS_InferenceResponseOutputCount(response, &outputCount));
    uint32_t parameterCount = 42;
    ASSERT_CAPI_STATUS_NULL(OVMS_InferenceResponseParameterCount(response, &parameterCount));
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
        if (std::find(outputsSet.begin(), outputsSet.end(), outputName) == outputsSet.end()) {
            output.emplace(outputName, std::move(makeOvTensor(datatype, shape, dimCount, voutputData, bytesize)));
        }
    }
#if (OVMS_DUMP_TO_FILE == 1)
    dumpOvTensorInput(output,"output");
#endif
}
InferenceOutput OVMSInferenceAdapter::infer(const InferenceInput& input) {
    /////////////////////
    // PREPARE REQUEST
    /////////////////////
    OVMS_InferenceRequest* request{nullptr};
    ASSERT_CAPI_STATUS_NULL(OVMS_InferenceRequestNew(&request, cserver, servableName.c_str(), servableVersion));
    CREATE_GUARD(requestGuard, OVMS_InferenceRequest, request);
    
    InferenceOutput output;
    OVMS_Status* status{nullptr};
    // PREPARE EACH INPUT
    for (const auto& [name, input_tensor] : input) {
        const char* realName = name.c_str();
        const auto& ovShape = input_tensor.get_shape();
        std::vector<int64_t> capiShape = getShapeAcceptableByCAPI(ovShape);
        OVMS_DataType inputDataType = OVPrecision2CAPI(input_tensor.get_element_type());
        ASSERT_CAPI_STATUS_NULL(OVMS_InferenceRequestAddInput(request, realName, inputDataType, capiShape.data(), capiShape.size()));
        const uint32_t NOT_USED_NUM = 0;
        ASSERT_CAPI_STATUS_NULL(OVMS_InferenceRequestInputSetData(request,
            realName,
            reinterpret_cast<void*>(input_tensor.data()),
            input_tensor.get_byte_size(),
            OVMS_BUFFERTYPE_CPU,
            NOT_USED_NUM));
    }
#if (OVMS_DUMP_TO_FILE == 1)
    dumpOvTensorInput(input,"input");
#endif
    //////////////////
    //  INFERENCE
    //////////////////
    OVMS_InferenceResponse* response = nullptr;
    status = OVMS_Inference(cserver, request, &response);
    if (nullptr != status) {
        uint32_t code = 0;
        const char* msg = nullptr;
        THROW_IF_CIRCULAR_ERR(OVMS_StatusCode(status, &code));
        THROW_IF_CIRCULAR_ERR(OVMS_StatusDetails(status, &msg));
        std::stringstream ss;
        ss << "Inference in OVMSAdapter failed: ";
        ss << msg << " code: " << code;
        LOG(INFO) << ss.str();
        OVMS_StatusDelete(status);
        return output;
    }
    CREATE_GUARD(responseGuard, OVMS_InferenceResponse, response);
    uint32_t outputCount = 42;
    ASSERT_CAPI_STATUS_NULL(OVMS_InferenceResponseOutputCount(response, &outputCount));
    uint32_t parameterCount = 42;
    ASSERT_CAPI_STATUS_NULL(OVMS_InferenceResponseParameterCount(response, &parameterCount));
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
        output.emplace(outputName, std::move(makeOvTensor(datatype, shape, dimCount, voutputData, bytesize)));
    }
#if (OVMS_DUMP_TO_FILE == 1)
    dumpOvTensorInput(output,"output");
#endif
    return output;
}

void OVMSInferenceAdapter::loadModel(const std::shared_ptr<const ov::Model>& model, ov::Core& core,
    const std::string& device, const ov::AnyMap& compilationConfig, size_t max_num_requests) {
    // no need to load but we need to extract metadata
    OVMS_ServableMetadata* servableMetadata = nullptr;
    ASSERT_CAPI_STATUS_NULL(OVMS_GetServableMetadata(cserver, servableName.c_str(), servableVersion, &servableMetadata));
    CREATE_GUARD(metadataGuard, OVMS_ServableMetadata, servableMetadata);
    uint32_t inputCount = 0;
    uint32_t outputCount = 0;
    ASSERT_CAPI_STATUS_NULL(OVMS_ServableMetadataInputCount(servableMetadata, &inputCount));
    ASSERT_CAPI_STATUS_NULL(OVMS_ServableMetadataOutputCount(servableMetadata, &outputCount));

    uint32_t id = 0;
    OVMS_DataType datatype = (OVMS_DataType)199;
    int64_t* shapeMin{nullptr};
    int64_t* shapeMax{nullptr};
    size_t dimCount = 42;
    const char* tensorName{nullptr};
    for (id = 0; id < inputCount; ++id) {
        ASSERT_CAPI_STATUS_NULL(OVMS_ServableMetadataInput(servableMetadata, id, &tensorName, &datatype, &dimCount, &shapeMin, &shapeMax));
        inputNames.emplace_back(tensorName);
        shape_min_max_t inputMinMax;
        for (size_t i = 0; i < dimCount; ++i) {
            inputMinMax.first.emplace_back(shapeMin[i]);
            inputMinMax.second.emplace_back(shapeMax[i]);
        }
        this->inShapesMinMaxes.insert({tensorName, std::move(inputMinMax)});
        this->inputDatatypes.insert({tensorName, CAPI2OVPrecision(datatype)});
    }
    for (id = 0; id < outputCount; ++id) {
        ASSERT_CAPI_STATUS_NULL(OVMS_ServableMetadataOutput(servableMetadata, id, &tensorName, &datatype, &dimCount, &shapeMin, &shapeMax));
        outputNames.emplace_back(tensorName);
        shape_min_max_t outputMinMax;
        for (size_t i = 0; i < dimCount; ++i) {
            outputMinMax.first.emplace_back(shapeMin[i]);
            outputMinMax.second.emplace_back(shapeMax[i]);
        }
        this->outShapesMinMaxes.insert({tensorName, std::move(outputMinMax)});
        this->outputDatatypes.insert({tensorName, CAPI2OVPrecision(datatype)});
    }
    const ov::AnyMap* servableMetadataRtInfo;
    ASSERT_CAPI_STATUS_NULL(OVMS_ServableMetadataInfo(servableMetadata, reinterpret_cast<const void**>(&servableMetadataRtInfo)));
    this->modelConfig = *servableMetadataRtInfo;
}

ov::element::Type_t OVMSInferenceAdapter::getInputDatatype(const std::string& inputName) const {
    return inputDatatypes.at(inputName);
}

ov::element::Type_t OVMSInferenceAdapter::getOutputDatatype(const std::string& outputName) const {
    return outputDatatypes.at(outputName);
}

ov::PartialShape OVMSInferenceAdapter::getInputShape(const std::string& inputName) const {
    auto it = inShapesMinMaxes.find(inputName);
    if (it == inShapesMinMaxes.end()) {
        LOG(INFO) << "Could not find input:" << inputName;
        throw std::runtime_error(std::string("Adapter could not find input:") + inputName);
    }

    ov::PartialShape ovShape;
    const auto& [minBorder, maxBorder] = it->second;
    ovShape.reserve(minBorder.size());
    for (size_t i = 0; i < minBorder.size(); ++i) {
        ovShape.emplace_back(ov::Dimension{minBorder[i], maxBorder[i]});
    }
    return ovShape;
}
ov::PartialShape OVMSInferenceAdapter::getOutputShape(const std::string& outputName) const {
    auto it = outShapesMinMaxes.find(outputName);
    if (it == outShapesMinMaxes.end()) {
        LOG(INFO) << "Could not find output:" << outputName;
        throw std::runtime_error(std::string("Adapter could not find output:") + outputName);
    }

    ov::PartialShape ovShape;
    const auto& [minBorder, maxBorder] = it->second;
    ovShape.reserve(minBorder.size());
    for (size_t i = 0; i < minBorder.size(); ++i) {
        ovShape.emplace_back(ov::Dimension{minBorder[i], maxBorder[i]});
    }
    return ovShape;
}

std::vector<std::string> OVMSInferenceAdapter::getInputNames() const { return inputNames; }

std::vector<std::string> OVMSInferenceAdapter::getOutputNames() const { return outputNames; }

const ov::AnyMap& OVMSInferenceAdapter::getModelConfig() const {
    return modelConfig;
}

void OVMSInferenceAdapter::inferAsync(const InferenceInput& input, CallbackData callback_args) {
    throw std::runtime_error("Model_api async calls not implemented exception.");
}

void OVMSInferenceAdapter::setCallback(std::function<void(ov::InferRequest, CallbackData)> callback) {
    throw std::runtime_error("Model_api async calls not implemented exception.");
}

bool OVMSInferenceAdapter::isReady() {
    throw std::runtime_error("Model_api async calls not implemented exception.");
    return false;
}

void OVMSInferenceAdapter::awaitAll() {
    throw std::runtime_error("Model_api async calls not implemented exception.");
}

void OVMSInferenceAdapter::awaitAny() {
    throw std::runtime_error("Model_api async calls not implemented exception.");
}

size_t OVMSInferenceAdapter::getNumAsyncExecutors() const {
    throw std::runtime_error("Model_api async calls not implemented exception.");
    return 0;
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
        {ov::element::Type_t::dynamic, OVMS_DATATYPE_DYNAMIC},
        {ov::element::Type_t::string, OVMS_DATATYPE_STRING}
        
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

static ov::Tensor makeOvTensor(OVMS_DataType datatype, const int64_t* shape, size_t dimCount, const void* voutputData, size_t bytesize) {
    ov::Shape ovShape;
    for (size_t i = 0; i < dimCount; ++i) {
        ovShape.push_back(shape[i]);
    }
    // here we make copy of underlying OVMS repsonse tensor
    ov::Tensor output(CAPI2OVPrecision(datatype), ovShape);
    std::memcpy(output.data(), voutputData, bytesize);
    return output;
}

#pragma GCC diagnostic pop
}  // namespace mediapipe::ovms
