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
#include <algorithm>
#include <iostream>
#include <memory>
#include <openvino/core/type/element_type.hpp>
#include <sstream>
#include <unordered_map>

#include <adapters/inference_adapter.h>  // TODO fix path  model_api/model_api/cpp/adapters/include/adapters/inference_adapter.h
#include <openvino/core/shape.hpp>
#include <openvino/openvino.hpp>

#include "ovms.h"  // NOLINT
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#include "tensorflow/core/framework/tensor.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/port/canonical_errors.h"
#include "mediapipe/framework/formats/tensor.h"
#include "mediapipe/calculators/ovms/openvinoinferencecalculator.pb.h"
#include "tensorflow/lite/c/common.h"
#pragma GCC diagnostic pop
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wall"
#include "tensorflow/lite/interpreter.h"
#pragma GCC diagnostic pop
// here we need to decide if we have several calculators (1 for OVMS repository, 1-N inside mediapipe)
// for the one inside OVMS repo it makes sense to reuse code from ovms lib
namespace mediapipe {
using std::endl;

namespace {

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
            RET_CHECK(err == nullptr);                                                      \
        }                                                                                   \
    }
#define CREATE_GUARD(GUARD_NAME, CAPI_TYPE, CAPI_PTR) \
    std::unique_ptr<CAPI_TYPE, decltype(&(CAPI_TYPE##Delete))> GUARD_NAME(CAPI_PTR, &(CAPI_TYPE##Delete));

}  // namespace

const std::string SESSION_TAG{"SESSION"};
const std::string OVTENSOR_TAG{"OVTENSOR"};
const std::string OVTENSORS_TAG{"OVTENSORS"};
const std::string TFTENSOR_TAG{"TFTENSOR"};
const std::string TFTENSORS_TAG{"TFTENSORS"};
const std::string MPTENSOR_TAG{"TENSOR"};
const std::string MPTENSORS_TAG{"TENSORS"};
const std::string TFLITE_TENSOR_TAG{"TFLITE_TENSOR"};
const std::string TFLITE_TENSORS_TAG{"TFLITE_TENSORS"};

using TFSDataType = tensorflow::DataType;

// Function from ovms/src/string_utils.h
bool startsWith(const std::string& str, const std::string& prefix) {
    auto it = prefix.begin();
    bool sizeCheck = (str.size() >= prefix.size());
    if (!sizeCheck) {
        return false;
    }
    bool allOf = std::all_of(str.begin(),
        std::next(str.begin(), prefix.size()),
        [&it](const char& c) {
            return c == *(it++);
        });
    return allOf;
}

TFSDataType getPrecisionAsDataType(ov::element::Type_t precision) {
    static std::unordered_map<ov::element::Type_t, TFSDataType> precisionMap{
        {ov::element::Type_t::f32, TFSDataType::DT_FLOAT},
        {ov::element::Type_t::f64, TFSDataType::DT_DOUBLE},
        {ov::element::Type_t::f16, TFSDataType::DT_HALF},
        {ov::element::Type_t::i64, TFSDataType::DT_INT64},
        {ov::element::Type_t::i32, TFSDataType::DT_INT32},
        {ov::element::Type_t::i16, TFSDataType::DT_INT16},
        {ov::element::Type_t::i8, TFSDataType::DT_INT8},
        {ov::element::Type_t::u64, TFSDataType::DT_UINT64},
        {ov::element::Type_t::u16, TFSDataType::DT_UINT16},
        {ov::element::Type_t::u8, TFSDataType::DT_UINT8},
        {ov::element::Type_t::boolean, TFSDataType::DT_BOOL}
    };
    auto it = precisionMap.find(precision);
    if (it == precisionMap.end()) {
        return TFSDataType::DT_INVALID;
    }
    return it->second;
}

static Tensor::ElementType OVType2MPType(ov::element::Type_t precision) {
    static std::unordered_map<ov::element::Type_t, Tensor::ElementType> precisionMap{
//        {ov::element::Type_t::f64, Tensor::ElementType::},
        {ov::element::Type_t::f32, Tensor::ElementType::kFloat32},
        {ov::element::Type_t::f16, Tensor::ElementType::kFloat16},
//        {ov::element::Type_t::i64, Tensor::ElementType::64},
        {ov::element::Type_t::i32, Tensor::ElementType::kInt32},
//        {ov::element::Type_t::i16, Tensor::ElementType::},
        {ov::element::Type_t::i8, Tensor::ElementType::kInt8},
//        {ov::element::Type_t::u64, Tensor::ElementType::64},
//        {ov::element::Type_t::u32, Tensor::ElementType::},
//        {ov::element::Type_t::u16, Tensor::ElementType::},
        {ov::element::Type_t::u8, Tensor::ElementType::kUInt8},
        {ov::element::Type_t::boolean, Tensor::ElementType::kBool}
//        {ov::element::Type_t::, Tensor::ElementType::kChar}
    };
    auto it = precisionMap.find(precision);
    if (it == precisionMap.end()) {
        return Tensor::ElementType::kNone;
    }
    return it->second;
}

static ov::element::Type_t MPType2OVType(Tensor::ElementType precision) {
    static std::unordered_map<Tensor::ElementType, ov::element::Type_t> precisionMap{
//        Tensor::ElementType::, ov::element::Type_t::f64},
        {Tensor::ElementType::kFloat32, ov::element::Type_t::f32},
        {Tensor::ElementType::kFloat16, ov::element::Type_t::f16},
//        {Tensor::ElementType::, ov::element::Type_t::i64},
        {Tensor::ElementType::kInt32, ov::element::Type_t::i32},
//        {Tensor::ElementType::, ov::element::Type_t::i16},
        {Tensor::ElementType::kInt8, ov::element::Type_t::i8},
//        {Tensor::ElementType::, ov::element::Type_t::u64},
//        {Tensor::ElementType::, ov::element::Type_t::u32},
//        {Tensor::ElementType::, ov::element::Type_t::u16},
        {Tensor::ElementType::kUInt8, ov::element::Type_t::u8},
        {Tensor::ElementType::kBool, ov::element::Type_t::boolean}
//        {Tensor::ElementType::kChar, ov::element::Type_t::}
    };
    auto it = precisionMap.find(precision);
    if (it == precisionMap.end()) {
        return ov::element::Type_t::undefined;
    }
    return it->second;
}

static ov::Tensor convertMPTensor2OVTensor(const Tensor& inputTensor) {
    void* data;
    switch(inputTensor.element_type()) {
    case Tensor::ElementType::kFloat32:
    case Tensor::ElementType::kFloat16:
        data = reinterpret_cast<void*>(const_cast<float*>(inputTensor.GetCpuReadView().buffer<float>()));
        break;
    case Tensor::ElementType::kUInt8:
        data = reinterpret_cast<void*>(const_cast<uint8_t*>(inputTensor.GetCpuReadView().buffer<uint8_t>()));
        break;
    case Tensor::ElementType::kInt8:
        data = reinterpret_cast<void*>(const_cast<int8_t*>(inputTensor.GetCpuReadView().buffer<int8_t>()));
        break;
    case Tensor::ElementType::kInt32:
        data = reinterpret_cast<void*>(const_cast<int32_t*>(inputTensor.GetCpuReadView().buffer<int32_t>()));
        break;
    case Tensor::ElementType::kBool:
        data = reinterpret_cast<void*>(const_cast<bool*>(inputTensor.GetCpuReadView().buffer<bool>()));
        break;
    default:
        data = reinterpret_cast<void*>(const_cast<void*>(inputTensor.GetCpuReadView().buffer<void>()));
        break;
    }
    auto datatype = MPType2OVType(inputTensor.element_type());;
    if (datatype == ov::element::Type_t::undefined) {
        std::stringstream ss;
        LOG(INFO) << "Not supported precision for Mediapipe tensor deserialization";
        std::runtime_error exc("Not supported precision for Mediapipe tensor deserialization");
        throw exc;
    }
    ov::Shape shape;
    for (const auto& dim : inputTensor.shape().dims) {
        shape.emplace_back(dim);
    }
    ov::Tensor result(datatype, shape, data);
    return result;
}

static Tensor convertOVTensor2MPTensor(const ov::Tensor& inputTensor) {
    std::vector<int> rawShape;
    for (size_t i = 0; i < inputTensor.get_shape().size(); i++) {
        rawShape.emplace_back(inputTensor.get_shape()[i]);
    }
    Tensor::Shape shape{rawShape};
    auto datatype = OVType2MPType(inputTensor.get_element_type());
    if (datatype == mediapipe::Tensor::ElementType::kNone) {
        std::stringstream ss;
        LOG(INFO) << "Not supported precision for Mediapipe tensor serialization: " << inputTensor.get_element_type();
        std::runtime_error exc("Not supported precision for Mediapipe tensor serialization");
        throw exc;
    }
    Tensor outputTensor(datatype, shape);
    void* data;
    switch(inputTensor.get_element_type()) {
    case ov::element::Type_t::f32:
    case ov::element::Type_t::f16:
        data = reinterpret_cast<void*>(const_cast<float*>(outputTensor.GetCpuWriteView().buffer<float>()));
        break;
    case ov::element::Type_t::u8:
        data = reinterpret_cast<void*>(const_cast<uint8_t*>(outputTensor.GetCpuWriteView().buffer<uint8_t>()));
        break;
    case ov::element::Type_t::i8:
        data = reinterpret_cast<void*>(const_cast<int8_t*>(outputTensor.GetCpuWriteView().buffer<int8_t>()));
        break;
    case ov::element::Type_t::i32:
        data = reinterpret_cast<void*>(const_cast<int32_t*>(outputTensor.GetCpuWriteView().buffer<int32_t>()));
        break;
    case ov::element::Type_t::boolean:
        data = reinterpret_cast<void*>(const_cast<bool*>(outputTensor.GetCpuWriteView().buffer<bool>()));
        break;
    default:
        data = reinterpret_cast<void*>(const_cast<void*>(outputTensor.GetCpuWriteView().buffer<void>()));
        break;
    }
    std::memcpy(data, inputTensor.data(), inputTensor.get_byte_size());
    return outputTensor;
}

ov::element::Type_t TFSPrecisionToIE2Precision(TFSDataType precision) {
    static std::unordered_map<TFSDataType, ov::element::Type_t> precisionMap{
        {TFSDataType::DT_DOUBLE, ov::element::Type_t::f64},
        {TFSDataType::DT_FLOAT, ov::element::Type_t::f32},
        {TFSDataType::DT_HALF, ov::element::Type_t::f16},
        {TFSDataType::DT_INT64, ov::element::Type_t::i64},
        {TFSDataType::DT_INT32, ov::element::Type_t::i32},
        {TFSDataType::DT_INT16, ov::element::Type_t::i16},
        {TFSDataType::DT_INT8, ov::element::Type_t::i8},
        {TFSDataType::DT_UINT64, ov::element::Type_t::u64},
        {TFSDataType::DT_UINT32, ov::element::Type_t::u32},
        {TFSDataType::DT_UINT16, ov::element::Type_t::u16},
        {TFSDataType::DT_UINT8, ov::element::Type_t::u8},
        {TFSDataType::DT_BOOL, ov::element::Type_t::boolean},
        //    {Precision::MIXED, ov::element::Type_t::MIXED},
        //    {Precision::Q78, ov::element::Type_t::Q78},
        //    {Precision::BIN, ov::element::Type_t::BIN},
        //    {Precision::CUSTOM, ov::element::Type_t::CUSTOM
    };
    auto it = precisionMap.find(precision);
    if (it == precisionMap.end()) {
        return ov::element::Type_t::undefined;
    }
    return it->second;
}

static tensorflow::Tensor convertOVTensor2TFTensor(const ov::Tensor& t) {
    using tensorflow::Tensor;
    using tensorflow::TensorShape;
    auto datatype = getPrecisionAsDataType(t.get_element_type());
    if (datatype == TFSDataType::DT_INVALID) {
        std::stringstream ss;
        LOG(INFO) << "Not supported precision for Tensorflow tensor serialization: " << t.get_element_type();
        std::runtime_error exc("Not supported precision for Tensorflow tensor serialization");
        throw exc;
    }
    TensorShape tensorShape;
    std::vector<int64_t> rawShape;
    for (size_t i = 0; i < t.get_shape().size(); i++) {
        rawShape.emplace_back(t.get_shape()[i]);
    }
    int64_t dimsCount = rawShape.size();
    tensorflow::TensorShapeUtils::MakeShape(rawShape.data(), dimsCount, &tensorShape);
    TensorShape::BuildTensorShapeBase(rawShape, static_cast<tensorflow::TensorShapeBase<TensorShape>*>(&tensorShape));
    // here we allocate default TF CPU allocator
    tensorflow::Tensor result(datatype, tensorShape);
    void* tftensordata = result.data();
    std::memcpy(tftensordata, t.data(), t.get_byte_size());
    return result;
}

static ov::Tensor convertTFTensor2OVTensor(const tensorflow::Tensor& t) {
    void* data = t.data();
    auto datatype = TFSPrecisionToIE2Precision(t.dtype());
    if (datatype == ov::element::Type_t::undefined) {
        std::stringstream ss;
        LOG(INFO) << "Not supported precision for Tensorflow tensor deserialization: " << t.dtype();
        std::runtime_error exc("Not supported precision for Tensorflow tensor deserialization");
        throw exc;
    }
    ov::Shape shape;
    for (const auto& dim : t.shape()) {
        shape.emplace_back(dim.size);
    }
    if (ov::shape_size(shape) <= 0)
        return ov::Tensor(datatype, shape);  // OV does not allow nullptr as data
    return ov::Tensor(datatype, shape, data);
}


static ov::Tensor convertTFLiteTensor2OVTensor(const TfLiteTensor& t) {
    void* data = t.data.f; // probably works only for floats
    auto datatype = ov::element::f32;
    ov::Shape shape;
    // for some reason TfLite tensor does not have bs dim
    shape.emplace_back(1);
    // TODO: Support scalars and no data tensors with 0-dim
    for (int i = 0; i < t.dims->size; ++i) {
        shape.emplace_back(t.dims->data[i]);
    }
    ov::Tensor result(datatype, shape, data);
    return result;
}

class OpenVINOInferenceCalculator : public CalculatorBase {
    std::shared_ptr<::InferenceAdapter> session{nullptr};
    std::unordered_map<std::string, std::string> outputNameToTag;
    std::vector<std::string> input_order_list;
    std::vector<std::string> output_order_list;
    std::unique_ptr<tflite::Interpreter> interpreter_ = absl::make_unique<tflite::Interpreter>();
    bool initialized = false;

public:
    static absl::Status GetContract(CalculatorContract* cc) {
        LOG(INFO) << "OpenVINOInferenceCalculator GetContract start";
        RET_CHECK(!cc->Inputs().GetTags().empty());
        RET_CHECK(!cc->Outputs().GetTags().empty());
        RET_CHECK(cc->InputSidePackets().HasTag(SESSION_TAG));
        for (const std::string& tag : cc->Inputs().GetTags()) {
            // could be replaced with absl::StartsWith when migrated to MP
            if (startsWith(tag, OVTENSORS_TAG)) {
                LOG(INFO) << "setting input tag:" << tag << " to OVTensors";
                cc->Inputs().Tag(tag).Set<std::vector<ov::Tensor>>();
            } else if (startsWith(tag, OVTENSOR_TAG)) {
                LOG(INFO) << "setting input tag:" << tag << " to OVTensor";
                cc->Inputs().Tag(tag).Set<ov::Tensor>();
            } else if (startsWith(tag, MPTENSORS_TAG)) {
                LOG(INFO) << "setting input tag:" << tag << " to MPTensors";
                cc->Inputs().Tag(tag).Set<std::vector<Tensor>>();
            } else if (startsWith(tag, MPTENSOR_TAG)) {
                LOG(INFO) << "setting input tag:" << tag << " to MPTensor";
                cc->Inputs().Tag(tag).Set<Tensor>();
            } else if (startsWith(tag, TFTENSORS_TAG)) {
                LOG(INFO) << "setting input tag:" << tag << " to TFTensors";
                cc->Inputs().Tag(tag).Set<std::vector<tensorflow::Tensor>>();
            } else if (startsWith(tag, TFTENSOR_TAG)) {
                LOG(INFO) << "setting input tag:" << tag << " to TFTensor";
                cc->Inputs().Tag(tag).Set<tensorflow::Tensor>();
            } else if (startsWith(tag, TFLITE_TENSORS_TAG)) {
                LOG(INFO) << "setting input tag:" << tag << " to TFLITE_Tensors";
                cc->Inputs().Tag(tag).Set<std::vector<TfLiteTensor>>();
            } else if (startsWith(tag, TFLITE_TENSOR_TAG)) {
                LOG(INFO) << "setting input tag:" << tag << " to TFLITE_Tensor";
                cc->Inputs().Tag(tag).Set<TfLiteTensor>();
            } else {
                LOG(INFO) << "setting input tag:" << tag << " to OVTensor";
                cc->Inputs().Tag(tag).Set<ov::Tensor>();
            }
        }
        for (const std::string& tag : cc->Outputs().GetTags()) {
            if (startsWith(tag, OVTENSORS_TAG)) {
                LOG(INFO) << "setting input tag:" << tag << " to std::vector<ov::Tensor>";
                cc->Outputs().Tag(tag).Set<std::vector<ov::Tensor>>();
            } else if (startsWith(tag, OVTENSOR_TAG)) {
                LOG(INFO) << "setting input tag:" << tag << " to OVTensor";
                cc->Outputs().Tag(tag).Set<ov::Tensor>();
            } else if (startsWith(tag, MPTENSORS_TAG)) {
                LOG(INFO) << "setting input tag:" << tag << " to MPTensor";
                cc->Outputs().Tag(tag).Set<std::vector<Tensor>>();
            } else if (startsWith(tag, MPTENSOR_TAG)) {
                LOG(INFO) << "setting input tag:" << tag << " to MPTensor";
                cc->Outputs().Tag(tag).Set<Tensor>();
            } else if (startsWith(tag, TFTENSORS_TAG)) {
                LOG(INFO) << "setting input tag:" << tag << " to TFTensor";
                cc->Outputs().Tag(tag).Set<std::vector<tensorflow::Tensor>>();
            } else if (startsWith(tag, TFTENSOR_TAG)) {
                LOG(INFO) << "setting input tag:" << tag << " to TFTensor";
                cc->Outputs().Tag(tag).Set<tensorflow::Tensor>();
            } else if (startsWith(tag, TFLITE_TENSORS_TAG)) {
                LOG(INFO) << "setting input tag:" << tag << " to TFLITE_Tensor";
                cc->Outputs().Tag(tag).Set<std::vector<TfLiteTensor>>();
            } else if (startsWith(tag, TFLITE_TENSOR_TAG)) {
                LOG(INFO) << "setting input tag:" << tag << " to TFLITE_Tensor";
                cc->Outputs().Tag(tag).Set<TfLiteTensor>();
            } else {
                LOG(INFO) << "setting output tag:" << tag << " to OVTensor";
                cc->Outputs().Tag(tag).Set<ov::Tensor>();
            }
        }
        cc->InputSidePackets().Tag(SESSION_TAG.c_str()).Set<std::shared_ptr<::InferenceAdapter>>();
        LOG(INFO) << "OpenVINOInferenceCalculator GetContract end";
        return absl::OkStatus();
    }

    absl::Status Close(CalculatorContext* cc) final {
        LOG(INFO) << "OpenVINOInferenceCalculator Close";
        return absl::OkStatus();
    }
    absl::Status Open(CalculatorContext* cc) final {
        LOG(INFO) << "OpenVINOInferenceCalculator Open start";
        session = cc->InputSidePackets()
                      .Tag(SESSION_TAG.c_str())
                      .Get<std::shared_ptr<::InferenceAdapter>>();
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
        const auto& options = cc->Options<OpenVINOInferenceCalculatorOptions>();
        for (const auto& [key, value] : options.tag_to_output_tensor_names()) {
            outputNameToTag[value] = key;
        }

        auto& input_list = options.input_order_list();
        input_order_list.clear();
        for(int i = 0; i < input_list.size(); i++){
            input_order_list.push_back(input_list[i]);
        }
        auto& output_list = options.output_order_list();
        output_order_list.clear();
        for(int i = 0; i < output_list.size(); i++){
            output_order_list.push_back(output_list[i]);
        }

        cc->SetOffset(TimestampDiff(0));
        LOG(INFO) << "OpenVINOInferenceCalculator Open end";
        return absl::OkStatus();
    }

    absl::Status Process(CalculatorContext* cc) final {
        LOG(INFO) << "OpenVINOInferenceCalculator process start";
        if (cc->Inputs().NumEntries() == 0) {
            return tool::StatusStop();
        }
        /////////////////////
        // PREPARE INPUT MAP
        /////////////////////

        const auto& options = cc->Options<OpenVINOInferenceCalculatorOptions>();
        const auto& inputTagInputMap = options.tag_to_input_tensor_names();
        ::InferenceInput input;
        ::InferenceOutput output;
        for (const std::string& tag : cc->Inputs().GetTags()) {
            const char* realInputName{nullptr};
            auto it = inputTagInputMap.find(tag);
            if (it == inputTagInputMap.end()) {
                realInputName = tag.c_str();
            } else {
                realInputName = it->second.c_str();
            } 
#define DESERIALIZE_TENSORS(TYPE, DESERIALIZE_FUN) \
                auto& packet = cc->Inputs().Tag(tag).Get<std::vector<TYPE>>();                \
                if ( packet.size() > 1 && input_order_list.size() != packet.size()) {                 \
                    LOG(INFO) << "input_order_list not set properly in options for multiple inputs."; \
                    RET_CHECK(false);                                                                 \
                }                                                                                     \
                if (this->input_order_list.size() > 0){                                               \
                    for (size_t i = 0; i < this->input_order_list.size(); i++) {                         \
                        auto& tensor = packet[i];                                                     \
                        input[this->input_order_list[i]] = DESERIALIZE_FUN(tensor);                   \
                    }                                                                                 \
                } else if (packet.size() == 1) {                                                      \
                    input[realInputName] = DESERIALIZE_FUN(packet[0]);                                \
                }
            try {
            if (startsWith(tag, OVTENSORS_TAG)) {
                DESERIALIZE_TENSORS(ov::Tensor,);
            } else if (startsWith(tag, TFLITE_TENSORS_TAG)) {
                DESERIALIZE_TENSORS(TfLiteTensor, convertTFLiteTensor2OVTensor);
            } else if (startsWith(tag, MPTENSORS_TAG)) {
                DESERIALIZE_TENSORS(Tensor, convertMPTensor2OVTensor);
            } else if (startsWith(tag, OVTENSOR_TAG)) {
                auto& packet = cc->Inputs().Tag(tag).Get<ov::Tensor>();
                input[realInputName] = packet;
            } else if (startsWith(tag, TFLITE_TENSOR_TAG)) {
                auto& packet = cc->Inputs().Tag(tag).Get<TfLiteTensor>();
                input[realInputName] = convertTFLiteTensor2OVTensor(packet);
            } else if (startsWith(tag, MPTENSOR_TAG)) {
                auto& packet = cc->Inputs().Tag(tag).Get<Tensor>();
                input[realInputName] = convertMPTensor2OVTensor(packet);
            } else if (startsWith(tag, TFTENSOR_TAG)) {
                auto& packet = cc->Inputs().Tag(tag).Get<tensorflow::Tensor>();
                input[realInputName] = convertTFTensor2OVTensor(packet);
            } else {
                auto& packet = cc->Inputs().Tag(tag).Get<ov::Tensor>();
                input[realInputName] = packet;
            }
            } catch (const std::runtime_error& e) {
                LOG(INFO) << "Failed to deserialize tensor error:" << e.what();
                RET_CHECK(false);
            }
        }
        //////////////////
        //  INFERENCE
        //////////////////
        try {
            output = session->infer(input);
        } catch (const std::exception& e) {
            LOG(INFO) << "Caught exception from session infer():" << e.what();
            RET_CHECK(false);
        } catch (...) {
            LOG(INFO) << "Caught unknown exception from session infer()";
            RET_CHECK(false);
        }
        auto outputsCount = output.size();
        RET_CHECK(outputsCount >= cc->Outputs().GetTags().size());
        LOG(INFO) << "output tags size: " << cc->Outputs().GetTags().size();
        for (const auto& tag : cc->Outputs().GetTags()) {
            LOG(INFO) << "Processing tag: " << tag;
            std::string tensorName;
            auto it = options.tag_to_output_tensor_names().find(tag);
            if (it == options.tag_to_output_tensor_names().end()) {
                tensorName = tag;
            } else {
                tensorName = it->second;
            }
            auto tensorIt = output.find(tensorName);
            if (tensorIt == output.end()) {
                LOG(INFO) << "Could not find: " << tensorName << " in inference output";
                RET_CHECK(false);
            }
            try {
            if (startsWith(tag, OVTENSORS_TAG)) {
                LOG(INFO) << "OVMS calculator will process vector<ov::Tensor>";
                auto tensors = std::make_unique<std::vector<ov::Tensor>>();
                if ( output.size() > 1 && this->output_order_list.size() != this->output_order_list.size())
                {
                    LOG(INFO) << "output_order_list not set properly in options for multiple outputs.";
                    RET_CHECK(false);
                }
                if (this->output_order_list.size() > 0) {
                    for (size_t i = 0; i < this->output_order_list.size(); i++) {
                        tensorName = this->output_order_list[i];
                        tensorIt = output.find(tensorName);
                        if (tensorIt == output.end()) {
                            LOG(INFO) << "Could not find: " << tensorName << " in inference output";
                            RET_CHECK(false);
                        }
                        tensors->emplace_back(tensorIt->second);
                    }
                } else {
                    for (auto& [name,tensor] : output) {
                        tensors->emplace_back(tensor);
                    }
                }
                cc->Outputs().Tag(tag).Add(
                    tensors.release(),
                    cc->InputTimestamp());
                //break; // TODO FIXME order of outputs
                // no need to break since we only have one tag
                // create concatenator calc
            } else if (startsWith(tag, MPTENSORS_TAG)) {
                LOG(INFO) << "OVMS calculator will process vector<Tensor>";
                auto tensors = std::make_unique<std::vector<Tensor>>();
                if ( output.size() > 1 && this->output_order_list.size() != this->output_order_list.size())
                {
                    LOG(INFO) << "output_order_list not set properly in options for multiple outputs.";
                    RET_CHECK(false);
                }
                if (this->output_order_list.size() > 0) {
                    for (size_t i = 0; i < this->output_order_list.size(); i++) {
                        tensorName = this->output_order_list[i];
                        tensorIt = output.find(tensorName);
                        if (tensorIt == output.end()) {
                            LOG(INFO) << "Could not find: " << tensorName << " in inference output";
                            RET_CHECK(false);
                        }
                        tensors->emplace_back(convertOVTensor2MPTensor(tensorIt->second));
                    }
                } else {
                    for (auto& [name,tensor] : output) {
                        tensors->emplace_back(convertOVTensor2MPTensor(tensor));
                    }
                }
                cc->Outputs().Tag(tag).Add(
                    tensors.release(),
                    cc->InputTimestamp());
                //break; // TODO FIXME order of outputs
                // no need to break since we only have one tag
                // create concatenator calc
            } else if (startsWith(tag, TFLITE_TENSORS_TAG)) {
                // TODO FIXME use output_order_list
                LOG(INFO) << "OVMS calculator will process vector<TfLiteTensor>";
                auto outputStreamTensors = std::vector<TfLiteTensor>();
                if (!this->initialized) {
                    interpreter_->AddTensors(output.size());
                    std::vector<int> indexes(output.size());
                    std::iota(indexes.begin(), indexes.end(), 0);
                    interpreter_->SetInputs(indexes);
                    size_t tensorId = 0;
                    for (auto& [name,tensor] : output) {
                        std::vector<int> tfliteshape;
                        for (auto& d : tensor.get_shape()) {
                            tfliteshape.emplace_back(d);
                        }
                        interpreter_->SetTensorParametersReadWrite(
                                        tensorId,
                                        kTfLiteFloat32, // TODO datatype
                                        name.c_str(),
                                        tfliteshape,
                                        TfLiteQuantization());
                        ++tensorId;
                    }
                    interpreter_->AllocateTensors();
                    this->initialized = true;
                }
                size_t tensorId = 0;
                for (auto& [name,tensor] : output) {
                    const int interpreterTensorId = interpreter_->inputs()[tensorId];
                    TfLiteTensor* tflitetensor = interpreter_->tensor(interpreterTensorId);
                    void* tensor_ptr = tflitetensor->data.f;
                    std::memcpy(tensor_ptr, tensor.data(), tensor.get_byte_size());
                    outputStreamTensors.emplace_back(*tflitetensor);
                    ++tensorId;
                }
                cc->Outputs().Tag(tag).AddPacket(MakePacket<std::vector<TfLiteTensor>>(std::move(outputStreamTensors)).At( cc->InputTimestamp()));
                break;
            } else if (startsWith(tag, OVTENSOR_TAG)) {
                LOG(INFO) << "OVMS calculator will process ov::Tensor";
                cc->Outputs().Tag(tag).Add(
                    new ov::Tensor(tensorIt->second),
                    cc->InputTimestamp());
            } else if (startsWith(tag, TFTENSOR_TAG)) {
                LOG(INFO) << "OVMS calculator will process tensorflow::Tensor";
                cc->Outputs().Tag(tag).Add(
                    new tensorflow::Tensor(convertOVTensor2TFTensor(tensorIt->second)),
                    cc->InputTimestamp());
            } else if (startsWith(tag, MPTENSOR_TAG)) {
                LOG(INFO) << "OVMS calculator will process mediapipe::Tensor";
                cc->Outputs().Tag(tag).Add(
                    new Tensor(convertOVTensor2MPTensor(tensorIt->second)),
                    cc->InputTimestamp());
            } else {
                LOG(INFO) << "OVMS calculator will process ov::Tensor";
                cc->Outputs().Tag(tag).Add(
                    new ov::Tensor(tensorIt->second),
                    cc->InputTimestamp());
            }
            } catch (const std::runtime_error& e) {
                LOG(INFO) << "Failed to deserialize tensor error:" << e.what();
                RET_CHECK(false);
            }
        }
        LOG(INFO) << "OpenVINOInferenceCalculator process end";
        return absl::OkStatus();
    }
};

REGISTER_CALCULATOR(OpenVINOInferenceCalculator);
}  // namespace mediapipe
