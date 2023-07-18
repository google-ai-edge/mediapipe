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

#include <adapters/inference_adapter.h>  // TODO fix path  model_api/model_api/cpp/adapters/include/adapters/inference_adapter.h
#include <openvino/core/shape.hpp>
#include <openvino/openvino.hpp>

#include "ovms.h"  // NOLINT
#include "stringutils.hpp"
#include "tfs_frontend/tfs_utils.hpp"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/port/canonical_errors.h"
#include "mediapipe/calculators/ovms/modelapiovmsinferencecalculator.pb.h"
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
            OVMS_StatusGetCode(err, &code);                                                 \
            OVMS_StatusGetDetails(err, &msg);                                               \
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
const std::string TFTENSOR_TAG{"TENSOR"};
const std::string TFTENSORS_TAG{"TENSORS"};

static tensorflow::Tensor convertOVTensor2TFTensor(const ov::Tensor& t) {
    using tensorflow::Tensor;
    using tensorflow::TensorShape;
    auto datatype = ovms::getPrecisionAsDataType(ovms::ovElementTypeToOvmsPrecision(t.get_element_type()));
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
    auto datatype = ovms::ovmsPrecisionToIE2Precision(ovms::TFSPrecisionToOvmsPrecision(t.dtype()));
    ov::Shape shape;
    for (const auto& dim : t.shape()) {
        shape.emplace_back(dim.size);
    }
    ov::Tensor result(datatype, shape, data);
    return result;
}

class ModelAPISideFeedCalculator : public CalculatorBase {
    std::shared_ptr<::InferenceAdapter> session{nullptr};
    std::unordered_map<std::string, std::string> outputNameToTag;
    std::vector<std::string> input_order_list;
    std::vector<std::string> output_order_list;
public:
    static absl::Status GetContract(CalculatorContract* cc) {
        LOG(INFO) << "Main GetContract start";
        RET_CHECK(!cc->Inputs().GetTags().empty());
        RET_CHECK(!cc->Outputs().GetTags().empty());
        for (const std::string& tag : cc->Inputs().GetTags()) {
            // could be replaced with absl::StartsWith when migrated to MP
            if (ovms::startsWith(tag, OVTENSORS_TAG)) {
                LOG(INFO) << "setting input tag:" << tag << " to OVTensor";
                cc->Inputs().Tag(tag).Set<std::vector<ov::Tensor>>();
            } else if (ovms::startsWith(tag, OVTENSOR_TAG)) {
                LOG(INFO) << "setting input tag:" << tag << " to OVTensor";
                cc->Inputs().Tag(tag).Set<ov::Tensor>();
            } else if (ovms::startsWith(tag, TFTENSORS_TAG)) {
                LOG(INFO) << "setting input tag:" << tag << " to TFTensor";
                cc->Inputs().Tag(tag).Set<std::vector<tensorflow::Tensor>>();
            } else if (ovms::startsWith(tag, TFTENSOR_TAG)) {
                LOG(INFO) << "setting input tag:" << tag << " to TFTensor";
                cc->Inputs().Tag(tag).Set<tensorflow::Tensor>();
            } else {
                // TODO decide which will be easier to migrating later
                // using OV tensor by default will be more performant
                // but harder to migrate
                /*
                cc->Inputs().Tag(tag).Set<tensorflow::Tensor>();
                */
                LOG(INFO) << "setting input tag:" << tag << " to OVTensor";
                cc->Inputs().Tag(tag).Set<ov::Tensor>();
            }
        }
        for (const std::string& tag : cc->Outputs().GetTags()) {
            if (ovms::startsWith(tag, OVTENSORS_TAG)) {
                LOG(INFO) << "setting input tag:" << tag << " to std::vector<ov::Tensor>";
                cc->Outputs().Tag(tag).Set<std::vector<ov::Tensor>>();
            } else if (ovms::startsWith(tag, OVTENSOR_TAG)) {
                LOG(INFO) << "setting input tag:" << tag << " to OVTensor";
                cc->Outputs().Tag(tag).Set<ov::Tensor>();
            } else if (ovms::startsWith(tag, TFTENSORS_TAG)) {
                LOG(INFO) << "setting input tag:" << tag << " to TFTensor";
                cc->Outputs().Tag(tag).Set<std::vector<tensorflow::Tensor>>();
            } else if (ovms::startsWith(tag, TFTENSOR_TAG)) {
                LOG(INFO) << "setting input tag:" << tag << " to TFTensor";
                cc->Outputs().Tag(tag).Set<tensorflow::Tensor>();
            } else {
                // TODO decide which will be easier to migrating later
                // using OV tensor by default will be more performant
                // but harder to migrate
                /*    
                cc->Outputs().Tag(tag).Set<tensorflow::Tensor>();
                */
                LOG(INFO) << "setting output tag:" << tag << " to OVTensor";
                cc->Outputs().Tag(tag).Set<ov::Tensor>();
            }
        }
        cc->InputSidePackets().Tag(SESSION_TAG.c_str()).Set<std::shared_ptr<::InferenceAdapter>>();
        LOG(INFO) << "Main GetContract end";
        return absl::OkStatus();
    }

    absl::Status Close(CalculatorContext* cc) final {
        LOG(INFO) << "Main Close";
        return absl::OkStatus();
    }
    absl::Status Open(CalculatorContext* cc) final {
        LOG(INFO) << "Main Open start";
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
        const auto& options = cc->Options<ModelAPIInferenceCalculatorOptions>();
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
        LOG(INFO) << "Main Open end";
        return absl::OkStatus();
    }

    absl::Status Process(CalculatorContext* cc) final {
        LOG(INFO) << "Main process start";
        if (cc->Inputs().NumEntries() == 0) {
            return tool::StatusStop();
        }
        /////////////////////
        // PREPARE INPUT MAP
        /////////////////////

        const auto& options = cc->Options<ModelAPIInferenceCalculatorOptions>();
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

            if (ovms::startsWith(tag, OVTENSORS_TAG)) {
                auto& packet = cc->Inputs().Tag(tag).Get<std::vector<ov::Tensor>>();
                
                if ( packet.size() > 1 && input_order_list.size() <= 1)
                {
                    LOG(INFO) << "input_order_list not set in options for multiple inputs.";
                    RET_CHECK(false);
                }
                if (this->input_order_list.size() > 0){
                    for (int i = 0; i < this->input_order_list.size(); i++)
                    {
                        auto& tensor = packet[i];
                        input[this->input_order_list[i]] = tensor;
                    }
                } else if (packet.size() == 1)  {
                    input[realInputName] = packet[0];
                }
            } else if (ovms::startsWith(tag, OVTENSOR_TAG)) {
                auto& packet = cc->Inputs().Tag(tag).Get<ov::Tensor>();
                input[realInputName] = packet;
#if 0
                ov::Tensor input_tensor(packet);
                const float* input_tensor_access = reinterpret_cast<float*>(input_tensor.data());
                std::stringstream ss;
                ss << "ModelAPICalculator received tensor: [ ";
                for (int x = 0; x < 10; ++x) {
                    ss << input_tensor_access[x] << " ";
                }
                ss << " ] timestamp: " << cc->InputTimestamp().DebugString() << endl;
                LOG(INFO) << ss.str();
#endif
            } else if (ovms::startsWith(tag, TFTENSOR_TAG)) {
                auto& packet = cc->Inputs().Tag(tag).Get<tensorflow::Tensor>();
                input[realInputName] = convertTFTensor2OVTensor(packet);
            } else {
                /*
                auto& packet = cc->Inputs().Tag(tag).Get<tensorflow::Tensor>();
                input[realInputName] = convertTFTensor2OVTensor(packet);
                */
                auto& packet = cc->Inputs().Tag(tag).Get<ov::Tensor>();
                input[realInputName] = packet;
            }
        }
        //////////////////
        //  INFERENCE
        //////////////////
        try {
            output = session->infer(input);
        } catch (const std::exception& e) {
            LOG(INFO) << "Catched exception from session infer():" << e.what();
            RET_CHECK(false);
        } catch (...) {
            LOG(INFO) << "Catched unknown exception from session infer()";
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
            if (ovms::startsWith(tag, OVTENSORS_TAG)) {
                auto tensors = std::make_unique<std::vector<ov::Tensor>>();
                if ( output.size() > 1 && this->output_order_list.size() <= 1)
                {
                    LOG(INFO) << "output_order_list not set in options for multiple outputs.";
                    RET_CHECK(false);
                }
                if (this->output_order_list.size() > 0) {
                    for (int i = 0; i < this->output_order_list.size(); i++)
                    {
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
            }else if (ovms::startsWith(tag, OVTENSOR_TAG)) {
                cc->Outputs().Tag(tag).Add(
                    new ov::Tensor(tensorIt->second),
                    cc->InputTimestamp());
            } else if (ovms::startsWith(tag, TFTENSOR_TAG)) {
                cc->Outputs().Tag(tag).Add(
                    new tensorflow::Tensor(convertOVTensor2TFTensor(tensorIt->second)),
                    cc->InputTimestamp());
            } else {
                /*
                cc->Outputs().Tag(tag).Add(
                    new tensorflow::Tensor(convertOVTensor2TFTensor(tensorIt->second)),
                    cc->InputTimestamp());
                    */
                cc->Outputs().Tag(tag).Add(
                    new ov::Tensor(tensorIt->second),
                    cc->InputTimestamp());
            }
        }
        LOG(INFO) << "Main process end";
        return absl::OkStatus();
    }
};

REGISTER_CALCULATOR(ModelAPISideFeedCalculator);
}  // namespace mediapipe
