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
#include <unordered_map>

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#include "mediapipe/framework/calculator_contract.h"
#include "mediapipe/calculators/ovms/openvinoinferencecalculator.pb.h"
#pragma GCC diagnostic pop
#include "mediapipe/calculators/ovms/openvinoinferencecalculatoroptions.h"
#include "mediapipe/calculators/ovms/openvinoinferenceutils.h"

namespace mediapipe {

static bool ValidateOrderLists(std::set<std::string> calculatorTags, const google::protobuf::RepeatedPtrField<std::string>& order_list) {
    // Get output_stream types defined in the graph
    std::vector<std::string> inputTypes;
    for (const std::string& tag : calculatorTags) {
        std::vector<std::string> tokens = tokenize(tag, ':');
        if (tokens.size() > 0) {
            std::string inputType = tokens[0];

            // Check if supported vector tag was used
            for (const auto& supportedVectorTag : supportedVectorTags) {
                if ( startsWith(inputType, supportedVectorTag)){
                    if (order_list.size() < 1)
                    {
                        LOG(ERROR) << "OpenVINOInferenceCalculator GetContract error. Order list is requiered for vector types: " << inputType;
                        return false;
                    }
                }
            }
        }
    }

    return true;
}

static bool ValidateOrderListsForNonVector(std::set<std::string> calculatorTags, const google::protobuf::RepeatedPtrField<std::string>& order_list) {
    // Get output_stream types defined in the graph
    std::vector<std::string> inputTypes;
    bool vectorTypeExists = false;
    for (const std::string& tag : calculatorTags) {
        std::vector<std::string> tokens = tokenize(tag, ':');
        if (tokens.size() > 0) {
            std::string inputType = tokens[0];
            if (IsVectorTag(inputType)){
                vectorTypeExists = true;
            }
        }
    }
    if (!vectorTypeExists && order_list.size() > 0) {
        LOG(ERROR) << "OpenVINOInferenceCalculator GetContract error. Using odrer_list for non vector type. " << order_list[0];
        return false;
    }

    return true;
}

bool IsVectorTag(const std::string& tag) {
    bool isVectorType = false;
    // Check if supported vector tag was used
    for (const auto& supportedVectorTag : supportedVectorTags) {
        if (startsWith(tag, supportedVectorTag)) {
            return true;
        }
    }

    return false;
}

static bool ValidateTagToNames(std::set<std::string> calculatorTags, const google::protobuf::Map<std::string, std::string>& tags_to_names) {
    // Get output_stream types defined in the graph
    std::vector<std::string> inputTypes;
    for (const std::string& tag : calculatorTags) {
        std::vector<std::string> tokens = tokenize(tag, ':');
        if (tokens.size() > 0) {
            inputTypes.push_back(tokens[0]);
        } else {
            inputTypes.push_back(tag);
        }
    }

    for (const auto& [key, value] : tags_to_names) {
        bool nameMatch = false;

        // Check if supported tag was used
        for (const auto& supportedTag : supportedTags) {
            if ( startsWith(key, supportedTag)){
                if (endsWith(key, "S") && !endsWith(supportedTag, "S"))
                    continue;

                // Check if used tag is defined in the input_stream
                for (const auto& graphInput : inputTypes) {
                    if (startsWith(graphInput, supportedTag)) {
                        if (endsWith(graphInput, "S") && !endsWith(supportedTag, "S"))
                            continue;
                        nameMatch = true;
                        break;
                    }
                }
            }

            // Check if empty tag used - OV:Tensor default type
            if ( tokenize(key, ':').size() == 0) {
                // Check if used tag is defined in the input_stream
                for (const auto& graphInput : inputTypes) {
                    // Check if empty tag used - OV:Tensor default type
                    if (key == graphInput) {
                        nameMatch = true;
                        break;
                    }                       
                }
            }

            // Type used in tag_to__tensor_names does match input_stream type
            if (nameMatch){
                break;
            }
        }

        // Check if no supported tag used - OV:Tensor default type
        for (const auto& graphInput : inputTypes) {
            // Full match required
            if (key == graphInput) {
                nameMatch = true;
                break;
            }
        }

        // Type used in tag_to__tensor_names does not match outputstream type
        if (!nameMatch)
        {
            LOG(ERROR) << "OpenVINOInferenceCalculator GetContract error. Stream names mismatch for tag_to__tensor_names name key " << key;
            return false;
        } 
    }

    return true;
}

static bool ValidateOptions(CalculatorContract* cc) {
    const auto& options = cc->Options<OpenVINOInferenceCalculatorOptions>();

    if (options.tag_to_output_tensor_names().size() > 0 && options.output_order_list().size() > 0) {
        LOG(ERROR) << "OpenVINOInferenceCalculator GetContract error. Use tag_to_output_tensor_names or output_order_list not both at once.";
        return false;
    }

    if (options.tag_to_input_tensor_names().size() > 0 && options.input_order_list().size() > 0) {
        LOG(ERROR) << "OpenVINOInferenceCalculator GetContract error. Use tag_to_input_tensor_names or input_order_list not both at once.";
        return false;
    }

    return true;
}

bool ValidateCalculatorSettings(CalculatorContract* cc)
{
    if (!ValidateOptions(cc))
    {
        LOG(INFO) << "OpenVINOInferenceCalculator ValidateOptions failed.";
        return false;
    }

    const auto& options = cc->Options<OpenVINOInferenceCalculatorOptions>();

    if (!ValidateOrderListsForNonVector(cc->Inputs().GetTags(), options.input_order_list())) {
        LOG(INFO) << "OpenVINOInferenceCalculator ValidateOrderListsForNonVector for inputs failed.";
        return false;
    }
    if (!ValidateOrderListsForNonVector(cc->Outputs().GetTags(), options.output_order_list())) {
        LOG(INFO) << "OpenVINOInferenceCalculator ValidateOrderListsForNonVector for outputs failed.";
        return false;
    }

    if (!ValidateOrderLists(cc->Inputs().GetTags(), options.input_order_list())) {
        LOG(INFO) << "OpenVINOInferenceCalculator ValidateOrderLists for inputs failed.";
        return false;
    }
    if (!ValidateOrderLists(cc->Outputs().GetTags(), options.output_order_list())) {
        LOG(INFO) << "OpenVINOInferenceCalculator ValidateOrderLists for outputs failed.";
        return false;
    }

    if (!ValidateTagToNames(cc->Inputs().GetTags(), options.tag_to_input_tensor_names())) {
        LOG(INFO) << "OpenVINOInferenceCalculator ValidateInputTagToNames failed.";
        return false;
    }
    if (!ValidateTagToNames(cc->Outputs().GetTags(), options.tag_to_output_tensor_names())) {
        LOG(INFO) << "OpenVINOInferenceCalculator ValidateOutputTagToNames failed.";
        return false;
    }

    LOG(INFO) << "OpenVINOInferenceCalculator ValidateCalculatorSettings passed.";
    return true;
}

}  // namespace mediapipe
