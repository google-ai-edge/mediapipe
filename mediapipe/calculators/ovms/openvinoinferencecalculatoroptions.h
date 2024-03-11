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

const std::string SESSION_TAG{"SESSION"};
const std::string OVTENSOR_TAG{"OVTENSOR"};
const std::string OVTENSORS_TAG{"OVTENSORS"};
const std::string TFTENSOR_TAG{"TFTENSOR"};
const std::string TFTENSORS_TAG{"TFTENSORS"};
const std::string MPTENSOR_TAG{"TENSOR"};
const std::string MPTENSORS_TAG{"TENSORS"};
const std::string TFLITE_TENSOR_TAG{"TFLITE_TENSOR"};
const std::string TFLITE_TENSORS_TAG{"TFLITE_TENSORS"};

const std::vector<std::string> supportedTags = {SESSION_TAG, OVTENSOR_TAG, OVTENSORS_TAG, TFTENSOR_TAG, TFTENSORS_TAG, MPTENSOR_TAG, MPTENSORS_TAG, TFLITE_TENSOR_TAG, TFLITE_TENSORS_TAG};

const std::vector<std::string> supportedVectorTags = {OVTENSORS_TAG, TFTENSORS_TAG, MPTENSORS_TAG, TFLITE_TENSORS_TAG};

namespace mediapipe {

bool IsVectorTag(const std::string& tag);

bool ValidateCalculatorSettings(CalculatorContract* cc);

}  // namespace mediapipe
