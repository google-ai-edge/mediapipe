/**
 *  INTEL CONFIDENTIAL
 *
 *  Copyright (C) 2024 Intel Corporation
 *
 *  This software and the related documents are Intel copyrighted materials, and
 * your use of them is governed by the express license under which they were
 * provided to you ("License"). Unless the License provides otherwise, you may
 * not use, modify, copy, publish, distribute, disclose or transmit this
 * software or the related documents without Intel's prior written permission.
 *
 *  This software and the related documents are provided as is, with no express
 * or implied warranties, other than those that are expressly stated in the
 * License.
 */

#include "../serialization/httpserialization_calculator.h"

#include <memory>
#include <nlohmann/json.hpp>
#include <nlohmann/json_fwd.hpp>
#include <string>

#include "http_payload.hpp"
#include "../inference/utils.h"
#include "../serialization/result_serialization.h"
#include "../utils/data_structures.h"
#include "utils/ocv_common.hpp"
using InputDataType = ovms::HttpPayload;

namespace mediapipe {

absl::Status HttpSerializationCalculator::GetContract(CalculatorContract *cc) {
  LOG(INFO) << "HttpSerializationCalculator::GetContract()";
  cc->Inputs().Tag("INFERENCE_RESULT").Set<geti::InferenceResult>().Optional();
  cc->Inputs().Tag("RESULT").Set<geti::InferenceResult>().Optional();

  cc->Inputs().Tag("HTTP_REQUEST_PAYLOAD").Set<InputDataType>();
  cc->Outputs().Tag("HTTP_RESPONSE_PAYLOAD").Set<std::string>();

  return absl::OkStatus();
}

absl::Status HttpSerializationCalculator::Open(CalculatorContext *cc) {
  LOG(INFO) << "HttpSerializationCalculator::Open()";
  return absl::OkStatus();
}

absl::Status HttpSerializationCalculator::GetiProcess(CalculatorContext *cc) {
  LOG(INFO) << "HttpSerializationCalculator::GetiProcess()";
  std::string input_tag =
      geti::get_input_tag("INFERENCE_RESULT", {"RESULT"}, cc);
  auto result = cc->Inputs().Tag(input_tag).Get<geti::InferenceResult>();

  const InputDataType http_request =
      cc->Inputs().Tag("HTTP_REQUEST_PAYLOAD").Get<InputDataType>();

  bool include_xai = false;
  // TODO check http_request for include_xai parameter

  bool label_only = false;
    // TODO check http_request for label_only parameter

  int roi_x = 0;
  int roi_y = 0;
      // TODO check http_request for x parameter

  if (!include_xai) {
    result.saliency_maps.clear();
  }

  geti::translate_inference_result_by_roi(result, roi_x,
                                          roi_y);  // Apply ROI translation
  nlohmann::json data = result;
  if (include_xai) {
    geti::filter_maps_by_prediction_prevalence(data);
  } else {
    data.erase("maps");  // Remove empty array added by serializer.
  }

  // Remove shape if only labels need to be returned
  if (label_only) {
    for (auto &prediction : data["predictions"]) {
      prediction.erase("shape");
    }
  }
  std::string response = data.dump();
  cc->Outputs()
      .Tag("HTTP_RESPONSE_PAYLOAD")
      .AddPacket(MakePacket<std::string>(response).At(cc->InputTimestamp()));
  return absl::OkStatus();
}
absl::Status HttpSerializationCalculator::Close(CalculatorContext *cc) {
  LOG(INFO) << "HttpSerializationCalculator::Close()";
  return absl::OkStatus();
}

REGISTER_CALCULATOR(HttpSerializationCalculator);

}  // namespace mediapipe
