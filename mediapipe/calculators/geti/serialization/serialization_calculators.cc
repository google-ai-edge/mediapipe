/**
 *  INTEL CONFIDENTIAL
 *
 *  Copyright (C) 2023 Intel Corporation
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

#include "serialization_calculators.h"

#include <nlohmann/json_fwd.hpp>

#include "../inference/kserve.h"
#include "../inference/utils.h"
#include "nlohmann/json.hpp"
#include "result_serialization.h"
#include "../utils/data_structures.h"
#include "utils/ocv_common.hpp"

namespace mediapipe {

absl::Status SerializationCalculator::GetContract(CalculatorContract *cc) {
  LOG(INFO) << "SerializationCalculator::GetContract()";
  cc->Inputs().Tag("INFERENCE_RESULT").Set<geti::InferenceResult>().Optional();
  cc->Inputs().Tag("RESULT").Set<geti::InferenceResult>().Optional();

  cc->Inputs().Tag("REQUEST").Set<const KFSRequest *>();
  cc->Outputs().Tag("RESPONSE").Set<KFSResponse *>();

  return absl::OkStatus();
}

absl::Status SerializationCalculator::Open(CalculatorContext *cc) {
  LOG(INFO) << "SerializationCalculator::Open()";
  return absl::OkStatus();
}

absl::Status SerializationCalculator::GetiProcess(CalculatorContext *cc) {
  LOG(INFO) << "SerializationCalculator::GetiProcess()";
  std::string input_tag =
      geti::get_input_tag("INFERENCE_RESULT", {"RESULT"}, cc);
  auto result = cc->Inputs().Tag(input_tag).Get<geti::InferenceResult>();

  const KFSRequest *request =
      cc->Inputs().Tag("REQUEST").Get<const KFSRequest *>();
  LOG(INFO) << "KFSRequest for model " << request->model_name();

  bool include_xai = false;
  if (request->parameters().find("include_xai") != request->parameters().end())
    include_xai = request->parameters().at("include_xai").bool_param();

  bool label_only = false;
  if (request->parameters().find("label_only") != request->parameters().end())
    label_only = request->parameters().at("label_only").bool_param();

  int roi_x = 0;
  int roi_y = 0;
  if (request->parameters().find("x") != request->parameters().end()) {
    roi_x = (int)request->parameters().at("x").int64_param();
    roi_y = (int)request->parameters().at("y").int64_param();
  }

  auto response = std::make_unique<inference::ModelInferResponse>();
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

  auto param = inference::InferParameter();
  param.mutable_string_param()->assign(data.dump());
  response->mutable_parameters()->insert({"predictions", param});
  cc->Outputs()
      .Tag("RESPONSE")
      .AddPacket(MakePacket<KFSResponse *>(response.release())
                     .At(cc->InputTimestamp()));
  return absl::OkStatus();
}
absl::Status SerializationCalculator::Close(CalculatorContext *cc) {
  LOG(INFO) << "SerializationCalculator::Close()";
  return absl::OkStatus();
}

REGISTER_CALCULATOR(SerializationCalculator);
REGISTER_CALCULATOR(DetectionSerializationCalculator);
REGISTER_CALCULATOR(DetectionClassificationSerializationCalculator);
REGISTER_CALCULATOR(DetectionSegmentationSerializationCalculator);
REGISTER_CALCULATOR(RotatedDetectionSerializationCalculator);
REGISTER_CALCULATOR(ClassificationSerializationCalculator);
REGISTER_CALCULATOR(SegmentationSerializationCalculator);
REGISTER_CALCULATOR(AnomalySerializationCalculator);

}  // namespace mediapipe
