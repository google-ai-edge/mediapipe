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

#include "model_infer_http_request_calculator.h"

#include <string>
#include <vector>

#include "http_payload.hpp"
#include "third_party/cpp-base64/base64.h"
using InputDataType = ovms::HttpPayload;

namespace mediapipe {

absl::Status ModelInferHttpRequestCalculator::GetContract(
    CalculatorContract *cc) {
  LOG(INFO) << "ModelInferHttpRequestCalculator::GetContract()";
  cc->Inputs().Tag("HTTP_REQUEST_PAYLOAD").Set<InputDataType>();
  cc->Outputs().Tag("IMAGE").Set<cv::Mat>();

  return absl::OkStatus();
}

absl::Status ModelInferHttpRequestCalculator::Open(CalculatorContext *cc) {
  LOG(INFO) << "ModelInferHttpRequestCalculator::Open()";
  return absl::OkStatus();
}

absl::Status ModelInferHttpRequestCalculator::GetiProcess(
    CalculatorContext *cc) {
  LOG(INFO) << "ModelInferHttpRequestCalculator::GetiProcess()";

  InputDataType payload =
      cc->Inputs().Tag("HTTP_REQUEST_PAYLOAD").Get<InputDataType>();
  rapidjson::Document doc;
  doc.Parse(payload.body.c_str());

  if (doc.HasMember("input") && doc["input"].HasMember("image")) {
    std::string base64Image = doc["input"]["image"].GetString();
    std::string decoded = base64_decode(base64Image);
    const std::vector<char> image_data(decoded.begin(), decoded.end());
    auto out = load_image(image_data);
    cv::cvtColor(out, out, cv::COLOR_BGR2RGB);

    cc->Outputs().Tag("IMAGE").AddPacket(
        MakePacket<cv::Mat>(out).At(cc->InputTimestamp()));
  }

  return absl::OkStatus();
}

absl::Status ModelInferHttpRequestCalculator::Close(CalculatorContext *cc) {
  LOG(INFO) << "ModelInferHttpRequestCalculator::Close()";
  return absl::OkStatus();
}

cv::Mat ModelInferHttpRequestCalculator::load_image(
    const std::vector<char> &image_data) {
  cv::Mat mat;
  try {
    mat = cv::imdecode(image_data, 1);
  } catch (cv::Exception &e) {
    std::string error = e.what();
    if (error.find("CV_IO_MAX_IMAGE") == std::string::npos) {
      throw;
    } else {
      throw std::runtime_error(OUT_OF_BOUNDS_ERROR);
    }
  }

  if (mat.cols < MIN_SIZE || mat.rows < MIN_SIZE) {
    throw std::runtime_error(OUT_OF_BOUNDS_ERROR);
  }

  return mat;
}

REGISTER_CALCULATOR(ModelInferHttpRequestCalculator);

}  // namespace mediapipe
