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

#include "model_infer_request_image_calculator.h"

namespace mediapipe {

    absl::Status ModelInferRequestImageCalculator::GetContract(
            CalculatorContract *cc) {
        LOG(INFO) << "ModelInferRequestImageCalculator::GetContract()";
        cc->Inputs().Tag("REQUEST").Set<const KFSRequest *>();
        cc->Outputs().Tag("IMAGE").Set<cv::Mat>();

        return absl::OkStatus();
    }

    absl::Status ModelInferRequestImageCalculator::Open(CalculatorContext *cc) {
        LOG(INFO) << "ModelInferRequestImageCalculator::Open()";
        return absl::OkStatus();
    }

    absl::Status ModelInferRequestImageCalculator::GetiProcess(
            CalculatorContext *cc) {
        LOG(INFO) << "ModelInferRequestImageCalculator::GetiProcess()";
        const KFSRequest *request =
                cc->Inputs().Tag("REQUEST").Get<const KFSRequest *>();

        LOG(INFO) << "KFSRequest for model " << request->model_name();
        auto data = request->raw_input_contents().Get(0);
        std::vector<char> image_data(data.begin(), data.end());
        auto out = cv::imdecode(image_data, 1);
        cv::cvtColor(out, out, cv::COLOR_BGR2RGB);

        // crop ROI if its passed
        if (request->parameters().find("width") != request->parameters().end()) {
            auto roi_x = (int) request->parameters().at("x").int64_param();
            auto roi_y = (int) request->parameters().at("y").int64_param();
            auto roi_width = (int) request->parameters().at("width").int64_param();
            auto roi_height = (int) request->parameters().at("height").int64_param();
            if (roi_width > 0 && roi_height > 0) {
                auto roi = cv::Rect(roi_x, roi_y, roi_width, roi_height);
                out = out(roi).clone();
            }
        }

        cc->Outputs().Tag("IMAGE").AddPacket(MakePacket<cv::Mat>(out).At(cc->InputTimestamp()));
        return absl::OkStatus();
    }

    absl::Status ModelInferRequestImageCalculator::Close(CalculatorContext *cc) {
        LOG(INFO) << "ModelInferRequestImageCalculator::Close()";
        return absl::OkStatus();
    }

    REGISTER_CALCULATOR(ModelInferRequestImageCalculator);

}  // namespace mediapipe
