// Copyright 2026 The MediaPipe Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <algorithm>
#include <numeric>
#include <utility>
#include <vector>

#include "absl/log/absl_log.h"
#include "mediapipe/calculators/tflite/tflite_model_metadata.pb.h"
#include "mediapipe/calculators/yolo/auto_nms_calculator.pb.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/formats/detection.pb.h"
#include "mediapipe/framework/formats/location_data.pb.h"

namespace mediapipe {

namespace {
constexpr char kDetectionsTag[]    = "DETECTIONS";
constexpr char kModelMetadataTag[] = "MODEL_METADATA";
constexpr int  kEndToEndFeatureDim = 6;

float ComputeIoU(const LocationData::RelativeBoundingBox& a,
                 const LocationData::RelativeBoundingBox& b) {
  const float x_inter = std::max(
      0.0f, std::min(a.xmin() + a.width(), b.xmin() + b.width()) -
                std::max(a.xmin(), b.xmin()));
  const float y_inter = std::max(
      0.0f, std::min(a.ymin() + a.height(), b.ymin() + b.height()) -
                std::max(a.ymin(), b.ymin()));
  const float inter = x_inter * y_inter;
  const float uni =
      a.width() * a.height() + b.width() * b.height() - inter;
  return uni > 0.0f ? inter / uni : 0.0f;
}

std::vector<Detection> GreedyNms(const std::vector<Detection>& dets,
                                  float iou_threshold) {
  if (dets.empty()) return {};
  std::vector<int> order(dets.size());
  std::iota(order.begin(), order.end(), 0);
  std::sort(order.begin(), order.end(), [&](int a, int b) {
    return dets[a].score(0) > dets[b].score(0);
  });
  std::vector<bool> suppressed(dets.size(), false);
  std::vector<Detection> result;
  for (int i = 0; i < static_cast<int>(order.size()); ++i) {
    if (suppressed[order[i]]) continue;
    result.push_back(dets[order[i]]);
    const auto& bb_i =
        dets[order[i]].location_data().relative_bounding_box();
    for (int j = i + 1; j < static_cast<int>(order.size()); ++j) {
      if (suppressed[order[j]]) continue;
      if (ComputeIoU(bb_i,
                     dets[order[j]].location_data().relative_bounding_box()) >
          iou_threshold) {
        suppressed[order[j]] = true;
      }
    }
  }
  return result;
}

bool MetadataIndicatesEndToEnd(const TfLiteModelMetadata& meta) {
  if (meta.outputs_size() == 0) return false;
  for (int d : meta.outputs(0).shape())
    if (d == kEndToEndFeatureDim) return true;
  return false;
}
}  // namespace

class AutoNmsCalculator : public CalculatorBase {
 public:
  static absl::Status GetContract(CalculatorContract* cc) {
    cc->Inputs().Tag(kDetectionsTag).Set<std::vector<Detection>>();
    cc->Outputs().Tag(kDetectionsTag).Set<std::vector<Detection>>();
    if (cc->InputSidePackets().HasTag(kModelMetadataTag)) {
      cc->InputSidePackets()
          .Tag(kModelMetadataTag)
          .Set<TfLiteModelMetadata>();
    }
    return absl::OkStatus();
  }

  absl::Status Open(CalculatorContext* cc) override {
    const auto& opts = cc->Options<AutoNmsCalculatorOptions>();
    iou_threshold_ = opts.iou_threshold();

    if (opts.postprocess_mode() == AutoNmsCalculatorOptions::SKIP_NMS) {
      skip_nms_ = true;
    } else if (opts.postprocess_mode() ==
               AutoNmsCalculatorOptions::APPLY_NMS) {
      skip_nms_ = false;
    } else if (cc->InputSidePackets().HasTag(kModelMetadataTag) &&
               !cc->InputSidePackets()
                    .Tag(kModelMetadataTag)
                    .IsEmpty()) {
      const auto& meta =
          cc->InputSidePackets()
              .Tag(kModelMetadataTag)
              .Get<TfLiteModelMetadata>();
      skip_nms_ = MetadataIndicatesEndToEnd(meta);
    } else {
      ABSL_LOG(WARNING)
          << "AutoNmsCalculator: MODEL_METADATA not connected, "
             "defaulting to running NMS.";
      skip_nms_ = false;
    }
    return absl::OkStatus();
  }

  absl::Status Process(CalculatorContext* cc) override {
    if (cc->Inputs().Tag(kDetectionsTag).IsEmpty()) return absl::OkStatus();
    const auto& input =
        cc->Inputs().Tag(kDetectionsTag).Get<std::vector<Detection>>();

    auto output = std::make_unique<std::vector<Detection>>(
        skip_nms_ ? input : GreedyNms(input, iou_threshold_));
    cc->Outputs()
        .Tag(kDetectionsTag)
        .Add(output.release(), cc->InputTimestamp());
    return absl::OkStatus();
  }

 private:
  bool  skip_nms_      = false;
  float iou_threshold_ = 0.45f;
};

REGISTER_CALCULATOR(AutoNmsCalculator);

}  // namespace mediapipe
