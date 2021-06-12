// Copyright 2019 The MediaPipe Authors.
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

#include "mediapipe/calculators/video/tool/flow_quantizer_model.h"

#include "mediapipe/framework/port/ret_check.h"
#include "mediapipe/framework/type_map.h"

namespace mediapipe {

// Uniform normalization to 0-255.
uint8 FlowQuantizerModel::Apply(const float val, const int channel) const {
  CHECK_LT(channel, model_.min_value_size());
  const auto& min_value = model_.min_value(channel);
  const auto& max_value = model_.max_value(channel);
  QCHECK_GT(max_value, min_value);
  float res = (val - min_value) / (max_value - min_value);
  if (res < 0.0) {
    res = 0.0;
  } else if (res > 1.0) {
    res = 1.0;
  }
  return static_cast<uint8>(res * 255);
}

void FlowQuantizerModel::LoadFromProto(const QuantizerModelData& data) {
  QCHECK_GT(data.max_value(0), data.min_value(0));
  QCHECK_GT(data.max_value(1), data.min_value(1));

  model_ = data;
}

const QuantizerModelData& FlowQuantizerModel::GetModelData() const {
  return model_;
}

// Used for training, update the (min, max) range. We want to estimate the range
// of optical flow fields (Theorectically it is (-num_pixels_along_diag,
// num_pixels_along_diag).
// TODO: Taking the min and max over all training flow fields might be
// sensitive to noise. We should use more robust statistics.
void FlowQuantizerModel::AddSampleFlowField(const OpticalFlowField& flow) {
  CHECK_EQ(model_.min_value_size(), 2);
  const cv::Mat_<cv::Point2f>& flow_mat = flow.flow_data();
  for (int i = 0; i != flow.width(); ++i) {
    for (int j = 0; j != flow.height(); ++j) {
      const auto& x = flow_mat.at<cv::Point2f>(i, j).x;
      const auto& y = flow_mat.at<cv::Point2f>(i, j).y;
      // Always use the minimum and maximum value occurred in training flow
      // fields.
      model_.set_min_value(0, std::min<float>(x, model_.min_value(0)));
      model_.set_min_value(1, std::min<float>(y, model_.min_value(1)));
      model_.set_max_value(0, std::max<float>(x, model_.max_value(0)));
      model_.set_max_value(1, std::max<float>(y, model_.max_value(1)));
    }
  }
}

void FlowQuantizerModel::Init() {
  model_.Clear();
  // Initialize the values.
  for (int i = 0; i != 2; ++i) {
    model_.add_min_value(std::numeric_limits<float>::max());
    model_.add_max_value(-std::numeric_limits<float>::max());
  }
}
}  // namespace mediapipe
