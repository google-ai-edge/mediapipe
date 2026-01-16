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

#include "mediapipe/util/tracking/tone_models.h"

#include <cmath>

#include "absl/log/absl_check.h"
#include "absl/strings/str_format.h"

namespace mediapipe {

LogDomainLUTImpl::LogDomainLUTImpl() {
  log_lut_.resize(256);
  const float alpha = 0.05f;

  // Mapping: log(1 + x) = y \in [0, log(256.0)].
  for (int k = 0; k < 256; ++k) {
    log_lut_[k] = std::log(1.0f + alpha * k);
  }

  exp_lut_.resize(kExpBins);
  max_log_value_ = std::log(1.0f + alpha * 255.0f) * 1.001;
  inv_max_log_value_ = 1.0f / max_log_value_;

  const float denom = (1.0f / (kExpBins - 2));
  // Inverse operation: exp(y) - 1.0 = x \in [0, 255].
  for (int k = 0; k < kExpBins; ++k) {
    exp_lut_[k] =
        (std::exp(k * denom * max_log_value_) - 1.0f) * (1.0f / alpha);
  }
}

template <class Model, class Adapter>
void ToneModelMethods<Model, Adapter>::MapImage(const Model& model,
                                                bool log_domain,
                                                bool normalized_model,
                                                const cv::Mat& input,
                                                cv::Mat* output) {
  ABSL_CHECK(output != nullptr);

  const int out_channels = output->channels();
  ABSL_CHECK_EQ(input.channels(), 3);
  ABSL_CHECK_LE(out_channels, 3);
  ABSL_CHECK_EQ(input.rows, output->rows);
  ABSL_CHECK_EQ(input.cols, output->cols);

  float norm_scale =
      normalized_model
          ? (log_domain ? (1.0f / LogDomainLUT().MaxLogDomainValue())
                        : (1.0f / 255.0f))
          : 1.0f;

  const float inv_norm_scale = 1.0f / norm_scale;

  for (int i = 0; i < input.rows; ++i) {
    const uint8_t* input_ptr = input.ptr<uint8_t>(i);
    uint8_t* output_ptr = output->ptr<uint8_t>(i);
    for (int j = 0; j < input.cols;
         ++j, input_ptr += 3, output_ptr += out_channels) {
      Vector3_f color_vec(input_ptr[0], input_ptr[1], input_ptr[2]);
      Vector3_f mapped;
      if (log_domain) {
        mapped = Adapter::TransformColor(
            model, LogDomainLUT().MapVec(color_vec) * norm_scale);
        mapped = LogDomainLUT().UnMapVec(mapped * inv_norm_scale);
      } else {
        mapped = Adapter::TransformColor(model, color_vec * norm_scale) *
                 inv_norm_scale;
      }
      // Clamp to visible range and output.
      const Vector3_i result = RoundAndClampColor(mapped);
      for (int c = 0; c < out_channels; ++c) {
        output_ptr[c] = result[c];
      }
    }
  }
}

GainBiasModel ToneModelAdapter<GainBiasModel>::AddIdentity(
    const GainBiasModel& model) {
  GainBiasModel result = model;
  result.set_gain_c1(result.gain_c1() + 1.0f);
  result.set_gain_c2(result.gain_c2() + 1.0f);
  result.set_gain_c3(result.gain_c3() + 1.0f);
  return result;
}

GainBiasModel ToneModelAdapter<GainBiasModel>::ScaleParameters(
    const GainBiasModel& model, float scale) {
  GainBiasModel result = model;
  result.set_gain_c1(result.gain_c1() * scale);
  result.set_gain_c2(result.gain_c2() * scale);
  result.set_gain_c3(result.gain_c3() * scale);
  result.set_bias_c1(result.bias_c1() * scale);
  result.set_bias_c2(result.bias_c2() * scale);
  result.set_bias_c3(result.bias_c3() * scale);
  return result;
}

std::string ToneModelAdapter<GainBiasModel>::ToString(
    const GainBiasModel& model) {
  return absl::StrFormat("%f %f | %f %f | %f %f", model.gain_c1(),
                         model.bias_c1(), model.gain_c2(), model.bias_c2(),
                         model.gain_c3(), model.bias_c3());
}

AffineToneModel ToneModelAdapter<AffineToneModel>::AddIdentity(
    const AffineToneModel& model) {
  AffineToneModel result = model;
  result.set_g_00(result.g_00() + 1.0f);
  result.set_g_11(result.g_11() + 1.0f);
  result.set_g_22(result.g_22() + 1.0f);
  return result;
}

AffineToneModel ToneModelAdapter<AffineToneModel>::ScaleParameters(
    const AffineToneModel& model, float scale) {
  float elems[kNumParameters];
  ToPointer<float>(model, elems);
  for (int k = 0; k < kNumParameters; ++k) {
    elems[k] *= scale;
  }
  return FromPointer<float>(elems, false);
}

// Explicit instantiations.
template class MixtureToneAdapter<GainBiasModelTraits>;
template class MixtureToneAdapter<AffineToneModelTraits>;

template class ToneModelMethods<GainBiasModel, GainBiasModelAdapter>;
template class ToneModelMethods<AffineToneModel, AffineToneModelAdapter>;

}  // namespace mediapipe
