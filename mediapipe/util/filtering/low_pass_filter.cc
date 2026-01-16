// Copyright 2020 The MediaPipe Authors.
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

#include "mediapipe/util/filtering/low_pass_filter.h"

#include "absl/log/absl_log.h"

namespace mediapipe {

LowPassFilter::LowPassFilter(float alpha) : initialized_{false} {
  SetAlpha(alpha);
}

float LowPassFilter::Apply(float value) {
  float result;
  if (initialized_) {
    result = alpha_ * value + (1.0 - alpha_) * stored_value_;
  } else {
    result = value;
    initialized_ = true;
  }
  raw_value_ = value;
  stored_value_ = result;
  return result;
}

float LowPassFilter::ApplyWithAlpha(float value, float alpha) {
  SetAlpha(alpha);
  return Apply(value);
}

bool LowPassFilter::HasLastRawValue() { return initialized_; }

float LowPassFilter::LastRawValue() { return raw_value_; }

float LowPassFilter::LastValue() { return stored_value_; }

void LowPassFilter::SetAlpha(float alpha) {
  if (alpha < 0.0f || alpha > 1.0f) {
    ABSL_LOG(ERROR) << "alpha: " << alpha << " should be in [0.0, 1.0] range";
    return;
  }
  alpha_ = alpha;
}

}  // namespace mediapipe
