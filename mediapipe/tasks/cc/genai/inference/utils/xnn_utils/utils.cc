// Copyright 2024 The MediaPipe Authors.
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

#include "mediapipe/tasks/cc/genai/inference/utils/xnn_utils/utils.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <random>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/types/span.h"
#include "mediapipe/framework/port/ret_check.h"
#include "mediapipe/framework/port/status_macros.h"

namespace mediapipe::tasks::genai {
namespace xnn_utils {

std::vector<float> FillXnnRoPEWeights(size_t max_seq_len, size_t num_channels) {
  std::vector<float> out_array(max_seq_len * num_channels);
  for (size_t ch_id = 0; ch_id < num_channels / 2; ++ch_id) {
    auto timescale = std::pow(1e-4, 2.0 * ch_id / num_channels);
    for (size_t seq_id = 0; seq_id < max_seq_len; ++seq_id) {
      auto sinusoid_inp = seq_id * timescale;
      out_array[seq_id * num_channels + ch_id] = cos(sinusoid_inp);
      out_array[seq_id * num_channels + ch_id + num_channels / 2] =
          sin(sinusoid_inp);
    }
  }
  return out_array;
}

absl::StatusOr<std::vector<uint8_t>> PackInt4ToInt8(absl::Span<uint8_t> vec) {
  if (vec.empty() || vec.size() % 2 != 0) {
    return absl::InvalidArgumentError(
        "The input vector needs to be non-empty and contains even number of "
        "elements.");
  }
  std::vector<uint8_t> output;
  output.reserve(vec.size() / 2);
  for (int i = 0; i < vec.size() / 2; ++i) {
    RET_CHECK_LT(vec[i * 2 + 1], 16);
    RET_CHECK_LT(vec[i * 2], 16);
    uint8_t v = (vec[i * 2 + 1] & 0x0f) << 4;
    output.push_back(v | (vec[i * 2] & 0x0f));
  }
  return output;
}

absl::StatusOr<std::vector<uint8_t>> UnpackInt8ToInt4(
    absl::Span<uint8_t> packed_vec) {
  std::vector<uint8_t> output;
  output.reserve(packed_vec.size() * 2);
  for (int i = 0; i < packed_vec.size(); ++i) {
    uint8_t v = packed_vec[i];
    output.push_back(v & 0x0f);
    output.push_back(v >> 4);
  }
  return output;
}

absl::StatusOr<std::vector<float>> PositionEmbedding(int seq_length,
                                                     int embedding_dim,
                                                     float min_timescale,
                                                     float max_timescale) {
  return FullPositionEmbedding(seq_length, seq_length, embedding_dim,
                               min_timescale, max_timescale);
}

absl::StatusOr<std::vector<float>> FullPositionEmbedding(int input_length,
                                                         int seq_length,
                                                         int embedding_dim,
                                                         float min_timescale,
                                                         float max_timescale) {
  if (embedding_dim % 2 != 0) {
    return absl::InvalidArgumentError(absl::StrCat(
        "embedding_dim must be divided by 2. But got ", embedding_dim));
  }
  std::vector<float> embedding(seq_length * embedding_dim, 0.0f);
  float num_timescales = static_cast<float>(embedding_dim / 2);
  float log_timescale_inc = std::log(max_timescale / min_timescale) /
                            std::max(num_timescales - 1.0, 1.0);
  for (int s = 0; s < seq_length; ++s) {
    for (int i = 0; i < embedding_dim / 2; ++i) {
      float inv_timescale =
          min_timescale * std::exp(i * -1.0f * log_timescale_inc);
      int scale = s < input_length ? s : 0;
      // Sine
      embedding[s * embedding_dim + i] = std::sin(scale * inv_timescale);
      // Cosine
      embedding[s * embedding_dim + embedding_dim / 2 + i] =
          std::cos(scale * inv_timescale);
    }
  }
  return embedding;
}

}  // namespace xnn_utils
}  // namespace mediapipe::tasks::genai
