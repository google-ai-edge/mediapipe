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

#include "mediapipe/tasks/cc/genai/inference/utils/xnn_utils/sampling.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <memory>
#include <random>
#include <utility>
#include <vector>

#include "absl/memory/memory.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "mediapipe/framework/port/ret_check.h"
#include "mediapipe/framework/port/status_macros.h"
#include "mediapipe/tasks/cc/genai/inference/utils/xnn_utils/xnn_tensor.h"

namespace mediapipe::tasks::genai::xnn_utils {

absl::StatusOr<std::unique_ptr<Sampler>> Sampler::Create(Type type, int top_k,
                                                         float top_p,
                                                         float temperature,
                                                         int seed) {
  if (type == Type::kTopK || type == Type::kTopP) {
    RET_CHECK_GT(top_k, 1).SetCode(absl::StatusCode::kInvalidArgument)
        << "top_k must be > 1";
    RET_CHECK_GE(temperature, 0.0f).SetCode(absl::StatusCode::kInvalidArgument)
        << "temperature must be >= 0";
    RET_CHECK_LE(temperature, 1.0f).SetCode(absl::StatusCode::kInvalidArgument)
        << "temperature must be <= 1";
  }
  if (type == Type::kTopP) {
    RET_CHECK_GT(top_p, 0).SetCode(absl::StatusCode::kInvalidArgument)
        << "top_p must be between 0 and 1";
    RET_CHECK_LE(top_p, 1.0).SetCode(absl::StatusCode::kInvalidArgument)
        << "top_p must be between 0 and 1";
  }
  return absl::WrapUnique(new Sampler(type, top_k, top_p, temperature, seed));
}

absl::StatusOr<std::vector<std::vector<int>>> Sampler::Sample(
    const Tensor& logits) {
  if (logits.dims.size() != 3) {
    return absl::InvalidArgumentError(
        "Tensor must be (Batch, seq_len, vocab_size)");
  }

  switch (type_) {
    case Type::kGreedy:
      return SampleGreedy(logits);
    case Type::kTopK:
      return SampleTopK(logits);
    case Type::kTopP:
      return SampleTopP(logits);
    default:
      return absl::InvalidArgumentError("Unsupported sampler type");
  }
};

Sampler::Sampler(Type type, int top_k, float top_p, float temperature, int seed)
    : type_(type),
      top_k_(top_k),
      top_p_(top_p),
      temperature_(temperature),
      generator_(std::make_unique<std::mt19937>(seed)) {}

absl::StatusOr<std::vector<std::vector<int>>> Sampler::SampleGreedy(
    const Tensor& logits) {
  size_t batch_size = logits.dims[0];
  size_t draft_size = logits.dims[1];
  size_t vocab_size = logits.dims[2];

  const float* float_logits = logits.DataAs<float>();
  std::vector<std::vector<int>> outputs;
  outputs.reserve(batch_size);
  // select the token with the highest logit directly.
  for (int batch = 0; batch < batch_size; ++batch) {
    outputs.push_back(std::vector<int>());
    outputs[batch].reserve(draft_size);
    for (int draft = 0; draft < draft_size; ++draft) {
      // the index of the first logit for a single token
      int token_index =
          (batch * draft_size * vocab_size) + (draft * vocab_size);
      float max_logit = float_logits[token_index];
      int max_id = 0;
      for (int v = 0; v < vocab_size; ++v) {
        float prob = float_logits[token_index + v];
        if (prob > max_logit) {
          max_logit = prob;
          max_id = v;
        }
      }
      outputs[batch].push_back(max_id);
    }
  }
  return outputs;
};

absl::StatusOr<std::vector<std::vector<int>>> Sampler::SampleTopK(
    const Tensor& logits) {
  const size_t batch_size = logits.dims[0];
  const size_t draft_size = logits.dims[1];
  const size_t vocab_size = logits.dims[2];
  const float* flat_data = logits.DataAs<float>();

  std::vector<std::vector<int>> outputs;
  outputs.reserve(batch_size);
  for (int batch = 0; batch < batch_size; ++batch) {
    outputs.push_back(std::vector<int>());
    outputs[batch].reserve(draft_size);
    for (int draft = 0; draft < draft_size; ++draft) {
      // the index of the first logit for a single token
      int token_index =
          (batch * draft_size * vocab_size) + (draft * vocab_size);
      std::vector<std::pair<float, int>> logits_ids;
      logits_ids.reserve(vocab_size);
      for (int v = 0; v < vocab_size; ++v) {
        float logit = flat_data[token_index + v];
        logits_ids.push_back(std::make_pair(logit, v));
      }
      MP_RETURN_IF_ERROR(SelectTopK(logits_ids, top_k_));
      // No need to normalize logits here, sampler takes care of that.
      MP_RETURN_IF_ERROR(ScaledSoftmax(logits_ids, /*normalize=*/false));
      MP_ASSIGN_OR_RETURN(int sample_idx, DoSampling(logits_ids));
      outputs[batch].push_back(sample_idx);
    }
  }
  return outputs;
}

absl::StatusOr<std::vector<std::vector<int>>> Sampler::SampleTopP(
    const Tensor& logits) {
  const size_t batch_size = logits.dims[0];
  const size_t draft_size = logits.dims[1];
  const size_t vocab_size = logits.dims[2];
  const int k = top_k_ > 0 ? top_k_ : vocab_size;
  const float* flat_data = logits.DataAs<float>();

  std::vector<std::vector<int>> outputs;
  outputs.reserve(batch_size);
  for (int batch = 0; batch < batch_size; ++batch) {
    outputs.push_back(std::vector<int>());
    outputs[batch].reserve(draft_size);
    for (int draft = 0; draft < draft_size; ++draft) {
      // the index of the first logit for a single token
      int token_index =
          (batch * draft_size * vocab_size) + (draft * vocab_size);
      std::vector<std::pair<float, int>> logits_ids;
      logits_ids.reserve(vocab_size);
      for (int v = 0; v < vocab_size; ++v) {
        float logit = flat_data[token_index + v];
        logits_ids.push_back(std::make_pair(logit, v));
      }
      MP_RETURN_IF_ERROR(SelectTopK(logits_ids, k));
      MP_RETURN_IF_ERROR(ScaledSoftmax(logits_ids, /*normalize=*/true));
      MP_RETURN_IF_ERROR(SelectTopP(logits_ids, top_p_));
      MP_ASSIGN_OR_RETURN(int sample_idx, DoSampling(logits_ids));
      outputs[batch].push_back(sample_idx);
    }
  }
  return outputs;
}

absl::Status Sampler::SelectTopK(std::vector<std::pair<float, int>>& logits_ids,
                                 int k) {
  if (k > logits_ids.size()) {
    return absl::InvalidArgumentError(
        "Top k value must be smaller than the number of logits.");
  }
  std::partial_sort(
      logits_ids.begin(), logits_ids.begin() + k, logits_ids.end(),
      [](const std::pair<float, int>& a, const std::pair<float, int>& b) {
        // reverse order.
        return a.first > b.first;
      });
  logits_ids.resize(k);
  return absl::OkStatus();
}

absl::Status Sampler::SelectTopP(std::vector<std::pair<float, int>>& logits_ids,
                                 float p) {
  int included = 0;
  float prob_sum = 0.0;
  for (const auto& [logit, _] : logits_ids) {
    ++included;
    prob_sum += logit;
    if (prob_sum >= p) {
      break;
    }
  }
  if (included == 0) {
    return absl::InternalError("Bad top_p value.");
  }
  logits_ids.resize(included);
  return absl::OkStatus();
}

absl::Status Sampler::ScaledSoftmax(
    std::vector<std::pair<float, int>>& logits_ids, bool normalize) {
  float scale = 1 / (temperature_ ? temperature_ : 1.0);
  double sum = 0.0;
  float max_logit = logits_ids[0].first;
  for (int i = 0; i < logits_ids.size(); ++i) {
    const float logit = logits_ids[i].first;
    const float p = expf(scale * (logit - max_logit));
    sum += p;
    logits_ids[i].first = p;
  }
  if (normalize) {
    for (int i = 0; i < logits_ids.size(); ++i) {
      logits_ids[i].first /= sum;
    }
  }
  return absl::OkStatus();
}

absl::StatusOr<int> Sampler::DoSampling(
    std::vector<std::pair<float, int>>& logits_ids) {
  std::vector<float> probs;
  probs.reserve(logits_ids.size());
  for (const auto& [logit, _] : logits_ids) {
    probs.push_back(logit);
  }
  // Probabilities are normalized by `discrete_distribution`.
  std::discrete_distribution<> dist(probs.begin(), probs.end());
  int sample_idx = dist(*generator_);
  return logits_ids[sample_idx].second;
}

}  // namespace mediapipe::tasks::genai::xnn_utils
