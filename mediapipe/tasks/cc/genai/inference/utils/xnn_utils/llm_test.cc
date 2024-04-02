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

#include "mediapipe/tasks/cc/genai/inference/utils/xnn_utils/llm.h"

#include <cstddef>
#include <initializer_list>
#include <limits>
#include <memory>
#include <optional>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include "absl/flags/flag.h"
#include "absl/log/absl_check.h"
#include "absl/log/absl_log.h"
#include "absl/status/statusor.h"
#include "absl/strings/match.h"
#include "absl/strings/string_view.h"
#include "mediapipe/framework/port/benchmark.h"
#include "mediapipe/framework/port/gmock.h"
#include "mediapipe/framework/port/gtest.h"
#include "mediapipe/framework/port/status_matchers.h"
#include "mediapipe/tasks/cc/genai/inference/proto/llm_params.pb.h"
#include "mediapipe/tasks/cc/genai/inference/utils/llm_utils/well_known_models.h"
#include "mediapipe/tasks/cc/genai/inference/utils/xnn_utils/benchmark_weight_accessor.h"
#include "mediapipe/tasks/cc/genai/inference/utils/xnn_utils/falcon.h"
#include "mediapipe/tasks/cc/genai/inference/utils/xnn_utils/graph_builder.h"
#include "mediapipe/tasks/cc/genai/inference/utils/xnn_utils/llm_weights.h"
#include "mediapipe/tasks/cc/genai/inference/utils/xnn_utils/phi.h"
#include "mediapipe/tasks/cc/genai/inference/utils/xnn_utils/stablelm.h"
#include "mediapipe/tasks/cc/genai/inference/utils/xnn_utils/xnn_tensor.h"
#include "xnnpack.h"  // from @XNNPACK

ABSL_FLAG(
    std::string, benchmark_method, "decode",
    "The method to benchmark the latency, can be either 'decode', 'encode'.");

ABSL_FLAG(std::string, model_type, "GEMMA_2B",
          "The type of model to benchmark, e.g. GEMMA_2B, FALCON_RW_1B");

ABSL_FLAG(int, num_threads, 4, "The number of threads to use");

namespace mediapipe::tasks::genai::xnn_utils {
namespace {

constexpr ::absl::string_view kXnnProfileCsvFile{
#if __ANDROID__
    "/data/local/tmp/xnn_profile.csv"
#else
    "/tmp/xnn_profile.csv"
#endif
};

std::unique_ptr<RuntimeConfigs> GetRunTimeConfigsForBenchmark() {
  auto runtime_config = std::make_unique<RuntimeConfigs>();
  runtime_config->xnn_num_threads = absl::GetFlag(FLAGS_num_threads);
  runtime_config->xnn_profile = false;
  runtime_config->xnn_profile_csv = std::string(kXnnProfileCsvFile);
  return runtime_config;
}

std::pair<std::unique_ptr<xnn_utils::LlmBuilder>, LlmParams>
GetLlmBuilderAndParamsForBenchmark(size_t seq_size) {
  auto model_type_string = absl::GetFlag(FLAGS_model_type);
  if (absl::EqualsIgnoreCase(model_type_string, "FALCON_RW_1B")) {
    LlmParams params =
        LlmParams::FromLLMParametersProto(llm_utils::GetFalconRW1BParams());
    params.seq_size_T = seq_size;
    params.enable_kv_cache = true;
    params.enable_dynamic_shape = true;
    return {std::make_unique<FalconRW1BBuilder>(
                params, GetRunTimeConfigsForBenchmark()),
            params};
  } else if (absl::EqualsIgnoreCase(model_type_string, "GEMMA_2B")) {
    LlmParams params =
        LlmParams::FromLLMParametersProto(llm_utils::GetGemma2BParams());
    params.seq_size_T = seq_size;
    params.enable_kv_cache = true;
    params.enable_dynamic_shape = true;
    return {
        std::make_unique<LlmBuilder>(params, GetRunTimeConfigsForBenchmark()),
        params};
  } else if (absl::EqualsIgnoreCase(model_type_string, "STABLELM_4E1T_3B")) {
    LlmParams params =
        LlmParams::FromLLMParametersProto(llm_utils::GetStablelm4E1T3BParams());
    params.seq_size_T = seq_size;
    params.enable_kv_cache = true;
    params.enable_dynamic_shape = true;
    return {std::make_unique<Stablelm4E1T3BBuilder>(
                params, GetRunTimeConfigsForBenchmark()),
            params};
  } else if (absl::EqualsIgnoreCase(model_type_string, "PHI_2")) {
    LlmParams params =
        LlmParams::FromLLMParametersProto(llm_utils::GetPhi2Params());
    params.seq_size_T = seq_size;
    params.enable_kv_cache = true;
    params.enable_dynamic_shape = true;
    return {
        std::make_unique<Phi2Builder>(params, GetRunTimeConfigsForBenchmark()),
        params};
  }

  ABSL_LOG(FATAL) << "Unsupported model type: " << model_type_string;
  return {nullptr, LlmParams()};
}

// Benchmark for the decoding latency.
void RunBenchmarkDecode(Llm& llm, benchmark::State& state) {
  std::vector<int> token_ids;
  int num_token_processed = 0;
  for (auto s : state) {
    MP_ASSERT_OK(llm.GetNextToken(&token_ids));
    num_token_processed++;
    if (llm.TotalTokenSize() >= llm.GetLlmParams().seq_size_T) {
      state.PauseTiming();
      ABSL_CHECK_OK(llm.InitInputTokens({0}));
      state.ResumeTiming();
    }
  }
  state.SetItemsProcessed(num_token_processed);
}

// Benchmark for the encoding latency.
void RunBenchmarkEncode(Llm& llm, benchmark::State& state) {
  std::vector<int> input_tokens(state.range(0), 0);
  int num_token_processed = 0;
  for (auto s : state) {
    MP_ASSERT_OK(llm.InitInputTokens(input_tokens));
    num_token_processed += state.range(0);
  }
  state.SetItemsProcessed(num_token_processed);
}

// Benchmark for the decoding/encoding latency (depending the value of the
// flag: FLAGS_benchmark_method).
void RunBenchmark(Llm& llm, benchmark::State& state) {
  const std::string& benchmark_method = absl::GetFlag(FLAGS_benchmark_method);
  if (benchmark_method == "decode") {
    RunBenchmarkDecode(llm, state);
  } else if (benchmark_method == "encode") {
    RunBenchmarkEncode(llm, state);
  } else {
    ABSL_LOG(FATAL) << "The value of flag benchamrk_method should be either "
                       "'decode' or 'encode', but got: "
                    << benchmark_method;
  }
}

class BenchmarkLlmWeightsLoader : public LlmWeightsLoader {
 public:
  BenchmarkLlmWeightsLoader(const LlmParams& params, xnn_datatype datatype,
                            std::optional<int> seed = std::nullopt)
      : LlmWeightsLoader(nullptr, params) {
    weight_accessor_ =
        std::make_unique<BenchmarkWeightAccessor>(datatype, seed);
  }
};

class BenchmarkLlmMixedInt48WeightsLoader : public LlmWeightsLoader {
 public:
  explicit BenchmarkLlmMixedInt48WeightsLoader(
      const LlmParams& params, std::optional<int> seed = std::nullopt)
      : LlmWeightsLoader(nullptr, params) {
    weight_accessor_ =
        std::make_unique<BenchmarkMixedInt48WeightAccessor>(seed);
  }
};

}  // namespace

// Benchmark LLM model specified by --model_type flag (QC8 weights, all
// default optimization)
void BM_Llm_QCINT8(benchmark::State& state) {
  auto [builder, params] = GetLlmBuilderAndParamsForBenchmark(state.range(0));
  auto weights_loader =
      std::make_unique<BenchmarkLlmWeightsLoader>(params, xnn_datatype_qcint8);

  MP_ASSERT_OK_AND_ASSIGN(
      auto llm, Llm::CreateLlm(std::move(weights_loader), std::move(builder)));
  MP_ASSERT_OK(llm->InitInputTokens({0}));

  RunBenchmark(*llm, state);
}

// Benchmark LLM model specified by --model_type flag (Mixed 4/8-bit weights,
// all default optimization)
void BM_Llm_Mixed_INT48(benchmark::State& state) {
  auto [builder, params] = GetLlmBuilderAndParamsForBenchmark(state.range(0));
  auto weights_loader =
      std::make_unique<BenchmarkLlmMixedInt48WeightsLoader>(params);

  MP_ASSERT_OK_AND_ASSIGN(
      auto llm, Llm::CreateLlm(std::move(weights_loader), std::move(builder)));
  MP_ASSERT_OK(llm->InitInputTokens({0}));

  RunBenchmark(*llm, state);
}

// Run benchmark for three different cache sizes: 64, 512, 1024.
BENCHMARK(BM_Llm_QCINT8)->Arg(64)->Arg(512)->Arg(1024);
BENCHMARK(BM_Llm_Mixed_INT48)->Arg(64)->Arg(512)->Arg(1024);

}  // namespace mediapipe::tasks::genai::xnn_utils
