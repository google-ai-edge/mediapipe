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

// ODML pipeline to execute or benchmark LLM graph on device.
//
// The pipeline does the following
// 1) Read the corresponding parameters, weight and model file paths.
// 2) Construct a graph model with the setting.
// 3) Execute model inference and generate the output.
//
// This binary should only be used as an example to run the
// llm_inference_engine_c_api

#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <optional>
#include <ostream>
#include <string>
#include <vector>

#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "absl/log/absl_check.h"
#include "absl/log/absl_log.h"
#include "absl/strings/string_view.h"
#include "glog/logging.h"
#include "mediapipe/framework/deps/file_path.h"
#include "mediapipe/tasks/cc/genai/inference/c/llm_inference_engine.h"

ABSL_FLAG(std::optional<std::string>, model_path, std::nullopt,
          "Path to the tflite model file.");

ABSL_FLAG(std::optional<std::string>, cache_dir, std::nullopt,
          "Path to the cache directory.");

// Maximum number of sequence length for input + output.
ABSL_FLAG(int, max_tokens, 512,
          "Maximum number of input and output tokens. This value needs to be "
          "at least larger than the number of input tokens.");

ABSL_FLAG(std::optional<uint32_t>, topk, std::nullopt,
          "Number of tokens to sample from at each decoding step for top-k "
          "sampling.");

ABSL_FLAG(
    std::optional<float>, temperature, std::nullopt,
    "Softmax temperature. For any value less than 1/1024 (the difference "
    "between 1.0 and the next representable value for half-precision floats), "
    "the sampling op collapses to an ArgMax.");

ABSL_FLAG(std::optional<uint32_t>, random_seed, std::nullopt,
          "Random seed for sampling tokens.");

ABSL_FLAG(
    std::optional<std::string>, prompt, std::nullopt,
    "The input prompt to be fed to the model. The flag is not relevant when "
    "running the benchmark, i.e. the input_token_limit value is set.");

namespace {

// Only cout the first response
void async_callback_print(void*, LlmResponseContext* response_context) {
  std::cout << response_context->response_array[0] << std::flush;
  LlmInferenceEngine_CloseResponseContext(response_context);
}

}  // namespace

int main(int argc, char** argv) {
  google::InitGoogleLogging(argv[0]);
  absl::ParseCommandLine(argc, argv);

  ABSL_QCHECK(absl::GetFlag(FLAGS_model_path).has_value())
      << "--model_path is required.";
  const std::string model_path = absl::GetFlag(FLAGS_model_path).value();
  std::string cache_dir;
  if (absl::GetFlag(FLAGS_cache_dir).has_value()) {
    cache_dir = absl::GetFlag(FLAGS_cache_dir).value();
  } else {
    cache_dir = std::string(mediapipe::file::Dirname(model_path));
  }
  const size_t max_tokens =
      static_cast<size_t>(absl::GetFlag(FLAGS_max_tokens));

  std::optional<std::string> prompt = absl::GetFlag(FLAGS_prompt);
  if (!prompt.has_value()) {
    prompt.emplace("Write an email");
  }

  const uint32_t topk = absl::GetFlag(FLAGS_topk).value_or(1);
  const float temperature = absl::GetFlag(FLAGS_temperature).value_or(0.0f);
  const uint32_t random_seed = absl::GetFlag(FLAGS_random_seed).value_or(0);

  const LlmModelSettings model_settings = {
      .model_path = model_path.c_str(),
      .cache_dir = cache_dir.c_str(),
      .max_num_tokens = max_tokens,
  };

  const LlmSessionConfig session_config = {
      .topk = topk,
      .topp = 1.0f,
      .temperature = temperature,
      .random_seed = random_seed,
  };

  ABSL_LOG(INFO) << "Prompt: " << prompt.value();

  // Create Llm inference engine session.
  void* llm_engine = nullptr;
  char* error_msg = nullptr;
  int error_code =
      LlmInferenceEngine_CreateEngine(&model_settings, &llm_engine, &error_msg);
  if (error_code) {
    ABSL_LOG(ERROR) << "Failed to create session: " << std::string(error_msg);
    free(error_msg);
    return EXIT_FAILURE;
  }
  void* llm_engine_session = nullptr;
  error_code = LlmInferenceEngine_CreateSession(
      llm_engine, &session_config, &llm_engine_session, &error_msg);
  if (error_code) {
    ABSL_LOG(ERROR) << "Failed to create session: " << std::string(error_msg);
    free(error_msg);
    return EXIT_FAILURE;
  }

  // Create a mutable character array to hold the string

  // Use a vector as a temporary container for the string characters
  std::vector<char> char_vec(prompt.value().begin(), prompt.value().end());
  // Add the null terminator to the vector
  char_vec.push_back('\0');
  // Get a pointer to the underlying character array
  char* prompt_str = char_vec.data();

  ABSL_LOG(INFO) << "AddQueryChunk";
  error_code = LlmInferenceEngine_Session_AddQueryChunk(llm_engine_session,
                                                        prompt_str, &error_msg);
  if (error_code) {
    ABSL_LOG(ERROR) << "Failed to add query chunk: " << std::string(error_msg);
    free(error_msg);
    return EXIT_FAILURE;
  }

  // Optional to receive the number of tokens of the input.
  // ABSL_LOG(INFO) << "SizeInToken";
  // int num_tokens = LlmInferenceEngine_Session_SizeInTokens(
  //     llm_engine_session, prompt_str, /** error_msg=*/nullptr);
  // ABSL_LOG(INFO) << "Number of tokens for input prompt: " << num_tokens;

  ABSL_LOG(INFO) << "PredictAsync";
  error_code = LlmInferenceEngine_Session_PredictAsync(
      llm_engine_session,
      /*callback_context=*/nullptr, &error_msg, async_callback_print);
  if (error_code) {
    ABSL_LOG(ERROR) << "Failed to predict asyncously: "
                    << std::string(error_msg);
    free(error_msg);
    return EXIT_FAILURE;
  }

  // Optional to use the following for the sync version.
  // ABSL_LOG(INFO) << "PredictSync";
  // auto output =
  //     LlmInferenceEngine_Session_PredictSync(llm_engine_session);
  // for (int i = 0; output.response_array[0][i] != '\0'; ++i) {
  //   std::cout << output.response_array[0][i] << std::flush;
  // }
  // std::cout << std::endl;
  //
  // LlmInferenceEngine_CloseResponseContext(&output);

  ABSL_LOG(INFO) << "DeleteSession";
  LlmInferenceEngine_Session_Delete(llm_engine_session);
  LlmInferenceEngine_Engine_Delete(llm_engine);

  return EXIT_SUCCESS;
}
