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

#include <pthread.h>

#include <cstdio>
#include <cstdlib>
#include <functional>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/log/absl_check.h"
#include "absl/log/absl_log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "mediapipe/framework/port/file_helpers.h"
#include "mediapipe/framework/port/ret_check.h"
#include "mediapipe/framework/port/status_macros.h"
#include "mediapipe/tasks/cc/genai/inference/c/llm_inference_engine.h"
#include "mediapipe/tasks/cc/genai/inference/proto/llm_params.pb.h"
#include "mediapipe/tasks/cc/genai/inference/proto/transformer_params.pb.h"
#include "mediapipe/tasks/cc/genai/inference/utils/llm_utils/memory_mapped_file.h"
#include "mediapipe/tasks/cc/genai/inference/utils/llm_utils/metadata_utils.h"
#include "mediapipe/tasks/cc/genai/inference/utils/llm_utils/model_data.h"
#include "mediapipe/tasks/cc/genai/inference/utils/llm_utils/scoped_file.h"
#include "mediapipe/tasks/cc/genai/inference/utils/xnn_utils/graph_builder.h"
#include "mediapipe/tasks/cc/genai/inference/utils/xnn_utils/llm.h"
#include "mediapipe/tasks/cc/genai/inference/utils/xnn_utils/llm_builder_factory.h"
#include "mediapipe/tasks/cc/genai/inference/utils/xnn_utils/llm_weights.h"
#include "sentencepiece/src/normalizer.h"  // from @com_google_sentencepiece
#include "sentencepiece/src/sentencepiece_processor.h"  // from @com_google_sentencepiece
#include "tensorflow/lite/model_builder.h"

namespace {

constexpr int kCheckLastKChars = 10;

struct LlmInferenceEngineCpu_Engine {
  sentencepiece::SentencePieceProcessor* tokenizer;
  sentencepiece::normalizer::Normalizer* normalizer;
  mediapipe::tasks::genai::xnn_utils::Llm* llm;
  int start_token_id;
  std::vector<std::string> stop_tokens;
  size_t max_num_tokens;
  ~LlmInferenceEngineCpu_Engine() {
    delete tokenizer;
    if (normalizer != nullptr) {
      delete normalizer;
    }
    delete llm;
  };
};

struct LlmInferenceEngineCpu_Session {
  const LlmInferenceEngineCpu_Engine* engine;
  std::string prompt;
  int max_num_output_tokens;
  int response_count;
  std::string last_10_char;
  std::string final_output;
  std::function<void(std::string)> cpu_callback;
  bool early_stop;
  pthread_t work_id;
  ~LlmInferenceEngineCpu_Session() { pthread_join(work_id, nullptr); };
};

void* next_token_function(void* args) {
  struct LlmInferenceEngineCpu_Session* cpu_session =
      (struct LlmInferenceEngineCpu_Session*)args;
  if (cpu_session->response_count++ < cpu_session->max_num_output_tokens) {
    if (cpu_session->early_stop) {
      return nullptr;
    }

    auto token_ids_per_step = std::vector<int>();
    auto status = cpu_session->engine->llm->GetNextToken(&token_ids_per_step);
    if (!status.ok()) {
      ABSL_LOG(FATAL) << "Failed to generate output: " << status;
    }

    // For future multithreading support.
    if (cpu_session->early_stop) {
      return nullptr;
    }

    if (cpu_session->response_count == cpu_session->max_num_output_tokens) {
      cpu_session->early_stop = true;
    }

    std::string token =
        cpu_session->engine->tokenizer->IdToPiece(token_ids_per_step[0]);
    if (cpu_session->engine->normalizer != nullptr) {
      token = cpu_session->engine->normalizer->Normalize(token);
    }
    cpu_session->last_10_char.append(token);

    int stop_index;
    for (const auto& stop_token : cpu_session->engine->stop_tokens) {
      stop_index = cpu_session->last_10_char.find(stop_token);
      if (stop_index != std::string::npos) {
        cpu_session->early_stop = true;
        cpu_session->last_10_char =
            cpu_session->last_10_char.substr(0, stop_index);
        break;
      }
    }

    std::string ready_char = "";
    if (cpu_session->early_stop) {
      ready_char = cpu_session->last_10_char;
    } else if (cpu_session->last_10_char.size() > kCheckLastKChars) {
      ready_char = cpu_session->last_10_char.substr(
          0, cpu_session->last_10_char.size() - kCheckLastKChars);
      cpu_session->last_10_char = cpu_session->last_10_char.substr(
          cpu_session->last_10_char.size() - kCheckLastKChars);
    }
    cpu_session->final_output.append(ready_char);

    cpu_session->cpu_callback(ready_char);

    next_token_function(args);
  }
  return nullptr;
};

void* start_llm_function(void* args) {
  struct LlmInferenceEngineCpu_Session* cpu_session =
      (struct LlmInferenceEngineCpu_Session*)args;

  std::vector<int> prompt_ids = {};

  auto status =
      cpu_session->engine->tokenizer->Encode(cpu_session->prompt, &prompt_ids);

  if (!status.ok()) {
    ABSL_LOG(FATAL) << "Failed to encode input: " << status;
  }
  prompt_ids.insert(prompt_ids.begin(), cpu_session->engine->start_token_id);

  ABSL_CHECK_OK(cpu_session->engine->llm->SeekTimeStep(0));
  ABSL_CHECK_OK(cpu_session->engine->llm->AddInputTokens({prompt_ids}));

  cpu_session->max_num_output_tokens =
      cpu_session->engine->max_num_tokens - prompt_ids.size();

  next_token_function(args);

  return nullptr;
}

absl::StatusOr<LlmInferenceEngine_Engine*>
LlmInferenceEngine_CreateEngine_Helper(const LlmModelSettings* model_settings) {
  MP_ASSIGN_OR_RETURN(auto model_file,
                      mediapipe::tasks::genai::llm_utils::ScopedFile::Open(
                          model_settings->model_path));
  MP_ASSIGN_OR_RETURN(auto model_data,
                      mediapipe::tasks::genai::llm_utils::ModelData::Create(
                          std::move(model_file)));

  if (model_settings->number_of_supported_lora_ranks != 0) {
    ABSL_LOG(FATAL) << "LoRA on CPU is not supported yet.";
  }

  auto llm_params_proto = model_data->GetLlmParameters();
  auto llm_params =
      mediapipe::tasks::genai::xnn_utils::LlmParams::FromLLMParametersProto(
          llm_params_proto);

  auto model_type = model_data->GetModelType();
  RET_CHECK(model_type) << "Failed to get model type.";

  MP_ASSIGN_OR_RETURN(auto backend,
                      model_data->ReadMetadata(
                          mediapipe::tasks::genai::llm_utils::kLlmBackendName));
  RET_CHECK_EQ(backend, "cpu");

  // Create directory for tokenizer and model cache file.
  if (model_settings->cache_dir != nullptr) {
    auto s = mediapipe::file::RecursivelyCreateDir(model_settings->cache_dir);
    if (!s.ok()) {
      ABSL_LOG(WARNING) << s;
    }
  }

  MP_ASSIGN_OR_RETURN(auto spm_model_content,
                      model_data->ReadMetadata("spm_vocab_model"));

  model_data.reset();

  llm_params.seq_size_T = model_settings->max_num_tokens;
  llm_params.cache_dir = model_settings->cache_dir;

  auto weight_loader = std::make_unique<
      mediapipe::tasks::genai::xnn_utils::DefaultLlmWeightsLoader>(
      model_settings->model_path, llm_params);

  auto runtime_configs =
      std::make_unique<mediapipe::tasks::genai::xnn_utils::RuntimeConfigs>();

  MP_ASSIGN_OR_RETURN(auto llm,
                      mediapipe::tasks::genai::xnn_utils::CreateLlm(
                          llm_params, std::move(runtime_configs),
                          std::move(weight_loader), nullptr, *model_type));

  auto tokenizer = std::make_unique<sentencepiece::SentencePieceProcessor>();
  MP_RETURN_IF_ERROR(tokenizer->LoadFromSerializedProto(spm_model_content));

  std::vector<int> prompt_ids;
  auto status = tokenizer->Encode("hello", &prompt_ids);

  std::unique_ptr<sentencepiece::normalizer::Normalizer> normalizer;
  if (tokenizer->model_proto().has_denormalizer_spec() &&
      tokenizer->model_proto().denormalizer_spec().has_precompiled_charsmap() &&
      !tokenizer->model_proto()
           .denormalizer_spec()
           .precompiled_charsmap()
           .empty()) {
    normalizer = std::make_unique<sentencepiece::normalizer::Normalizer>(
        tokenizer->model_proto().denormalizer_spec());
  }

  std::unique_ptr<LlmInferenceEngineCpu_Engine> engine(
      new LlmInferenceEngineCpu_Engine{
          .tokenizer = tokenizer.release(),
          .normalizer = normalizer.release(),
          .llm = llm.release(),
          .start_token_id = llm_params_proto.start_token_id(),
          .stop_tokens =
              std::vector<std::string>(llm_params_proto.stop_tokens().begin(),
                                       llm_params_proto.stop_tokens().end()),
          .max_num_tokens = model_settings->max_num_tokens,
      });

  return engine.release();
}

absl::StatusOr<LlmInferenceEngine_Session*>
LlmInferenceEngine_CreateSession_Helper(
    const LlmInferenceEngineCpu_Engine* engine,
    const LlmSessionConfig* session_config) {
  std::unique_ptr<LlmInferenceEngineCpu_Session> session(
      new LlmInferenceEngineCpu_Session{.engine = engine});

  return session.release();
}

}  // namespace

void LlmInferenceEngine_CloseResponseContext(
    LlmResponseContext* response_context) {
  for (size_t i = 0; i < response_context->response_count; i++) {
    free(const_cast<char*>(response_context->response_array[i]));
  }
  free(response_context->response_array);
  response_context->response_array = nullptr;
  response_context->response_count = 0;
}

int LlmInferenceEngine_CreateEngine(const LlmModelSettings* model_settings,
                                    LlmInferenceEngine_Session** engine_out,
                                    char** error_msg) {
  auto engine = LlmInferenceEngine_CreateEngine_Helper(model_settings);
  if (!engine.ok()) {
    if (error_msg) {
      *error_msg = strdup(
          absl::StrCat("Failed to create engine: ", engine.status().ToString())
              .c_str());
    }
    return static_cast<int>(engine.status().code());
  }
  *engine_out = engine.value();
  return 0;
}

void LlmInferenceEngine_Engine_Delete(LlmInferenceEngine_Engine* engine) {
  delete reinterpret_cast<LlmInferenceEngineCpu_Engine*>(engine);
}

int LlmInferenceEngine_CreateSession(LlmInferenceEngine_Engine* engine,
                                     const LlmSessionConfig* session_config,
                                     LlmInferenceEngine_Session** session_out,
                                     char** error_msg) {
  auto cpu_engine = reinterpret_cast<LlmInferenceEngineCpu_Engine*>(engine);
  auto session =
      LlmInferenceEngine_CreateSession_Helper(cpu_engine, session_config);
  if (!session.ok()) {
    if (error_msg) {
      *error_msg = strdup(absl::StrCat("Failed to create session: ",
                                       session.status().ToString())
                              .c_str());
    }
    return static_cast<int>(session.status().code());
  }
  *session_out = session.value();
  return 0;
}

void LlmInferenceEngine_Session_Delete(LlmInferenceEngine_Session* session) {
  delete reinterpret_cast<LlmInferenceEngineCpu_Session*>(session);
}

int LlmInferenceEngine_Session_AddQueryChunk(
    LlmInferenceEngine_Session* session, const char* input, char** error_msg) {
  auto cpu_session = reinterpret_cast<LlmInferenceEngineCpu_Session*>(session);
  cpu_session->prompt = input;
  return 0;
}

LlmResponseContext LlmInferenceEngine_Session_PredictSync(
    LlmInferenceEngine_Session* session) {
  LlmInferenceEngine_Session_PredictAsync(
      session, nullptr,
      [](void* callback_context, LlmResponseContext* response_context) {});

  auto cpu_session = reinterpret_cast<LlmInferenceEngineCpu_Session*>(session);
  pthread_join(cpu_session->work_id, nullptr);
  cpu_session->work_id = 0;
  auto final_output = cpu_session->final_output;

  char** result = (char**)malloc(sizeof(char*) * 1);
  if (result == nullptr) {
    ABSL_LOG(FATAL) << "Failed to allocate result for cpu session.";
  }

  result[0] = (char*)malloc(sizeof(char*) * (final_output.size() + 1));
  if (result[0] == nullptr) {
    ABSL_LOG(FATAL) << "Failed to allocate result for cpu session.";
  }

  snprintf(result[0], final_output.size() + 1, "%s", final_output.c_str());

  LlmResponseContext response_context = {
      .response_array = result,
      .response_count = 1,
      .done = true,
  };

  return response_context;
}

void LlmInferenceEngine_Session_PredictAsync(
    LlmInferenceEngine_Session* session, void* callback_context,
    void (*callback)(void* callback_context,
                     LlmResponseContext* response_context)) {
  auto cpu_session = reinterpret_cast<LlmInferenceEngineCpu_Session*>(session);

  cpu_session->cpu_callback = [=](std::string responses) -> void {
    char** result = (char**)malloc(sizeof(char*) * 1);
    if (result == nullptr) {
      ABSL_LOG(FATAL) << "Failed to allocate result for cpu session.";
    }

    result[0] = (char*)malloc(sizeof(char*) * (responses.size() + 1));
    if (result[0] == nullptr) {
      ABSL_LOG(FATAL) << "Failed to allocate result for cpu session.";
    }

    snprintf(result[0], responses.size() + 1, "%s", responses.c_str());
    auto response_context = std::make_unique<LlmResponseContext>();
    response_context->response_array = result,
    response_context->response_count = 1,
    response_context->done = cpu_session->early_stop;
    callback(callback_context, response_context.release());
  };

  cpu_session->final_output = "";
  cpu_session->last_10_char = "";
  cpu_session->early_stop = false;

  pthread_t work_id = 0;
  cpu_session->work_id = work_id;
  pthread_create(&cpu_session->work_id, nullptr, start_llm_function,
                 cpu_session);
}

int LlmInferenceEngine_Session_Clone(
    LlmInferenceEngine_Session* session,
    LlmInferenceEngine_Session** cloned_session, char** error_msg) {
  *error_msg = strdup("Not implemented");
  return 12;
}

int LlmInferenceEngine_Session_SizeInTokens(LlmInferenceEngine_Session* session,
                                            const char* input,
                                            char** error_msg) {
  auto cpu_session = reinterpret_cast<LlmInferenceEngineCpu_Session*>(session);
  std::vector<int> output_ids;
  auto status = cpu_session->engine->tokenizer->Encode(input, &output_ids);
  if (!status.ok()) {
    *error_msg = strdup(status.ToString().c_str());
    return -1;
  }
  return output_ids.size();
}
