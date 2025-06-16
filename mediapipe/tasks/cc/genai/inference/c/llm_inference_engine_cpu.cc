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

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <iterator>
#include <memory>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/absl_check.h"
#include "absl/log/absl_log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_replace.h"
#include "absl/strings/string_view.h"
#include "mediapipe/framework/deps/file_path.h"
#include "mediapipe/framework/port/file_helpers.h"
#include "mediapipe/framework/port/ret_check.h"
#include "mediapipe/framework/port/status_macros.h"
#include "mediapipe/tasks/cc/core/model_asset_bundle_resources.h"
#include "mediapipe/tasks/cc/genai/inference/c/llm_inference_engine.h"
#include "mediapipe/tasks/cc/genai/inference/proto/llm_params.pb.h"
#include "mediapipe/tasks/cc/genai/inference/proto/transformer_params.pb.h"
#include "mediapipe/tasks/cc/genai/inference/utils/llm_utils/metadata_utils.h"
#include "mediapipe/tasks/cc/genai/inference/utils/llm_utils/model_data.h"
#include "mediapipe/tasks/cc/genai/inference/utils/xnn_utils/graph_builder.h"
#include "mediapipe/tasks/cc/genai/inference/utils/xnn_utils/llm.h"
#include "mediapipe/tasks/cc/genai/inference/utils/xnn_utils/llm_builder_factory.h"
#include "mediapipe/tasks/cc/genai/inference/utils/xnn_utils/llm_weights.h"
// clang-format off
#include "mediapipe/tasks/cc/genai/inference/utils/llm_utils/scoped_file.h"
// clang-format on
#include "sentencepiece/src/sentencepiece_processor.h"  // from @com_google_sentencepiece
#include "sentencepiece/src/util.h"  // from @com_google_sentencepiece
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/delegates/xnnpack/xnnpack_delegate.h"
#include "tensorflow/lite/experimental/genai/genai_ops.h"
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/interpreter_builder.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model_builder.h"

namespace {

using ::mediapipe::tasks::genai::llm_utils::ScopedFile;

constexpr int kCheckLastKChars = 10;

struct TfLiteLlm {
  std::unique_ptr<tflite::Interpreter> interpreter;
  std::unique_ptr<mediapipe::tasks::core::ModelAssetBundleResources> resources;
};

struct LlmInferenceEngineCpu_Engine {
  const sentencepiece::SentencePieceProcessor* tokenizer;
  const absl::flat_hash_map<unsigned char, int>* bytes_to_unicode_mapper;
  const absl::flat_hash_map<int, unsigned char>* unicode_to_bytes_mapper;
  const std::variant<mediapipe::tasks::genai::xnn_utils::Llm*, TfLiteLlm*> llm;
  const int start_token_id;
  const std::vector<std::string> stop_tokens;
  const size_t max_num_tokens;

  ~LlmInferenceEngineCpu_Engine() {
    delete tokenizer;
    delete bytes_to_unicode_mapper;
    delete unicode_to_bytes_mapper;
    if (std::holds_alternative<mediapipe::tasks::genai::xnn_utils::Llm*>(llm)) {
      delete std::get<mediapipe::tasks::genai::xnn_utils::Llm*>(llm);
    } else {
      delete std::get<TfLiteLlm*>(llm);
    }
  };
};

struct LlmInferenceEngineCpu_Session {
  const LlmInferenceEngineCpu_Engine* engine;
  std::string prompt;
  int timestep;
  std::string last_10_char;
  std::string final_output;
  std::function<void(std::string)> cpu_callback;
  bool early_stop;
  pthread_t work_id;
  int next_token_id;
  ~LlmInferenceEngineCpu_Session() { pthread_join(work_id, nullptr); };
};

absl::StatusOr<std::unique_ptr<absl::flat_hash_map<unsigned char, int>>>
CreateBytesToUnicodeMapper() {
  auto bytes_to_unicode_mapper =
      std::make_unique<absl::flat_hash_map<unsigned char, int>>();
  // "!" - "~"
  for (int i = 33; i <= 126; i++) {
    bytes_to_unicode_mapper->insert({static_cast<uint8_t>(i), i});
  }
  // "¡" - "¬"
  for (int i = 161; i <= 172; i++) {
    bytes_to_unicode_mapper->insert({static_cast<uint8_t>(i), i});
  }
  // "®" - "ÿ"
  for (int i = 174; i < 256; i++) {
    bytes_to_unicode_mapper->insert({static_cast<uint8_t>(i), i});
  }
  int n = 0;
  for (int b = 0; b < 256; b++) {
    if (!bytes_to_unicode_mapper->contains(static_cast<uint8_t>(b))) {
      bytes_to_unicode_mapper->insert({static_cast<uint8_t>(b), 256 + n});
      n += 1;
    }
  }
  return bytes_to_unicode_mapper;
}

absl::StatusOr<std::unique_ptr<absl::flat_hash_map<int, unsigned char>>>
CreateUnicodeToBytesMapper() {
  MP_ASSIGN_OR_RETURN(auto bytes_to_unicode_mapper,
                      CreateBytesToUnicodeMapper());
  auto unicode_to_bytes_mapper =
      std::make_unique<absl::flat_hash_map<int, unsigned char>>();
  for (const auto& [key, value] : *bytes_to_unicode_mapper) {
    unicode_to_bytes_mapper->insert({value, key});
  }
  return unicode_to_bytes_mapper;
}

std::string MapBytesToUnicode(
    absl::string_view prompt,
    const absl::flat_hash_map<unsigned char, int>* bytes_to_unicode_mapper) {
  std::string converted_prompt = "";
  for (const uint8_t byte : prompt) {
    converted_prompt.append(sentencepiece::string_util::UnicodeCharToUTF8(
        bytes_to_unicode_mapper->at(byte)));
  }
  return converted_prompt;
}

std::string MapUnicodeToBytes(
    absl::string_view output,
    const absl::flat_hash_map<int, uint8_t>* unicode_to_bytes_mapper) {
  sentencepiece::string_util::UnicodeText unicode_text =
      sentencepiece::string_util::UTF8ToUnicodeText(output);
  std::string converted_output = "";
  for (const int code_point : unicode_text) {
    if (!unicode_to_bytes_mapper->contains(code_point)) {
      converted_output += code_point;
    } else {
      converted_output += unicode_to_bytes_mapper->at(code_point);
    }
  }
  return converted_output;
}

void* next_token_function(void* args) {
  struct LlmInferenceEngineCpu_Session* cpu_session =
      (struct LlmInferenceEngineCpu_Session*)args;
  if (cpu_session->timestep < cpu_session->engine->max_num_tokens) {
    if (cpu_session->early_stop) {
      return nullptr;
    }

    auto token_ids_per_step = std::vector<int>();
    if (std::holds_alternative<mediapipe::tasks::genai::xnn_utils::Llm*>(
            cpu_session->engine->llm)) {
      auto status = std::get<mediapipe::tasks::genai::xnn_utils::Llm*>(
                        cpu_session->engine->llm)
                        ->GetNextToken(&token_ids_per_step);
      if (!status.ok()) {
        ABSL_LOG(FATAL) << "Failed to generate output: " << status;
      }
    } else {
      auto llm = std::get<TfLiteLlm*>(cpu_session->engine->llm);
      auto* decode_runner = llm->interpreter->GetSignatureRunner("decode");
      ABSL_CHECK_EQ(decode_runner->AllocateTensors(), kTfLiteOk);
      TfLiteTensor* decode_input = decode_runner->input_tensor("args_0");
      TfLiteTensor* decode_input_pos = decode_runner->input_tensor("args_1");
      decode_input->data.i64[0] =
          static_cast<int64_t>(cpu_session->next_token_id);
      decode_input_pos->data.i64[0] =
          static_cast<int64_t>(cpu_session->timestep);

      // logits->dims->data[0] = batch size
      // logits->dims->data[1] = sequence length
      // logits->dims->data[2] = vocab size
      const TfLiteTensor* logits = decode_runner->output_tensor("output_0");

      ABSL_CHECK_EQ(decode_runner->Invoke(), kTfLiteOk);

      auto max_logit_it = std::max_element(
          logits->data.f, logits->data.f + logits->dims->data[2]);
      token_ids_per_step.push_back(std::distance(logits->data.f, max_logit_it));
    }

    // For future multithreading support.
    if (cpu_session->early_stop) {
      return nullptr;
    }

    if (cpu_session->timestep >= cpu_session->engine->max_num_tokens) {
      cpu_session->early_stop = true;
    }

    cpu_session->next_token_id = token_ids_per_step[0];

    std::string token =
        cpu_session->engine->tokenizer->IdToPiece(token_ids_per_step[0]);
    if (cpu_session->engine->unicode_to_bytes_mapper != nullptr) {
      token = MapUnicodeToBytes(token,
                                cpu_session->engine->unicode_to_bytes_mapper);
    } else {
      token = absl::StrReplaceAll(token, {{"▁", " "}});
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

    ++cpu_session->timestep;

    next_token_function(args);
  }
  return nullptr;
};

void* start_llm_function(void* args) {
  struct LlmInferenceEngineCpu_Session* cpu_session =
      (struct LlmInferenceEngineCpu_Session*)args;

  std::vector<int> prompt_ids = {};

  std::string prompt;
  if (cpu_session->engine->bytes_to_unicode_mapper != nullptr) {
    prompt = MapBytesToUnicode(cpu_session->prompt,
                               cpu_session->engine->bytes_to_unicode_mapper);
  } else {
    prompt = cpu_session->prompt;
  }

  auto status = cpu_session->engine->tokenizer->Encode(prompt, &prompt_ids);

  if (!status.ok()) {
    ABSL_LOG(FATAL) << "Failed to encode input: " << status;
  }
  prompt_ids.insert(prompt_ids.begin(), cpu_session->engine->start_token_id);

  if (std::holds_alternative<mediapipe::tasks::genai::xnn_utils::Llm*>(
          cpu_session->engine->llm)) {
    auto llm = std::get<mediapipe::tasks::genai::xnn_utils::Llm*>(
        cpu_session->engine->llm);
    ABSL_CHECK_OK(llm->SeekTimeStep(0));
    ABSL_CHECK_OK(llm->AddInputTokens({prompt_ids}));
  } else {
    auto llm = std::get<TfLiteLlm*>(cpu_session->engine->llm);
    auto* prefill_runner = llm->interpreter->GetSignatureRunner("prefill");

    ABSL_CHECK_EQ(prefill_runner->AllocateTensors(), kTfLiteOk);

    TfLiteTensor* prefill_input = prefill_runner->input_tensor("args_0");
    TfLiteTensor* prefill_input_pos = prefill_runner->input_tensor("args_1");
    memset(prefill_input->data.data, 0, prefill_input->bytes);
    memset(prefill_input_pos->data.data, 0, prefill_input_pos->bytes);
    cpu_session->next_token_id = prompt_ids.back();
    prompt_ids.pop_back();
    for (int i = 0; i < prompt_ids.size(); ++i) {
      prefill_input->data.i64[i] = static_cast<int64_t>(prompt_ids[i]);
      prefill_input_pos->data.i64[i] = static_cast<int64_t>(i);
    }
    ABSL_CHECK_EQ(prefill_runner->Invoke(), kTfLiteOk);
  }

  cpu_session->timestep = prompt_ids.size();

  next_token_function(args);

  return nullptr;
}

absl::StatusOr<std::unique_ptr<LlmInferenceEngineCpu_Engine>>
CreateXnnLlmCpuEngine(const LlmModelSettings* model_settings) {
  MP_ASSIGN_OR_RETURN(auto model_file,
                      ScopedFile::Open(model_settings->model_path));
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

  std::unique_ptr<absl::flat_hash_map<unsigned char, int>>
      bytes_to_unicode_mapper;
  std::unique_ptr<absl::flat_hash_map<int, unsigned char>>
      unicode_to_bytes_mapper;
  // These models uses GPT2 style unicode mapping, which additional mapping is
  // needed.
  if (model_type == odml::infra::proto::LLM_MODEL_TYPE_STABLELM_4E1T_3B ||
      model_type == odml::infra::proto::LLM_MODEL_TYPE_FALCON_RW_1B ||
      model_type == odml::infra::proto::LLM_MODEL_TYPE_PHI_2) {
    MP_ASSIGN_OR_RETURN(bytes_to_unicode_mapper, CreateBytesToUnicodeMapper());
    MP_ASSIGN_OR_RETURN(unicode_to_bytes_mapper, CreateUnicodeToBytesMapper());
  }

  std::unique_ptr<LlmInferenceEngineCpu_Engine> engine(
      new LlmInferenceEngineCpu_Engine{
          .tokenizer = tokenizer.release(),
          .bytes_to_unicode_mapper = bytes_to_unicode_mapper.release(),
          .unicode_to_bytes_mapper = unicode_to_bytes_mapper.release(),
          .llm = llm.release(),
          .start_token_id = llm_params_proto.start_token_id(),
          .stop_tokens =
              std::vector<std::string>(llm_params_proto.stop_tokens().begin(),
                                       llm_params_proto.stop_tokens().end()),
          .max_num_tokens = model_settings->max_num_tokens,
      });

  return engine;
}

// Creates an inference engine from a *.task file.
// This method extracts the TF_LITE_PREFILL_DECODE, TOKENIZER_MODEL and METADATA
// files from the task bundle and initializes the TfLIte XNNPack delegate.
absl::StatusOr<std::unique_ptr<LlmInferenceEngineCpu_Engine>>
CreateTfliteLlmCpuEngine(const LlmModelSettings* model_settings) {
  auto external_file =
      std::make_unique<mediapipe::tasks::core::proto::ExternalFile>();
  if (model_settings) {
    external_file->set_file_name(model_settings->model_path);
  }
  MP_ASSIGN_OR_RETURN(auto resources,
                      mediapipe::tasks::core::ModelAssetBundleResources::Create(
                          "", std::move(external_file)));
  const std::vector<std::string>& files_list = resources->ListFiles();
  const absl::flat_hash_set<std::string> files_set(files_list.begin(),
                                                   files_list.end());

  std::unique_ptr<tflite::Interpreter> interpreter;
  if (!files_set.contains("TF_LITE_PREFILL_DECODE")) {
    return absl::InvalidArgumentError("TF_LITE_PREFILL_DECODE not found.");
  }
  if (!files_set.contains("TOKENIZER_MODEL")) {
    return absl::InvalidArgumentError("TOKENIZER_MODEL not found.");
  }
  if (!files_set.contains("METADATA")) {
    return absl::InvalidArgumentError("METADATA not found.");
  }
  MP_ASSIGN_OR_RETURN(absl::string_view model_buffer,
                      resources->GetFile("TF_LITE_PREFILL_DECODE"));
  MP_ASSIGN_OR_RETURN(absl::string_view tokenizer_buffer,
                      resources->GetFile("TOKENIZER_MODEL"));
  MP_ASSIGN_OR_RETURN(absl::string_view params_buffer,
                      resources->GetFile("METADATA"));
  auto model = tflite::FlatBufferModel::BuildFromBuffer(model_buffer.data(),
                                                        model_buffer.size());
  RET_CHECK(model) << "Failed to build TF_LITE_PREFILL_DECODE model.";
  tflite::ops::builtin::BuiltinOpResolver resolver;
  // NOTE: We need to manually register optimized OPs for KV-cache and
  // Scaled Dot Product Attention (SDPA).
  tflite::ops::custom::GenAIOpsRegisterer(&resolver);
  tflite::InterpreterBuilder builder(*model, resolver);
  RET_CHECK(model_settings);
  builder(&interpreter);
  RET_CHECK_NE(interpreter, nullptr);

  // RET_CHECK(model_settings->xnnpack_options.has_value());
  auto delegate_options = TfLiteXNNPackDelegateOptionsDefault();
  // Set the number of threads to 4 as default.
  delegate_options.num_threads = 4;
  // Compute the path for the cache file.
  std::string weight_cache_path = model_settings->cache_dir;
  if (weight_cache_path != ":nocache") {
    if (weight_cache_path.empty()) {
      weight_cache_path =
          absl::StrCat(model_settings->model_path, ".xnnpack_cache");
    } else {
      weight_cache_path = mediapipe::file::JoinPath(
          weight_cache_path,
          absl::StrCat(mediapipe::file::Basename(model_settings->model_path),
                       ".xnnpack_cache"));
    }
    delegate_options.weight_cache_file_path = weight_cache_path.c_str();
  }
  RET_CHECK_EQ(interpreter->ModifyGraphWithDelegate(
                   tflite::Interpreter::TfLiteDelegatePtr(
                       TfLiteXNNPackDelegateCreate(&delegate_options),
                       [](TfLiteDelegate* delegate) {
                         TfLiteXNNPackDelegateDelete(delegate);
                       })),
               kTfLiteOk);
  RET_CHECK_EQ(interpreter->SetNumThreads(4), kTfLiteOk);

  auto tflite_llm = std::make_unique<TfLiteLlm>(
      TfLiteLlm{std::move(interpreter), std::move(resources)});

  auto tokenizer = std::make_unique<sentencepiece::SentencePieceProcessor>();
  MP_RETURN_IF_ERROR(tokenizer->LoadFromSerializedProto(tokenizer_buffer));

  auto llm_parameters = odml::infra::proto::LlmParameters();
  RET_CHECK(llm_parameters.ParseFromArray(params_buffer.data(),
                                          params_buffer.size()));

  auto start_token_id = tokenizer->PieceToId(llm_parameters.start_token());

  std::unique_ptr<LlmInferenceEngineCpu_Engine> engine(
      new LlmInferenceEngineCpu_Engine{
          .tokenizer = tokenizer.release(),
          .bytes_to_unicode_mapper = nullptr,
          .unicode_to_bytes_mapper = nullptr,
          .llm = tflite_llm.release(),
          .start_token_id = start_token_id,
          .stop_tokens =
              std::vector<std::string>(llm_parameters.stop_tokens().begin(),
                                       llm_parameters.stop_tokens().end()),
          .max_num_tokens = model_settings->max_num_tokens,
      });

  return engine;
}

absl::StatusOr<LlmInferenceEngine_Engine*>
LlmInferenceEngine_CreateEngine_Helper(const LlmModelSettings* model_settings) {
  std::unique_ptr<LlmInferenceEngineCpu_Engine> engine;
  if (absl::EndsWith(model_settings->model_path, ".tflite")) {
    MP_ASSIGN_OR_RETURN(engine, CreateXnnLlmCpuEngine(model_settings));
  } else {
    MP_ASSIGN_OR_RETURN(engine, CreateTfliteLlmCpuEngine(model_settings));
  }

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

int LlmInferenceEngine_Session_Delete(LlmInferenceEngine_Session* session) {
  delete reinterpret_cast<LlmInferenceEngineCpu_Session*>(session);
  return 0;
}

int LlmInferenceEngine_Session_AddQueryChunk(
    LlmInferenceEngine_Session* session, const char* input, char** error_msg) {
  auto cpu_session = reinterpret_cast<LlmInferenceEngineCpu_Session*>(session);
  cpu_session->prompt = input;
  return 0;
}

ODML_EXPORT int LlmInferenceEngine_Session_AddImage(
    LlmInferenceEngine_Session* session, const void* sk_bitmap,
    char** error_msg) {
  *error_msg = strdup("Not implemented");
  return 12;
}

ODML_EXPORT int LlmInferenceEngine_Session_AddAudio(
    LlmInferenceEngine_Engine* engine, LlmInferenceEngine_Session* session,
    const char* audio_bytes, int audio_bytes_size, char** error_msg) {
  *error_msg = strdup("Not implemented");
  return 12;
}

int LlmInferenceEngine_Session_PredictSync(LlmInferenceEngine_Session* session,
                                           LlmResponseContext* response_context,
                                           char** error_msg) {
  auto status = LlmInferenceEngine_Session_PredictAsync(
      session, nullptr, error_msg,
      [](void* callback_context, LlmResponseContext* response_context) {});
  if (status != 0) {
    return status;
  }

  auto cpu_session = reinterpret_cast<LlmInferenceEngineCpu_Session*>(session);
  pthread_join(cpu_session->work_id, nullptr);
  cpu_session->work_id = 0;
  auto final_output = cpu_session->final_output;

  char** result = (char**)malloc(sizeof(char*) * 1);
  if (result == nullptr) {
    *error_msg = strdup("Failed to allocate result for cpu session.");
    return static_cast<int>(absl::StatusCode::kResourceExhausted);
  }

  result[0] = (char*)malloc(sizeof(char*) * (final_output.size() + 1));
  if (result[0] == nullptr) {
    *error_msg = strdup("Failed to allocate result for cpu session.");
    return static_cast<int>(absl::StatusCode::kResourceExhausted);
  }

  snprintf(result[0], final_output.size() + 1, "%s", final_output.c_str());

  response_context->response_array = result;
  response_context->response_count = 1;
  response_context->done = true;

  return 0;
}

int LlmInferenceEngine_Session_PredictAsync(
    LlmInferenceEngine_Session* session, void* callback_context,
    char** error_msg,
    void (*callback)(void* callback_context,
                     LlmResponseContext* response_context)) {
  if (session == nullptr) {
    *error_msg = strdup("Session is null.");
    return static_cast<int>(absl::StatusCode::kInvalidArgument);
  }
  if (callback == nullptr) {
    *error_msg = strdup("Callback is null.");
    return static_cast<int>(absl::StatusCode::kInvalidArgument);
  }

  auto cpu_session = reinterpret_cast<LlmInferenceEngineCpu_Session*>(session);

  if (cpu_session == nullptr) {
    *error_msg = strdup("Provided session is not a CPU session.");
    return static_cast<int>(absl::StatusCode::kInvalidArgument);
  }

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

  return 0;
}

int LlmInferenceEngine_Session_PendingProcessCancellation(
    LlmInferenceEngine_Session* session, char** error_msg) {
  *error_msg = strdup("Not implemented");
  return 12;
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

int LlmInferenceEngine_UpdateRuntimeConfig(LlmInferenceEngine_Session* session,
                                           const SessionRuntimeConfig* config,
                                           char** error_msg) {
  *error_msg = strdup("Not implemented");
  return 12;
}

int LlmInferenceEngine_GetSentencePieceProcessor(
    LlmInferenceEngine_Engine* engine,
    const SentencePieceProcessor** processor_out, char** error_msg) {
  *error_msg = strdup("Not implemented");
  return 12;
}
