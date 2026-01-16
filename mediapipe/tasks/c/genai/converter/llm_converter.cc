// Copyright 2025 The MediaPipe Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "mediapipe/tasks/c/genai/converter/llm_converter.h"

#include <string>

#include "absl/status/status.h"
#include "mediapipe/tasks/c/core/mp_status.h"
#include "mediapipe/tasks/c/core/mp_status_converter.h"
#include "mediapipe/tasks/cc/text/utils/vocab_convert_utils.h"
#ifdef ENABLE_ODML_CONVERTER
#include "odml/infra/genai/inference/ml_drift/llm/tensor_loaders/model_ckpt_util.h"
#include "odml/infra/genai/inference/utils/xnn_utils/model_ckpt_util.h"
#endif  // ENABLE_ODML_CONVERTER

extern "C" {

MpStatus MpLlmConverterGenerateCpuTfLite(const char* model_type,
                                         const char* weight_path,
                                         const char* vocab_model_file,
                                         bool is_quantized,
                                         const char* output_tflite_file,
                                         char** error_message) {
#ifdef ENABLE_ODML_CONVERTER
  absl::Status status = odml::infra::xnn_utils::GenerateTfLite(
      model_type, weight_path, vocab_model_file, is_quantized,
      output_tflite_file);
  return mediapipe::tasks::c::core::HandleStatus(status, error_message);
#else
  return mediapipe::tasks::c::core::HandleStatus(
      absl::UnimplementedError("LLM converter is not enabled."), error_message);
#endif  // ENABLE_ODML_CONVERTER
}

MpStatus MpLlmConverterGenerateGpuTfLite(
    const char* model_type, const char* weight_path,
    const char* vocab_model_file, bool is_quantized, bool obfuscate,
    const char* output_tflite_file, int lora_rank, const char* lora_weight_path,
    const char* lora_output_tflite_file, const char* lora_main_model_type,
    const char* image_encoder_file, const char* image_adapter_file,
    const char* submodel_type, bool use_dynamic_ple, bool apply_srq,
    int block_size, char** error_message) {
#ifdef ENABLE_ODML_CONVERTER
  // TODO: Update the converter code base
  if (image_encoder_file || image_adapter_file) {
    return mediapipe::tasks::c::core::HandleStatus(
        absl::UnimplementedError("Image encoder not supported in this "
                                 "build."),
        error_message);
  }
  absl::Status status = odml::infra::gpu::GenerateTfLite(
      model_type, weight_path, vocab_model_file, is_quantized, obfuscate,
      output_tflite_file, lora_rank, lora_weight_path, lora_output_tflite_file);
  return mediapipe::tasks::c::core::HandleStatus(status, error_message);
#else
  return mediapipe::tasks::c::core::HandleStatus(
      absl::UnimplementedError("LLM converter is not enabled."), error_message);
#endif  // ENABLE_ODML_CONVERTER
}

MpStatus MpLlmConverterConvertHfTokenizer(const char* vocab_model_file,
                                          const char* output_vocab_file,
                                          char** error_message) {
  absl::Status status = mediapipe::tasks::text::ConvertHfTokenizer(
      vocab_model_file, output_vocab_file);
  return mediapipe::tasks::c::core::HandleStatus(status, error_message);
}

}  // extern "C"
