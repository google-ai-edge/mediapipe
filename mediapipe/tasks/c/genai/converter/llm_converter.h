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

#ifndef MEDIAPIPE_TASKS_C_GENAI_CONVERTER_LLM_CONVERTER_H_
#define MEDIAPIPE_TASKS_C_GENAI_CONVERTER_LLM_CONVERTER_H_

#include "mediapipe/tasks/c/core/base_options.h"
#include "mediapipe/tasks/c/core/mp_status.h"

#ifndef MP_EXPORT
#if defined(_MSC_VER)
#define MP_EXPORT __declspec(dllexport)
#else
#define MP_EXPORT __attribute__((visibility("default")))
#endif  // _MSC_VER
#endif  // MP_EXPORT

#ifdef __cplusplus
extern "C" {
#endif

// Generates the TfLite flatbuffer file from the serialized weight files
// for the CPU backend.
// Args:
//   model_type: Name of the model, e.g. GEMMA_2B.
//   weight_path: Directory or path for the input weight files.
//   vocab_model_file: The file path to the SentencePiece vocab model.
//   is_quantized: Whether the checkpoint is already quantized.
//   output_tflite_file: The output tflite filename.
//   error_message: An optional pointer to an error message.  If provided, it
//   will be populated with a newly-allocated error message upon
//   failure. It's the caller responsibility to free the error message with
//   `free()`.
// Returns:
//   kMpOk on success, otherwise an error code.
MP_EXPORT MpStatus MpLlmConverterGenerateCpuTfLite(
    const char* model_type, const char* weight_path,
    const char* vocab_model_file, bool is_quantized,
    const char* output_tflite_file, char** error_message);

// Generates the TfLite flatbuffer file from the serialized weight files
// for the GPU backend.
// Args:
//   model_type: Name of the model, e.g. GEMMA_2B.
//   weight_path: Directory or path for the input weight files.
//   vocab_model_file: The file path to the SentencePiece vocab model.
//   is_quantized: Whether the checkpoint is already quantized.
//   obfuscate: Whether to obfuscate the model.
//   output_tflite_file: The output tflite filename.
//   lora_rank: An integer representing the rank of LoRA.
//   lora_weight_path: The directory or path for the lora checkpoint.
//   lora_output_tflite_file: The name of the generated tflite file for LoRA.
//   lora_main_model_type: The main model type for LoRA.
//   image_encoder_file: The name of the image encoder tflite file.
//   image_adapter_file: The name of the image adapter tflite file.
//   submodel_type: Name of submodel, e.g. GEMMA_2B.
//   use_dynamic_ple: Whether any PLE embeddings should be loaded dynamically.
//   apply_srq: Whether to use SRQ.
//   error_message: An optional pointer to an error message.  If provided, it
//   will be populated with a newly-allocated error message upon
//   failure. It's the caller responsibility to free the error message with
//   `free()`.
// Returns:
//   kMpOk on success, otherwise an error code.
MP_EXPORT MpStatus MpLlmConverterGenerateGpuTfLite(
    const char* model_type, const char* weight_path,
    const char* vocab_model_file, bool is_quantized, bool obfuscate,
    const char* output_tflite_file, int lora_rank, const char* lora_weight_path,
    const char* lora_output_tflite_file, const char* lora_main_model_type,
    const char* image_encoder_file, const char* image_adapter_file,
    const char* submodel_type, bool use_dynamic_ple, bool apply_srq,
    int block_size, char** error_message);

// Converts the Hugging Face BPE tokenizer to internal SentencePiece
// vocab model.
// Args:
//   vocab_model_file: The directory containing tokenizer.json and
//     tokenizer_config.json.
//   output_vocab_file: The output file path for the SentencePiece model.
//   error_message: An optional pointer to an error message.  If provided, it
//   will be populated with a newly-allocated error message upon
//   failure. It's the caller responsibility to free the error message with
//   `free()`.
// Returns:
//   kMpOk on success, otherwise an error code.
MP_EXPORT MpStatus MpLlmConverterConvertHfTokenizer(
    const char* vocab_model_file, const char* output_vocab_file,
    char** error_message);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // MEDIAPIPE_TASKS_C_GENAI_CONVERTER_LLM_CONVERTER_H_
