/* Copyright 2023 The MediaPipe Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "mediapipe/tasks/cc/text/custom_ops/sentencepiece/sentencepiece_tokenizer_tflite.h"

#include <cstdint>

#include "flatbuffers/flexbuffers.h"
#include "mediapipe/tasks/cc/text/custom_ops/sentencepiece/optimized_encoder.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/context.h"
#include "tensorflow/lite/kernels/internal/tensor.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/model.h"
#include "tensorflow/lite/string_util.h"

namespace mediapipe::tflite_operations {
namespace sentencepiece::tokenizer {
namespace {

using ::tflite::SetTensorToDynamic;

constexpr int kSPModelIndex = 0;
constexpr int kInputIndex = 1;
constexpr int kAddBOSInput = 4;
constexpr int kAddEOSInput = 5;
constexpr int kReverseInput = 6;

constexpr int kOutputValuesInd = 0;
constexpr int kOutputSplitsInd = 1;

TfLiteIntArray* CreateSizeArray(const std::initializer_list<int>& sizes) {
  TfLiteIntArray* array_size = TfLiteIntArrayCreate(sizes.size());
  int index = 0;
  for (const int size : sizes) {
    array_size->data[index++] = size;
  }
  return array_size;
}
}  // namespace

// Initializes text encoder object from serialized parameters.
void* Initialize(TfLiteContext* /*context*/, const char* /*buffer*/,
                 size_t /*length*/) {
  return nullptr;
}
void Free(TfLiteContext* /*context*/, void* /*buffer*/) {}

TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) {
  // TODO: Add checks for input and output tensors.
  TfLiteTensor& output_values =
      context->tensors[node->outputs->data[kOutputValuesInd]];
  SetTensorToDynamic(&output_values);

  TfLiteTensor& output_splits =
      context->tensors[node->outputs->data[kOutputSplitsInd]];
  SetTensorToDynamic(&output_splits);
  return kTfLiteOk;
}

TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
  const TfLiteTensor& model_tensor =
      context->tensors[node->inputs->data[kSPModelIndex]];
  const auto model_buffer_data = model_tensor.data.data;
  const TfLiteTensor& input_text =
      context->tensors[node->inputs->data[kInputIndex]];

  const TfLiteTensor add_bos_tensor =
      context->tensors[node->inputs->data[kAddBOSInput]];
  const bool add_bos = add_bos_tensor.data.b[0];
  const TfLiteTensor add_eos_tensor =
      context->tensors[node->inputs->data[kAddEOSInput]];
  const bool add_eos = add_eos_tensor.data.b[0];
  const TfLiteTensor reverse_tensor =
      context->tensors[node->inputs->data[kReverseInput]];
  const bool reverse = reverse_tensor.data.b[0];

  std::vector<int32_t> encoded;
  std::vector<int32_t> splits;
  const int num_strings = tflite::GetStringCount(&input_text);
  for (int i = 0; i < num_strings; ++i) {
    const auto strref = tflite::GetString(&input_text, i);
    const auto res = EncodeString(std::string(strref.str, strref.len),
                                  model_buffer_data, add_bos, add_eos, reverse);
    TF_LITE_ENSURE_MSG(context, res.type == EncoderResultType::SUCCESS,
                       "Sentencepiece conversion failed");
    std::copy(res.codes.begin(), res.codes.end(), std::back_inserter(encoded));
    splits.emplace_back(encoded.size());
  }

  TfLiteTensor& output_values =
      context->tensors[node->outputs->data[kOutputValuesInd]];
  TF_LITE_ENSURE_OK(context,
                    context->ResizeTensor(
                        context, &output_values,
                        CreateSizeArray({static_cast<int>(encoded.size())})));
  int32_t* output_values_flat = output_values.data.i32;
  std::copy(encoded.begin(), encoded.end(), output_values_flat);
  TfLiteTensor& output_splits =
      context->tensors[node->outputs->data[kOutputSplitsInd]];
  TF_LITE_ENSURE_OK(
      context, context->ResizeTensor(
                   context, &output_splits,
                   CreateSizeArray({static_cast<int>(splits.size() + 1)})));
  int32_t* output_splits_flat = output_splits.data.i32;
  *output_splits_flat = 0;
  std::copy(splits.begin(), splits.end(), output_splits_flat + 1);
  return kTfLiteOk;
}
}  // namespace sentencepiece::tokenizer

TfLiteRegistration* Register_SENTENCEPIECE_TOKENIZER() {
  static TfLiteRegistration r = {
      sentencepiece::tokenizer::Initialize, sentencepiece::tokenizer::Free,
      sentencepiece::tokenizer::Prepare, sentencepiece::tokenizer::Eval};
  return &r;
}

}  // namespace mediapipe::tflite_operations
