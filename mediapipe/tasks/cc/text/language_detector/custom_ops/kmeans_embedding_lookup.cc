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

#include "mediapipe/tasks/cc/text/language_detector/custom_ops/kmeans_embedding_lookup.h"

#include <algorithm>
#include <cstdint>
#include <iostream>
#include <vector>

#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/kernel_util.h"

namespace mediapipe::tflite_operations {
namespace kmeans_embedding_lookup_op {

namespace {

constexpr int kInputMessage = 0;
constexpr int kEncodingTable = 1;
constexpr int kCodebook = 2;
constexpr int kOutputLabel = 0;

using ::tflite::GetInput;
using ::tflite::GetOutput;
using ::tflite::GetTensorData;

}  // namespace

TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) {
  TfLiteTensor* output = GetOutput(context, node, kOutputLabel);
  TF_LITE_ENSURE(context, output != nullptr);

  TfLiteIntArray* output_size = TfLiteIntArrayCreate(2);
  output_size->data[0] = 1;
  const TfLiteTensor* input = GetInput(context, node, kInputMessage);
  TF_LITE_ENSURE(context, input != nullptr);
  const TfLiteTensor* encoding_table = GetInput(context, node, kEncodingTable);
  TF_LITE_ENSURE(context, encoding_table != nullptr);
  const TfLiteTensor* codebook = GetInput(context, node, kCodebook);
  TF_LITE_ENSURE(context, codebook != nullptr);
  const int encoding_size = encoding_table->dims->data[1];
  const int block_size = codebook->dims->data[1];
  output_size->data[1] = encoding_size * block_size;

  // Check if the inputs and output are typed correctly.
  if (input->type != kTfLiteInt32) {
    context->ReportError(context, "Input type must be Int32.");
    return kTfLiteError;
  }
  if (encoding_table->type != kTfLiteUInt8) {
    context->ReportError(context, "Encoding Table type must be UInt8.");
    return kTfLiteError;
  }
  if (codebook->type != kTfLiteFloat32) {
    context->ReportError(context, "Codebook type must be Float32.");
    return kTfLiteError;
  }
  if (output->type != kTfLiteFloat32) {
    context->ReportError(context, "Output type must be Float32.");
    return kTfLiteError;
  }

  return context->ResizeTensor(context, output, output_size);
}

// This is the core method that generates the aggregated embedding from the
// given input, encoding table and codebook tensors.
void GetEmbedding(const TfLiteTensor* input, const TfLiteTensor* encoding_table,
                  const TfLiteTensor* codebook, float* data) {
  const int input_encoding_size = encoding_table->dims->data[1];
  const int block_size = codebook->dims->data[1];
  const int num_tokens = input->dims->data[1];
  const int output_embedding_size = input_encoding_size * block_size;

  int num_embeddings = 0;
  std::vector<float> final_embedding(output_embedding_size, 0.0);
  for (int token_idx = 0; token_idx < num_tokens; token_idx++) {
    const int32_t token = GetTensorData<int32_t>(input)[token_idx];
    if (token == 0) {
      break;
    }
    ++num_embeddings;

    for (int encoding_dim_idx = 0; encoding_dim_idx < input_encoding_size;
         encoding_dim_idx++) {
      int codebook_idx = GetTensorData<uint8_t>(
          encoding_table)[token * input_encoding_size + encoding_dim_idx];
      for (int block_offset = 0; block_offset < block_size; block_offset++) {
        final_embedding[encoding_dim_idx * block_size + block_offset] +=
            GetTensorData<float>(
                codebook)[codebook_idx * block_size + block_offset];
      }
    }
  }

  // Compute the mean of the embeddings.
  for (int embed_dim_idx = 0; embed_dim_idx < output_embedding_size;
       embed_dim_idx++) {
    data[embed_dim_idx] =
        final_embedding[embed_dim_idx] / (std::max(num_embeddings, 1));
  }
}

TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
  const TfLiteTensor* input = GetInput(context, node, kInputMessage);
  TF_LITE_ENSURE(context, input != nullptr);
  const TfLiteTensor* encoding_table = GetInput(context, node, kEncodingTable);
  TF_LITE_ENSURE(context, encoding_table != nullptr);
  const TfLiteTensor* codebook = GetInput(context, node, kCodebook);
  TF_LITE_ENSURE(context, codebook != nullptr);
  TfLiteTensor* output = GetOutput(context, node, kOutputLabel);
  TF_LITE_ENSURE(context, output != nullptr);

  // Sanity checks on the input.
  const int batch_size = input->dims->data[0];
  if (batch_size != 1) {
    context->ReportError(context, "`batch_size` must be == 1.");
    return kTfLiteError;
  }

  // Compute the output embedding.
  GetEmbedding(input, encoding_table, codebook, GetTensorData<float>(output));

  return kTfLiteOk;
}

}  // namespace kmeans_embedding_lookup_op

TfLiteRegistration* Register_KmeansEmbeddingLookup() {
  static TfLiteRegistration r = {nullptr, nullptr,
                                 kmeans_embedding_lookup_op::Prepare,
                                 kmeans_embedding_lookup_op::Eval};
  return &r;
}

}  // namespace mediapipe::tflite_operations
