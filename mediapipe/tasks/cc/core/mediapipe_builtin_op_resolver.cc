/* Copyright 2022 The MediaPipe Authors.

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

#include "mediapipe/tasks/cc/core/mediapipe_builtin_op_resolver.h"

#include "mediapipe/tasks/cc/text/custom_ops/ragged/ragged_tensor_to_tensor_tflite.h"
#include "mediapipe/tasks/cc/text/custom_ops/sentencepiece/sentencepiece_tokenizer_tflite.h"
#include "mediapipe/tasks/cc/text/language_detector/custom_ops/kmeans_embedding_lookup.h"
#include "mediapipe/tasks/cc/text/language_detector/custom_ops/ngram_hash.h"
#include "mediapipe/tasks/cc/vision/custom_ops/fused_batch_norm.h"
#include "mediapipe/util/tflite/operations/landmarks_to_transform_matrix.h"
#include "mediapipe/util/tflite/operations/max_pool_argmax.h"
#include "mediapipe/util/tflite/operations/max_unpooling.h"
#include "mediapipe/util/tflite/operations/transform_landmarks.h"
#include "mediapipe/util/tflite/operations/transform_tensor_bilinear.h"
#include "mediapipe/util/tflite/operations/transpose_conv_bias.h"

namespace mediapipe {
namespace tasks {
namespace core {

MediaPipeBuiltinOpResolver::MediaPipeBuiltinOpResolver() {
  AddCustom("MaxPoolingWithArgmax2D",
            mediapipe::tflite_operations::RegisterMaxPoolingWithArgmax2D());
  AddCustom("MaxUnpooling2D",
            mediapipe::tflite_operations::RegisterMaxUnpooling2D());
  AddCustom("Convolution2DTransposeBias",
            mediapipe::tflite_operations::RegisterConvolution2DTransposeBias());
  AddCustom("TransformTensorBilinear",
            mediapipe::tflite_operations::RegisterTransformTensorBilinearV2(),
            /*version=*/2);
  AddCustom("TransformLandmarks",
            mediapipe::tflite_operations::RegisterTransformLandmarksV2(),
            /*version=*/2);
  AddCustom(
      "Landmarks2TransformMatrix",
      mediapipe::tflite_operations::RegisterLandmarksToTransformMatrixV2(),
      /*version=*/2);
  // For the LanguageDetector model.
  AddCustom("NGramHash", mediapipe::tflite_operations::Register_NGRAM_HASH());
  AddCustom("KmeansEmbeddingLookup",
            mediapipe::tflite_operations::Register_KmeansEmbeddingLookup());
  // For the UniversalSentenceEncoder model.
  AddCustom("TFSentencepieceTokenizeOp",
            mediapipe::tflite_operations::Register_SENTENCEPIECE_TOKENIZER());
  AddCustom("RaggedTensorToTensor",
            mediapipe::tflite_operations::Register_RAGGED_TENSOR_TO_TENSOR());
  AddCustom("FusedBatchNormV3",
            mediapipe::tflite_operations::Register_FusedBatchNorm());
}

}  // namespace core
}  // namespace tasks
}  // namespace mediapipe
