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

#ifndef MEDIAPIPE_TASKS_C_TEXT_TEXT_EMBEDDER_TEXT_EMBEDDER_H_
#define MEDIAPIPE_TASKS_C_TEXT_TEXT_EMBEDDER_TEXT_EMBEDDER_H_

#include "mediapipe/tasks/c/components/containers/embedding_result.h"
#include "mediapipe/tasks/c/components/processors/embedder_options.h"
#include "mediapipe/tasks/c/core/base_options.h"
#include "mediapipe/tasks/c/core/mp_status.h"

#ifndef MP_EXPORT
#define MP_EXPORT __attribute__((visibility("default")))
#endif  // MP_EXPORT

#ifdef __cplusplus
extern "C" {
#endif

typedef struct MpTextEmbedderInternal* MpTextEmbedderPtr;
typedef struct EmbeddingResult TextEmbedderResult;

// The options for configuring a MediaPipe text embedder task.
struct TextEmbedderOptions {
  // Base options for configuring MediaPipe Tasks, such as specifying the model
  // file with metadata, accelerator options, op resolver, etc.
  struct BaseOptions base_options;

  // Options for configuring the embedder behavior, such as l2_normalize
  // and quantize.
  struct EmbedderOptions embedder_options;
};

// Creates a TextEmbedder from the provided `options`.
// If successful, returns `kMpOk` and sets `*embedder` to the new
// `MpTextEmbedderPtr`.
MP_EXPORT MpStatus MpTextEmbedderCreate(struct TextEmbedderOptions* options,
                                        MpTextEmbedderPtr* embedder);

// Performs embedding extraction on the input `utf8_str`.
// If successful, returns `kMpOk` and sets `*result` to the new
// `TextEmbedderResult`.
MP_EXPORT MpStatus MpTextEmbedderEmbed(MpTextEmbedderPtr embedder,
                                       const char* utf8_str,
                                       TextEmbedderResult* result);

// Frees the memory allocated inside a TextEmbedderResult result. Does not
// free the result pointer itself.
MP_EXPORT void MpTextEmbedderCloseResult(TextEmbedderResult* result);

// Shuts down the TextEmbedder when all the work is done. Frees all memory.
MP_EXPORT MpStatus MpTextEmbedderClose(MpTextEmbedderPtr embedder);

// Utility function to compute cosine similarity [1] between two embeddings.
// Returns `kMpOk` on success, or an error status if e.g. the embeddings are
// of different types (quantized vs. float), have different sizes, or have a
// an L2-norm of 0.
//
// [1]: https://en.wikipedia.org/wiki/Cosine_similarity
MP_EXPORT MpStatus MpTextEmbedderCosSimilarity(const struct Embedding* u,
                                               const struct Embedding* v,
                                               double* similarity);

#ifdef __cplusplus
}  // extern C
#endif

#endif  // MEDIAPIPE_TASKS_C_TEXT_TEXT_EMBEDDER_TEXT_EMBEDDER_H_
