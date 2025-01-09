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

#ifndef MEDIAPIPE_TASKS_C_COMPONENTS_PROCESSORS_EMBEDDER_OPTIONS_H_
#define MEDIAPIPE_TASKS_C_COMPONENTS_PROCESSORS_EMBEDDER_OPTIONS_H_

#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

// Embedder options for MediaPipe C embedding extraction tasks.
struct EmbedderOptions {
  // Whether to normalize the returned feature vector with L2 norm. Use this
  // option only if the model does not already contain a native L2_NORMALIZATION
  // TF Lite Op. In most cases, this is already the case and L2 norm is thus
  // achieved through TF Lite inference.
  bool l2_normalize;

  // Whether the returned embedding should be quantized to bytes via scalar
  // quantization. Embeddings are implicitly assumed to be unit-norm and
  // therefore any dimension is guaranteed to have a value in [-1.0, 1.0]. Use
  // the l2_normalize option if this is not the case.
  bool quantize;
};

#ifdef __cplusplus
}  // extern C
#endif

#endif  // MEDIAPIPE_TASKS_C_COMPONENTS_PROCESSORS_EMBEDDER_OPTIONS_H_
