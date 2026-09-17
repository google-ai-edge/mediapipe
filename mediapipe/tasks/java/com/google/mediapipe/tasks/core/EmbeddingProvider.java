/* Copyright 2026 The MediaPipe Authors. All Rights Reserved.

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

package com.google.mediapipe.tasks.core;

import androidx.annotation.Nullable;
import java.util.List;

/**
 * Interface for generating embeddings across different modalities using content parts. Distributed
 * via mediapipe:tasks-core.
 */
public interface EmbeddingProvider {

  /**
   * Generates a high-dimensional vector embedding for the given list of multi-modal content parts.
   *
   * @param content the list of multi-modal content to embed.
   * @return the generated embedding vector, or {@code null} if the input content parts type is not
   *     supported by this embedder.
   */
  @Nullable
  float[] embedContent(List<Object> content);
}
