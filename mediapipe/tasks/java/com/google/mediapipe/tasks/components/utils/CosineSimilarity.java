// Copyright 2022 The MediaPipe Authors.
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

package com.google.mediapipe.tasks.components.utils;

import com.google.mediapipe.tasks.components.containers.Embedding;

/** Utility class for computing cosine similarity between {@link Embedding} objects. */
public class CosineSimilarity {

  // Non-instantiable class.
  private CosineSimilarity() {}

  /**
   * Computes <a href="https://en.wikipedia.org/wiki/Cosine_similarity">cosine similarity</a>
   * between two {@link Embedding} objects.
   *
   * @throws IllegalArgumentException if the embeddings are of different types (float vs.
   *     quantized), have different sizes, or have an L2-norm of 0.
   */
  public static double compute(Embedding u, Embedding v) {
    if (u.floatEmbedding().length > 0 && v.floatEmbedding().length > 0) {
      return computeFloat(u.floatEmbedding(), v.floatEmbedding());
    }
    if (u.quantizedEmbedding().length > 0 && v.quantizedEmbedding().length > 0) {
      return computeQuantized(u.quantizedEmbedding(), v.quantizedEmbedding());
    }
    throw new IllegalArgumentException(
        "Cannot compute cosine similarity between quantized and float embeddings.");
  }

  private static double computeFloat(float[] u, float[] v) {
    if (u.length != v.length) {
      throw new IllegalArgumentException(
          String.format(
              "Cannot compute cosine similarity between embeddings of different sizes (%d vs."
                  + " %d).",
              u.length, v.length));
    }
    double dotProduct = 0.0;
    double normU = 0.0;
    double normV = 0.0;
    for (int i = 0; i < u.length; i++) {
      dotProduct += u[i] * v[i];
      normU += u[i] * u[i];
      normV += v[i] * v[i];
    }
    if (normU <= 0 || normV <= 0) {
      throw new IllegalArgumentException(
          "Cannot compute cosine similarity on embedding with 0 norm.");
    }
    return dotProduct / Math.sqrt(normU * normV);
  }

  private static double computeQuantized(byte[] u, byte[] v) {
    if (u.length != v.length) {
      throw new IllegalArgumentException(
          String.format(
              "Cannot compute cosine similarity between embeddings of different sizes (%d vs."
                  + " %d).",
              u.length, v.length));
    }
    double dotProduct = 0.0;
    double normU = 0.0;
    double normV = 0.0;
    for (int i = 0; i < u.length; i++) {
      dotProduct += u[i] * v[i];
      normU += u[i] * u[i];
      normV += v[i] * v[i];
    }
    if (normU <= 0 || normV <= 0) {
      throw new IllegalArgumentException(
          "Cannot compute cosine similarity on embedding with 0 norm.");
    }
    return dotProduct / Math.sqrt(normU * normV);
  }
}
