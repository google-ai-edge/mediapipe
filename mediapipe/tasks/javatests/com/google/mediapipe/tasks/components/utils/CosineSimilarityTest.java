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

import static com.google.common.truth.Truth.assertThat;
import static org.junit.Assert.assertThrows;

import androidx.test.ext.junit.runners.AndroidJUnit4;
import com.google.mediapipe.tasks.components.containers.Embedding;
import java.util.Optional;
import org.junit.Test;
import org.junit.runner.RunWith;

/** Tests for {@link CosineSimilarity}. */
@RunWith(AndroidJUnit4.class)
public final class CosineSimilarityTest {

  @Test
  public void failsWithQuantizedAndFloatEmbeddings() {
    Embedding u =
        Embedding.create(
            new float[] {1.0f}, new byte[0], /*headIndex=*/ 0, /*headName=*/ Optional.empty());
    Embedding v =
        Embedding.create(
            new float[0], new byte[] {1}, /*headIndex=*/ 0, /*headName=*/ Optional.empty());

    IllegalArgumentException exception =
        assertThrows(IllegalArgumentException.class, () -> CosineSimilarity.compute(u, v));
    assertThat(exception)
        .hasMessageThat()
        .contains("Cannot compute cosine similarity between quantized and float embeddings");
  }

  @Test
  public void failsWithZeroNorm() {
    Embedding u =
        Embedding.create(
            new float[] {0.0f}, new byte[0], /*headIndex=*/ 0, /*headName=*/ Optional.empty());

    IllegalArgumentException exception =
        assertThrows(IllegalArgumentException.class, () -> CosineSimilarity.compute(u, u));
    assertThat(exception)
        .hasMessageThat()
        .contains("Cannot compute cosine similarity on embedding with 0 norm");
  }

  @Test
  public void failsWithDifferentSizes() {
    Embedding u =
        Embedding.create(
            new float[] {1.0f, 2.0f},
            new byte[0],
            /*headIndex=*/ 0,
            /*headName=*/ Optional.empty());
    Embedding v =
        Embedding.create(
            new float[] {1.0f, 2.0f, 3.0f},
            new byte[0],
            /*headIndex=*/ 0,
            /*headName=*/ Optional.empty());

    IllegalArgumentException exception =
        assertThrows(IllegalArgumentException.class, () -> CosineSimilarity.compute(u, v));
    assertThat(exception)
        .hasMessageThat()
        .contains("Cannot compute cosine similarity between embeddings of different sizes");
  }

  @Test
  public void succeedsWithFloatEmbeddings() {
    Embedding u =
        Embedding.create(
            new float[] {1.0f, 0.0f, 0.0f, 0.0f},
            new byte[0],
            /*headIndex=*/ 0,
            /*headName=*/ Optional.empty());
    Embedding v =
        Embedding.create(
            new float[] {0.5f, 0.5f, 0.5f, 0.5f},
            new byte[0],
            /*headIndex=*/ 0,
            /*headName=*/ Optional.empty());

    assertThat(CosineSimilarity.compute(u, v)).isEqualTo(0.5);
  }

  @Test
  public void succeedsWithQuantizedEmbeddings() {
    Embedding u =
        Embedding.create(
            new float[0],
            new byte[] {127, 0, 0, 0},
            /*headIndex=*/ 0,
            /*headName=*/ Optional.empty());
    Embedding v =
        Embedding.create(
            new float[0],
            new byte[] {-128, 0, 0, 0},
            /*headIndex=*/ 0,
            /*headName=*/ Optional.empty());

    assertThat(CosineSimilarity.compute(u, v)).isEqualTo(-1.0);
  }
}
