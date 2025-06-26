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

package com.google.mediapipe.tasks.text.textembedder;

import static com.google.common.truth.Truth.assertThat;
import static org.junit.Assert.assertThrows;

import androidx.test.core.app.ApplicationProvider;
import androidx.test.ext.junit.runners.AndroidJUnit4;
import com.google.mediapipe.framework.MediaPipeException;
import org.junit.Test;
import org.junit.runner.RunWith;

/** Test for {@link TextEmbedder}/ */
@RunWith(AndroidJUnit4.class)
public class TextEmbedderTest {
  private static final String BERT_MODEL_FILE = "mobilebert_embedding_with_metadata.tflite";
  private static final String REGEX_MODEL_FILE = "regex_one_embedding_with_metadata.tflite";
  private static final String USE_MODEL_FILE = "universal_sentence_encoder_qa_with_metadata.tflite";

  private static final double DOUBLE_DIFF_TOLERANCE = 1e-4;
  private static final float FLOAT_DIFF_TOLERANCE = 1e-4f;

  @Test
  public void create_failsWithMissingModel() throws Exception {
    String nonExistentFile = "/path/to/non/existent/file";
    MediaPipeException exception =
        assertThrows(
            MediaPipeException.class,
            () ->
                TextEmbedder.createFromFile(
                    ApplicationProvider.getApplicationContext(), nonExistentFile));
    assertThat(exception).hasMessageThat().contains(nonExistentFile);
  }

  @Test
  public void embed_succeedsWithBert() throws Exception {
    TextEmbedder textEmbedder =
        TextEmbedder.createFromFile(ApplicationProvider.getApplicationContext(), BERT_MODEL_FILE);

    TextEmbedderResult result0 = textEmbedder.embed("it's a charming and often affecting journey");
    assertThat(result0.embeddingResult().embeddings().size()).isEqualTo(1);
    assertThat(result0.embeddingResult().embeddings().get(0).floatEmbedding()).hasLength(512);
    assertThat(result0.embeddingResult().embeddings().get(0).floatEmbedding()[0])
        .isWithin(FLOAT_DIFF_TOLERANCE)
        .of(20.53943f);
    TextEmbedderResult result1 = textEmbedder.embed("what a great and fantastic trip");
    assertThat(result1.embeddingResult().embeddings().size()).isEqualTo(1);
    assertThat(result1.embeddingResult().embeddings().get(0).floatEmbedding()).hasLength(512);
    assertThat(result1.embeddingResult().embeddings().get(0).floatEmbedding()[0])
        .isWithin(FLOAT_DIFF_TOLERANCE)
        .of(20.150091f);

    // Check cosine similarity.
    double similarity =
        TextEmbedder.cosineSimilarity(
            result0.embeddingResult().embeddings().get(0),
            result1.embeddingResult().embeddings().get(0));
    assertThat(similarity).isWithin(DOUBLE_DIFF_TOLERANCE).of(0.9485555196137594);
  }

  @Test
  public void embed_succeedsWithUSE() throws Exception {
    TextEmbedder textEmbedder =
        TextEmbedder.createFromFile(ApplicationProvider.getApplicationContext(), USE_MODEL_FILE);

    TextEmbedderResult result0 = textEmbedder.embed("it's a charming and often affecting journey");
    assertThat(result0.embeddingResult().embeddings().size()).isEqualTo(1);
    assertThat(result0.embeddingResult().embeddings().get(0).floatEmbedding()).hasLength(100);
    assertThat(result0.embeddingResult().embeddings().get(0).floatEmbedding()[0])
        .isWithin(FLOAT_DIFF_TOLERANCE)
        .of(1.422951f);
    TextEmbedderResult result1 = textEmbedder.embed("what a great and fantastic trip");
    assertThat(result1.embeddingResult().embeddings().size()).isEqualTo(1);
    assertThat(result1.embeddingResult().embeddings().get(0).floatEmbedding()).hasLength(100);
    assertThat(result1.embeddingResult().embeddings().get(0).floatEmbedding()[0])
        .isWithin(FLOAT_DIFF_TOLERANCE)
        .of(1.404664f);

    // Check cosine similarity.
    double similarity =
        TextEmbedder.cosineSimilarity(
            result0.embeddingResult().embeddings().get(0),
            result1.embeddingResult().embeddings().get(0));
    assertThat(similarity).isWithin(DOUBLE_DIFF_TOLERANCE).of(0.851961);
  }

  @Test
  public void embed_succeedsWithRegex() throws Exception {
    TextEmbedder textEmbedder =
        TextEmbedder.createFromFile(ApplicationProvider.getApplicationContext(), REGEX_MODEL_FILE);

    TextEmbedderResult result0 = textEmbedder.embed("it's a charming and often affecting journey");
    assertThat(result0.embeddingResult().embeddings().size()).isEqualTo(1);
    assertThat(result0.embeddingResult().embeddings().get(0).floatEmbedding()).hasLength(16);
    assertThat(result0.embeddingResult().embeddings().get(0).floatEmbedding()[0])
        .isWithin(FLOAT_DIFF_TOLERANCE)
        .of(0.030935612f);
    TextEmbedderResult result1 = textEmbedder.embed("what a great and fantastic trip");
    assertThat(result1.embeddingResult().embeddings().size()).isEqualTo(1);
    assertThat(result1.embeddingResult().embeddings().get(0).floatEmbedding()).hasLength(16);
    assertThat(result1.embeddingResult().embeddings().get(0).floatEmbedding()[0])
        .isWithin(FLOAT_DIFF_TOLERANCE)
        .of(0.0312863f);

    // Check cosine similarity.
    double similarity =
        TextEmbedder.cosineSimilarity(
            result0.embeddingResult().embeddings().get(0),
            result1.embeddingResult().embeddings().get(0));
    assertThat(similarity).isWithin(DOUBLE_DIFF_TOLERANCE).of(0.999937);
  }

  @Test
  public void classify_succeedsWithBertAndDifferentThemes() throws Exception {
    TextEmbedder textEmbedder =
        TextEmbedder.createFromFile(ApplicationProvider.getApplicationContext(), BERT_MODEL_FILE);

    TextEmbedderResult result0 =
        textEmbedder.embed(
            "When you go to this restaurant, they hold the pancake upside-down before they hand "
                + "it to you. It's a great gimmick.");
    TextEmbedderResult result1 =
        textEmbedder.embed("Let\'s make a plan to steal the declaration of independence.'");

    // Check cosine similarity.
    double similarity =
        TextEmbedder.cosineSimilarity(
            result0.embeddingResult().embeddings().get(0),
            result1.embeddingResult().embeddings().get(0));
    assertThat(similarity).isWithin(DOUBLE_DIFF_TOLERANCE).of(0.35211632234930357);
  }

  @Test
  public void classify_succeedsWithUSEAndDifferentThemes() throws Exception {
    TextEmbedder textEmbedder =
        TextEmbedder.createFromFile(ApplicationProvider.getApplicationContext(), USE_MODEL_FILE);

    TextEmbedderResult result0 =
        textEmbedder.embed(
            "When you go to this restaurant, they hold the pancake upside-down before they hand "
                + "it to you. It's a great gimmick.");
    TextEmbedderResult result1 =
        textEmbedder.embed("Let\'s make a plan to steal the declaration of independence.'");

    // Check cosine similarity.
    double similarity =
        TextEmbedder.cosineSimilarity(
            result0.embeddingResult().embeddings().get(0),
            result1.embeddingResult().embeddings().get(0));
    assertThat(similarity).isWithin(DOUBLE_DIFF_TOLERANCE).of(0.7835510599396296);
  }
}
