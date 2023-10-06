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

package com.google.mediapipe.tasks.text.textclassifier;

import static com.google.common.truth.Truth.assertThat;
import static org.junit.Assert.assertThrows;

import androidx.test.core.app.ApplicationProvider;
import androidx.test.ext.junit.runners.AndroidJUnit4;
import com.google.mediapipe.framework.MediaPipeException;
import com.google.mediapipe.tasks.components.containers.Category;
import com.google.mediapipe.tasks.core.BaseOptions;
import com.google.mediapipe.tasks.core.TestUtils;
import com.google.mediapipe.tasks.text.textclassifier.TextClassifier.TextClassifierOptions;
import java.util.Arrays;
import java.util.List;
import org.junit.Test;
import org.junit.runner.RunWith;

/** Test for {@link TextClassifier}/ */
@RunWith(AndroidJUnit4.class)
public class TextClassifierTest {
  private static final String BERT_MODEL_FILE = "bert_text_classifier.tflite";
  private static final String REGEX_MODEL_FILE =
      "test_model_text_classifier_with_regex_tokenizer.tflite";
  private static final String STRING_TO_BOOL_MODEL_FILE =
      "test_model_text_classifier_bool_output.tflite";
  private static final String NEGATIVE_TEXT = "unflinchingly bleak and desperate";
  private static final String POSITIVE_TEXT = "it's a charming and often affecting journey";

  @Test
  public void options_failsWithNegativeMaxResults() throws Exception {
    IllegalArgumentException exception =
        assertThrows(
            IllegalArgumentException.class,
            () ->
                TextClassifierOptions.builder()
                    .setBaseOptions(
                        BaseOptions.builder().setModelAssetPath(BERT_MODEL_FILE).build())
                    .setMaxResults(-1)
                    .build());
    assertThat(exception).hasMessageThat().contains("If specified, maxResults must be > 0");
  }

  @Test
  public void options_failsWithBothAllowlistAndDenylist() throws Exception {
    IllegalArgumentException exception =
        assertThrows(
            IllegalArgumentException.class,
            () ->
                TextClassifierOptions.builder()
                    .setBaseOptions(
                        BaseOptions.builder().setModelAssetPath(BERT_MODEL_FILE).build())
                    .setCategoryAllowlist(Arrays.asList("foo"))
                    .setCategoryDenylist(Arrays.asList("bar"))
                    .build());
    assertThat(exception)
        .hasMessageThat()
        .contains("Category allowlist and denylist are mutually exclusive");
  }

  @Test
  public void create_failsWithMissingModel() throws Exception {
    String nonExistentFile = "/path/to/non/existent/file";
    MediaPipeException exception =
        assertThrows(
            MediaPipeException.class,
            () ->
                TextClassifier.createFromFile(
                    ApplicationProvider.getApplicationContext(), nonExistentFile));
    assertThat(exception).hasMessageThat().contains(nonExistentFile);
  }

  @Test
  public void create_failsWithMissingOpResolver() throws Exception {
    TextClassifierOptions options =
        TextClassifierOptions.builder()
            .setBaseOptions(
                BaseOptions.builder().setModelAssetPath(STRING_TO_BOOL_MODEL_FILE).build())
            .build();
    MediaPipeException exception =
        assertThrows(
            MediaPipeException.class,
            () ->
                TextClassifier.createFromOptions(
                    ApplicationProvider.getApplicationContext(), options));
    // TODO: Make MediaPipe InferenceCalculator report the detailed.
    // interpreter errors (e.g., "Encountered unresolved custom op").
    assertThat(exception).hasMessageThat().contains("== kTfLiteOk");
  }

  @Test
  public void classify_succeedsWithBert() throws Exception {
    TextClassifier textClassifier =
        TextClassifier.createFromFile(ApplicationProvider.getApplicationContext(), BERT_MODEL_FILE);
    TextClassifierResult negativeResults = textClassifier.classify(NEGATIVE_TEXT);
    assertHasOneHead(negativeResults);
    assertCategoriesAre(
        negativeResults,
        Arrays.asList(
            Category.create(0.95630914f, 0, "negative", ""),
            Category.create(0.04369091f, 1, "positive", "")));

    TextClassifierResult positiveResults = textClassifier.classify(POSITIVE_TEXT);
    assertHasOneHead(positiveResults);
    assertCategoriesAre(
        positiveResults,
        Arrays.asList(
            Category.create(0.99997187f, 1, "positive", ""),
            Category.create(2.8132641E-5f, 0, "negative", "")));
  }

  @Test
  public void classify_succeedsWithFileObject() throws Exception {
    TextClassifier textClassifier =
        TextClassifier.createFromFile(
            ApplicationProvider.getApplicationContext(),
            TestUtils.loadFile(ApplicationProvider.getApplicationContext(), BERT_MODEL_FILE));
    TextClassifierResult negativeResults = textClassifier.classify(NEGATIVE_TEXT);
    assertHasOneHead(negativeResults);
    assertCategoriesAre(
        negativeResults,
        Arrays.asList(
            Category.create(0.95630914f, 0, "negative", ""),
            Category.create(0.04369091f, 1, "positive", "")));

    TextClassifierResult positiveResults = textClassifier.classify(POSITIVE_TEXT);
    assertHasOneHead(positiveResults);
    assertHasOneHead(positiveResults);
    assertCategoriesAre(
        positiveResults,
        Arrays.asList(
            Category.create(0.99997187f, 1, "positive", ""),
            Category.create(2.8132641E-5f, 0, "negative", "")));
  }

  @Test
  public void classify_succeedsWithRegex() throws Exception {
    TextClassifier textClassifier =
        TextClassifier.createFromFile(
            ApplicationProvider.getApplicationContext(), REGEX_MODEL_FILE);
    TextClassifierResult negativeResults = textClassifier.classify(NEGATIVE_TEXT);
    assertHasOneHead(negativeResults);
    assertCategoriesAre(
        negativeResults,
        Arrays.asList(
            Category.create(0.6647746f, 0, "Negative", ""),
            Category.create(0.33522537f, 1, "Positive", "")));

    TextClassifierResult positiveResults = textClassifier.classify(POSITIVE_TEXT);
    assertHasOneHead(positiveResults);
    assertCategoriesAre(
        positiveResults,
        Arrays.asList(
            Category.create(0.5120041f, 0, "Negative", ""),
            Category.create(0.48799595f, 1, "Positive", "")));
  }

  private static void assertHasOneHead(TextClassifierResult results) {
    assertThat(results.classificationResult().classifications()).hasSize(1);
    assertThat(results.classificationResult().classifications().get(0).headIndex()).isEqualTo(0);
    assertThat(results.classificationResult().classifications().get(0).headName().get())
        .isEqualTo("probability");
  }

  private static void assertCategoriesAre(TextClassifierResult results, List<Category> categories) {
    assertThat(results.classificationResult().classifications().get(0).categories())
        .isEqualTo(categories);
  }
}
