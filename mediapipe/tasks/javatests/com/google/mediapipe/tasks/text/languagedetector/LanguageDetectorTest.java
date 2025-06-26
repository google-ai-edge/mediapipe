// Copyright 2023 The MediaPipe Authors.
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

package com.google.mediapipe.tasks.text.languagedetector;

import static com.google.common.truth.Truth.assertThat;
import static org.junit.Assert.assertThrows;

import androidx.test.core.app.ApplicationProvider;
import androidx.test.ext.junit.runners.AndroidJUnit4;
import com.google.mediapipe.framework.MediaPipeException;
import com.google.mediapipe.tasks.core.BaseOptions;
import com.google.mediapipe.tasks.core.TestUtils;
import com.google.mediapipe.tasks.text.languagedetector.LanguageDetector.LanguageDetectorOptions;
import java.util.Arrays;
import org.junit.Test;
import org.junit.runner.RunWith;

/** Test for {@link LanguageDetector}/ */
@RunWith(AndroidJUnit4.class)
public class LanguageDetectorTest {
  private static final String MODEL_FILE = "language_detector.tflite";

  @Test
  public void options_failsWithNegativeMaxResults() throws Exception {
    IllegalArgumentException exception =
        assertThrows(
            IllegalArgumentException.class,
            () ->
                LanguageDetectorOptions.builder()
                    .setBaseOptions(BaseOptions.builder().setModelAssetPath(MODEL_FILE).build())
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
                LanguageDetectorOptions.builder()
                    .setBaseOptions(BaseOptions.builder().setModelAssetPath(MODEL_FILE).build())
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
                LanguageDetector.createFromFile(
                    ApplicationProvider.getApplicationContext(), nonExistentFile));
    assertThat(exception).hasMessageThat().contains(nonExistentFile);
  }

  @Test
  public void detect_succeedsWithL2CModel() throws Exception {
    LanguageDetector languageDetector =
        LanguageDetector.createFromFile(ApplicationProvider.getApplicationContext(), MODEL_FILE);
    LanguageDetectorResult enResult =
        languageDetector.detect("To be, or not to be, that is the question");
    assertThat(enResult.languagesAndScores().size()).isEqualTo(1);
    assertThat(enResult.languagesAndScores().get(0))
        .isEqualTo(LanguagePrediction.create("en", 0.999856f));
    LanguageDetectorResult frResult =
        languageDetector.detect(
            "Il y a beaucoup de bouches qui parlent et fort peu de têtes qui pensent.");
    assertThat(frResult.languagesAndScores().size()).isEqualTo(1);
    assertThat(frResult.languagesAndScores().get(0))
        .isEqualTo(LanguagePrediction.create("fr", 0.9997813f));
    LanguageDetectorResult ruResult = languageDetector.detect("это какой-то английский язык");
    assertThat(ruResult.languagesAndScores().size()).isEqualTo(1);
    assertThat(ruResult.languagesAndScores().get(0))
        .isEqualTo(LanguagePrediction.create("ru", 0.9933616f));
  }

  @Test
  public void detect_succeedsWithFileObject() throws Exception {
    LanguageDetector languageDetector =
        LanguageDetector.createFromFile(
            ApplicationProvider.getApplicationContext(),
            TestUtils.loadFile(ApplicationProvider.getApplicationContext(), MODEL_FILE));
    LanguageDetectorResult mixedResult = languageDetector.detect("分久必合合久必分");
    assertThat(mixedResult.languagesAndScores().size()).isEqualTo(2);
    assertThat(mixedResult.languagesAndScores().get(0))
        .isEqualTo(LanguagePrediction.create("zh", 0.50542367f));
    assertThat(mixedResult.languagesAndScores().get(1))
        .isEqualTo(LanguagePrediction.create("ja", 0.48161677f));
  }
}
