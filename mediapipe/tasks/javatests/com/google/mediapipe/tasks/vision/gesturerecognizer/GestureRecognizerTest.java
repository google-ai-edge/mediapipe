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

package com.google.mediapipe.tasks.vision.gesturerecognizer;

import static com.google.common.truth.Truth.assertThat;
import static org.junit.Assert.assertThrows;

import android.content.res.AssetManager;
import android.graphics.BitmapFactory;
import android.graphics.RectF;
import androidx.test.core.app.ApplicationProvider;
import androidx.test.ext.junit.runners.AndroidJUnit4;
import com.google.common.truth.Correspondence;
import com.google.mediapipe.formats.proto.ClassificationProto;
import com.google.mediapipe.framework.MediaPipeException;
import com.google.mediapipe.framework.image.BitmapImageBuilder;
import com.google.mediapipe.framework.image.MPImage;
import com.google.mediapipe.tasks.components.containers.Category;
import com.google.mediapipe.tasks.components.containers.NormalizedLandmark;
import com.google.mediapipe.tasks.components.containers.proto.LandmarksDetectionResultProto.LandmarksDetectionResult;
import com.google.mediapipe.tasks.components.processors.ClassifierOptions;
import com.google.mediapipe.tasks.core.BaseOptions;
import com.google.mediapipe.tasks.vision.core.ImageProcessingOptions;
import com.google.mediapipe.tasks.vision.core.RunningMode;
import com.google.mediapipe.tasks.vision.gesturerecognizer.GestureRecognizer.GestureRecognizerOptions;
import java.io.InputStream;
import java.util.Arrays;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Suite;
import org.junit.runners.Suite.SuiteClasses;

/** Test for {@link GestureRecognizer}. */
@RunWith(Suite.class)
@SuiteClasses({GestureRecognizerTest.General.class, GestureRecognizerTest.RunningModeTest.class})
public class GestureRecognizerTest {
  private static final String GESTURE_RECOGNIZER_BUNDLE_ASSET_FILE = "gesture_recognizer.task";
  private static final String GESTURE_RECOGNIZER_WITH_CUSTOM_CLASSIFIER_BUNDLE_ASSET_FILE =
      "gesture_recognizer_with_custom_classifier.task";
  private static final String TWO_HANDS_IMAGE = "right_hands.jpg";
  private static final String THUMB_UP_IMAGE = "thumb_up.jpg";
  private static final String POINTING_UP_ROTATED_IMAGE = "pointing_up_rotated.jpg";
  private static final String NO_HANDS_IMAGE = "cats_and_dogs.jpg";
  private static final String FIST_IMAGE = "fist.jpg";
  private static final String THUMB_UP_LANDMARKS = "thumb_up_landmarks.pb";
  private static final String FIST_LANDMARKS = "fist_landmarks.pb";
  private static final String TAG = "Gesture Recognizer Test";
  private static final String THUMB_UP_LABEL = "Thumb_Up";
  private static final String POINTING_UP_LABEL = "Pointing_Up";
  private static final String FIST_LABEL = "Closed_Fist";
  private static final String ROCK_LABEL = "Rock";
  private static final float LANDMARKS_ERROR_TOLERANCE = 0.03f;
  private static final int IMAGE_WIDTH = 382;
  private static final int IMAGE_HEIGHT = 406;
  private static final int GESTURE_EXPECTED_INDEX = -1;

  @RunWith(AndroidJUnit4.class)
  public static final class General extends GestureRecognizerTest {

    @Test
    public void recognize_successWithValidModels() throws Exception {
      GestureRecognizerOptions options =
          GestureRecognizerOptions.builder()
              .setBaseOptions(
                  BaseOptions.builder()
                      .setModelAssetPath(GESTURE_RECOGNIZER_BUNDLE_ASSET_FILE)
                      .build())
              .build();
      GestureRecognizer gestureRecognizer =
          GestureRecognizer.createFromOptions(ApplicationProvider.getApplicationContext(), options);
      GestureRecognizerResult actualResult =
          gestureRecognizer.recognize(getImageFromAsset(THUMB_UP_IMAGE));
      GestureRecognizerResult expectedResult =
          getExpectedGestureRecognizerResult(THUMB_UP_LANDMARKS, THUMB_UP_LABEL);
      assertActualResultApproximatelyEqualsToExpectedResult(actualResult, expectedResult);
    }

    @Test
    public void recognize_successWithEmptyResult() throws Exception {
      GestureRecognizerOptions options =
          GestureRecognizerOptions.builder()
              .setBaseOptions(
                  BaseOptions.builder()
                      .setModelAssetPath(GESTURE_RECOGNIZER_BUNDLE_ASSET_FILE)
                      .build())
              .build();
      GestureRecognizer gestureRecognizer =
          GestureRecognizer.createFromOptions(ApplicationProvider.getApplicationContext(), options);
      GestureRecognizerResult actualResult =
          gestureRecognizer.recognize(getImageFromAsset(NO_HANDS_IMAGE));
      assertThat(actualResult.landmarks()).isEmpty();
      assertThat(actualResult.worldLandmarks()).isEmpty();
      assertThat(actualResult.handedness()).isEmpty();
      assertThat(actualResult.gestures()).isEmpty();
    }

    @Test
    public void recognize_successWithScoreThreshold() throws Exception {
      GestureRecognizerOptions options =
          GestureRecognizerOptions.builder()
              .setBaseOptions(
                  BaseOptions.builder()
                      .setModelAssetPath(GESTURE_RECOGNIZER_BUNDLE_ASSET_FILE)
                      .build())
              .setCannedGesturesClassifierOptions(
                  ClassifierOptions.builder().setScoreThreshold(0.5f).build())
              .build();
      GestureRecognizer gestureRecognizer =
          GestureRecognizer.createFromOptions(ApplicationProvider.getApplicationContext(), options);
      GestureRecognizerResult actualResult =
          gestureRecognizer.recognize(getImageFromAsset(THUMB_UP_IMAGE));
      GestureRecognizerResult expectedResult =
          getExpectedGestureRecognizerResult(THUMB_UP_LANDMARKS, THUMB_UP_LABEL);
      // Only contains one top scoring gesture.
      assertThat(actualResult.gestures().get(0)).hasSize(1);
      assertActualGestureEqualExpectedGesture(
          actualResult.gestures().get(0).get(0), expectedResult.gestures().get(0).get(0));
    }

    @Test
    public void recognize_successWithNumHands() throws Exception {
      GestureRecognizerOptions options =
          GestureRecognizerOptions.builder()
              .setBaseOptions(
                  BaseOptions.builder()
                      .setModelAssetPath(GESTURE_RECOGNIZER_BUNDLE_ASSET_FILE)
                      .build())
              .setNumHands(2)
              .build();
      GestureRecognizer gestureRecognizer =
          GestureRecognizer.createFromOptions(ApplicationProvider.getApplicationContext(), options);
      GestureRecognizerResult actualResult =
          gestureRecognizer.recognize(getImageFromAsset(TWO_HANDS_IMAGE));
      assertThat(actualResult.handedness()).hasSize(2);
    }

    @Test
    public void recognize_successWithRotation() throws Exception {
      GestureRecognizerOptions options =
          GestureRecognizerOptions.builder()
              .setBaseOptions(
                  BaseOptions.builder()
                      .setModelAssetPath(GESTURE_RECOGNIZER_BUNDLE_ASSET_FILE)
                      .build())
              .setNumHands(1)
              .build();
      GestureRecognizer gestureRecognizer =
          GestureRecognizer.createFromOptions(ApplicationProvider.getApplicationContext(), options);
      ImageProcessingOptions imageProcessingOptions =
          ImageProcessingOptions.builder().setRotationDegrees(-90).build();
      GestureRecognizerResult actualResult =
          gestureRecognizer.recognize(
              getImageFromAsset(POINTING_UP_ROTATED_IMAGE), imageProcessingOptions);
      assertThat(actualResult.gestures()).hasSize(1);
      assertThat(actualResult.gestures().get(0).get(0).categoryName()).isEqualTo(POINTING_UP_LABEL);
    }

    @Test
    public void recognize_successWithCannedGestureFist() throws Exception {
      GestureRecognizerOptions options =
          GestureRecognizerOptions.builder()
              .setBaseOptions(
                  BaseOptions.builder()
                      .setModelAssetPath(GESTURE_RECOGNIZER_BUNDLE_ASSET_FILE)
                      .build())
              .setNumHands(1)
              .build();
      GestureRecognizer gestureRecognizer =
          GestureRecognizer.createFromOptions(ApplicationProvider.getApplicationContext(), options);
      GestureRecognizerResult actualResult =
          gestureRecognizer.recognize(getImageFromAsset(FIST_IMAGE));
      GestureRecognizerResult expectedResult =
          getExpectedGestureRecognizerResult(FIST_LANDMARKS, FIST_LABEL);
      assertActualResultApproximatelyEqualsToExpectedResult(actualResult, expectedResult);
    }

    @Test
    public void recognize_successWithCustomGestureRock() throws Exception {
      GestureRecognizerOptions options =
          GestureRecognizerOptions.builder()
              .setBaseOptions(
                  BaseOptions.builder()
                      .setModelAssetPath(
                          GESTURE_RECOGNIZER_WITH_CUSTOM_CLASSIFIER_BUNDLE_ASSET_FILE)
                      .build())
              .setNumHands(1)
              .build();
      GestureRecognizer gestureRecognizer =
          GestureRecognizer.createFromOptions(ApplicationProvider.getApplicationContext(), options);
      GestureRecognizerResult actualResult =
          gestureRecognizer.recognize(getImageFromAsset(FIST_IMAGE));
      GestureRecognizerResult expectedResult =
          getExpectedGestureRecognizerResult(FIST_LANDMARKS, ROCK_LABEL);
      assertActualResultApproximatelyEqualsToExpectedResult(actualResult, expectedResult);
    }

    @Test
    public void recognize_successWithAllowGestureFist() throws Exception {
      GestureRecognizerOptions options =
          GestureRecognizerOptions.builder()
              .setBaseOptions(
                  BaseOptions.builder()
                      .setModelAssetPath(GESTURE_RECOGNIZER_BUNDLE_ASSET_FILE)
                      .build())
              .setNumHands(1)
              .setCannedGesturesClassifierOptions(
                  ClassifierOptions.builder()
                      .setScoreThreshold(0.5f)
                      .setCategoryAllowlist(Arrays.asList("Closed_Fist"))
                      .build())
              .build();
      GestureRecognizer gestureRecognizer =
          GestureRecognizer.createFromOptions(ApplicationProvider.getApplicationContext(), options);
      GestureRecognizerResult actualResult =
          gestureRecognizer.recognize(getImageFromAsset(FIST_IMAGE));
      GestureRecognizerResult expectedResult =
          getExpectedGestureRecognizerResult(FIST_LANDMARKS, FIST_LABEL);
      assertActualResultApproximatelyEqualsToExpectedResult(actualResult, expectedResult);
    }

    @Test
    public void recognize_successWithDenyGestureFist() throws Exception {
      GestureRecognizerOptions options =
          GestureRecognizerOptions.builder()
              .setBaseOptions(
                  BaseOptions.builder()
                      .setModelAssetPath(GESTURE_RECOGNIZER_BUNDLE_ASSET_FILE)
                      .build())
              .setNumHands(1)
              .setCannedGesturesClassifierOptions(
                  ClassifierOptions.builder()
                      .setScoreThreshold(0.5f)
                      .setCategoryDenylist(Arrays.asList("Closed_Fist"))
                      .build())
              .build();
      GestureRecognizer gestureRecognizer =
          GestureRecognizer.createFromOptions(ApplicationProvider.getApplicationContext(), options);
      GestureRecognizerResult actualResult =
          gestureRecognizer.recognize(getImageFromAsset(FIST_IMAGE));
      assertThat(actualResult.landmarks()).isEmpty();
      assertThat(actualResult.worldLandmarks()).isEmpty();
      assertThat(actualResult.handedness()).isEmpty();
      assertThat(actualResult.gestures()).isEmpty();
    }

    @Test
    public void recognize_successWithAllowAllGestureExceptFist() throws Exception {
      GestureRecognizerOptions options =
          GestureRecognizerOptions.builder()
              .setBaseOptions(
                  BaseOptions.builder()
                      .setModelAssetPath(GESTURE_RECOGNIZER_BUNDLE_ASSET_FILE)
                      .build())
              .setNumHands(1)
              .setCannedGesturesClassifierOptions(
                  ClassifierOptions.builder()
                      .setScoreThreshold(0.5f)
                      .setCategoryAllowlist(
                          Arrays.asList(
                              "None",
                              "Open_Palm",
                              "Pointing_Up",
                              "Thumb_Down",
                              "Thumb_Up",
                              "Victory",
                              "ILoveYou"))
                      .build())
              .build();
      GestureRecognizer gestureRecognizer =
          GestureRecognizer.createFromOptions(ApplicationProvider.getApplicationContext(), options);
      GestureRecognizerResult actualResult =
          gestureRecognizer.recognize(getImageFromAsset(FIST_IMAGE));
      assertThat(actualResult.landmarks()).isEmpty();
      assertThat(actualResult.worldLandmarks()).isEmpty();
      assertThat(actualResult.handedness()).isEmpty();
      assertThat(actualResult.gestures()).isEmpty();
    }

    @Test
    public void recognize_successWithPreferAllowListThanDenyList() throws Exception {
      GestureRecognizerOptions options =
          GestureRecognizerOptions.builder()
              .setBaseOptions(
                  BaseOptions.builder()
                      .setModelAssetPath(GESTURE_RECOGNIZER_BUNDLE_ASSET_FILE)
                      .build())
              .setNumHands(1)
              .setCannedGesturesClassifierOptions(
                  ClassifierOptions.builder()
                      .setScoreThreshold(0.5f)
                      .setCategoryAllowlist(Arrays.asList("Closed_Fist"))
                      .setCategoryDenylist(Arrays.asList("Closed_Fist"))
                      .build())
              .build();
      GestureRecognizer gestureRecognizer =
          GestureRecognizer.createFromOptions(ApplicationProvider.getApplicationContext(), options);
      GestureRecognizerResult actualResult =
          gestureRecognizer.recognize(getImageFromAsset(FIST_IMAGE));
      GestureRecognizerResult expectedResult =
          getExpectedGestureRecognizerResult(FIST_LANDMARKS, FIST_LABEL);
      assertActualResultApproximatelyEqualsToExpectedResult(actualResult, expectedResult);
    }

    @Test
    public void recognize_failsWithRegionOfInterest() throws Exception {
      GestureRecognizerOptions options =
          GestureRecognizerOptions.builder()
              .setBaseOptions(
                  BaseOptions.builder()
                      .setModelAssetPath(GESTURE_RECOGNIZER_BUNDLE_ASSET_FILE)
                      .build())
              .setNumHands(1)
              .build();
      GestureRecognizer gestureRecognizer =
          GestureRecognizer.createFromOptions(ApplicationProvider.getApplicationContext(), options);
      ImageProcessingOptions imageProcessingOptions =
          ImageProcessingOptions.builder().setRegionOfInterest(new RectF(0, 0, 1, 1)).build();
      IllegalArgumentException exception =
          assertThrows(
              IllegalArgumentException.class,
              () ->
                  gestureRecognizer.recognize(
                      getImageFromAsset(THUMB_UP_IMAGE), imageProcessingOptions));
      assertThat(exception)
          .hasMessageThat()
          .contains("GestureRecognizer doesn't support region-of-interest");
    }
  }

  @RunWith(AndroidJUnit4.class)
  public static final class RunningModeTest extends GestureRecognizerTest {
    @Test
    public void create_failsWithIllegalResultListenerInNonLiveStreamMode() throws Exception {
      for (RunningMode mode : new RunningMode[] {RunningMode.IMAGE, RunningMode.VIDEO}) {
        IllegalArgumentException exception =
            assertThrows(
                IllegalArgumentException.class,
                () ->
                    GestureRecognizerOptions.builder()
                        .setBaseOptions(
                            BaseOptions.builder()
                                .setModelAssetPath(GESTURE_RECOGNIZER_BUNDLE_ASSET_FILE)
                                .build())
                        .setRunningMode(mode)
                        .setResultListener((gestureRecognitionResult, inputImage) -> {})
                        .build());
        assertThat(exception)
            .hasMessageThat()
            .contains("a user-defined result listener shouldn't be provided");
      }
    }
  }

  @Test
  public void create_failsWithMissingResultListenerInLiveSteamMode() throws Exception {
    IllegalArgumentException exception =
        assertThrows(
            IllegalArgumentException.class,
            () ->
                GestureRecognizerOptions.builder()
                    .setBaseOptions(
                        BaseOptions.builder()
                            .setModelAssetPath(GESTURE_RECOGNIZER_BUNDLE_ASSET_FILE)
                            .build())
                    .setRunningMode(RunningMode.LIVE_STREAM)
                    .build());
    assertThat(exception)
        .hasMessageThat()
        .contains("a user-defined result listener must be provided");
  }

  @Test
  public void recognize_failsWithCallingWrongApiInImageMode() throws Exception {
    GestureRecognizerOptions options =
        GestureRecognizerOptions.builder()
            .setBaseOptions(
                BaseOptions.builder()
                    .setModelAssetPath(GESTURE_RECOGNIZER_BUNDLE_ASSET_FILE)
                    .build())
            .setRunningMode(RunningMode.IMAGE)
            .build();

    GestureRecognizer gestureRecognizer =
        GestureRecognizer.createFromOptions(ApplicationProvider.getApplicationContext(), options);
    MediaPipeException exception =
        assertThrows(
            MediaPipeException.class,
            () ->
                gestureRecognizer.recognizeForVideo(
                    getImageFromAsset(THUMB_UP_IMAGE), /*timestampsMs=*/ 0));
    assertThat(exception).hasMessageThat().contains("not initialized with the video mode");
    exception =
        assertThrows(
            MediaPipeException.class,
            () ->
                gestureRecognizer.recognizeAsync(
                    getImageFromAsset(THUMB_UP_IMAGE), /*timestampsMs=*/ 0));
    assertThat(exception).hasMessageThat().contains("not initialized with the live stream mode");
  }

  @Test
  public void recognize_failsWithCallingWrongApiInVideoMode() throws Exception {
    GestureRecognizerOptions options =
        GestureRecognizerOptions.builder()
            .setBaseOptions(
                BaseOptions.builder()
                    .setModelAssetPath(GESTURE_RECOGNIZER_BUNDLE_ASSET_FILE)
                    .build())
            .setRunningMode(RunningMode.VIDEO)
            .build();

    GestureRecognizer gestureRecognizer =
        GestureRecognizer.createFromOptions(ApplicationProvider.getApplicationContext(), options);
    MediaPipeException exception =
        assertThrows(
            MediaPipeException.class,
            () -> gestureRecognizer.recognize(getImageFromAsset(THUMB_UP_IMAGE)));
    assertThat(exception).hasMessageThat().contains("not initialized with the image mode");
    exception =
        assertThrows(
            MediaPipeException.class,
            () ->
                gestureRecognizer.recognizeAsync(
                    getImageFromAsset(THUMB_UP_IMAGE), /*timestampsMs=*/ 0));
    assertThat(exception).hasMessageThat().contains("not initialized with the live stream mode");
  }

  @Test
  public void recognize_failsWithCallingWrongApiInLiveSteamMode() throws Exception {
    GestureRecognizerOptions options =
        GestureRecognizerOptions.builder()
            .setBaseOptions(
                BaseOptions.builder()
                    .setModelAssetPath(GESTURE_RECOGNIZER_BUNDLE_ASSET_FILE)
                    .build())
            .setRunningMode(RunningMode.LIVE_STREAM)
            .setResultListener((gestureRecognitionResult, inputImage) -> {})
            .build();

    GestureRecognizer gestureRecognizer =
        GestureRecognizer.createFromOptions(ApplicationProvider.getApplicationContext(), options);
    MediaPipeException exception =
        assertThrows(
            MediaPipeException.class,
            () -> gestureRecognizer.recognize(getImageFromAsset(THUMB_UP_IMAGE)));
    assertThat(exception).hasMessageThat().contains("not initialized with the image mode");
    exception =
        assertThrows(
            MediaPipeException.class,
            () ->
                gestureRecognizer.recognizeForVideo(
                    getImageFromAsset(THUMB_UP_IMAGE), /*timestampsMs=*/ 0));
    assertThat(exception).hasMessageThat().contains("not initialized with the video mode");
  }

  @Test
  public void recognize_successWithImageMode() throws Exception {
    GestureRecognizerOptions options =
        GestureRecognizerOptions.builder()
            .setBaseOptions(
                BaseOptions.builder()
                    .setModelAssetPath(GESTURE_RECOGNIZER_BUNDLE_ASSET_FILE)
                    .build())
            .setRunningMode(RunningMode.IMAGE)
            .build();

    GestureRecognizer gestureRecognizer =
        GestureRecognizer.createFromOptions(ApplicationProvider.getApplicationContext(), options);
    GestureRecognizerResult actualResult =
        gestureRecognizer.recognize(getImageFromAsset(THUMB_UP_IMAGE));
    GestureRecognizerResult expectedResult =
        getExpectedGestureRecognizerResult(THUMB_UP_LANDMARKS, THUMB_UP_LABEL);
    assertActualResultApproximatelyEqualsToExpectedResult(actualResult, expectedResult);
  }

  @Test
  public void recognize_successWithVideoMode() throws Exception {
    GestureRecognizerOptions options =
        GestureRecognizerOptions.builder()
            .setBaseOptions(
                BaseOptions.builder()
                    .setModelAssetPath(GESTURE_RECOGNIZER_BUNDLE_ASSET_FILE)
                    .build())
            .setRunningMode(RunningMode.VIDEO)
            .build();
    GestureRecognizer gestureRecognizer =
        GestureRecognizer.createFromOptions(ApplicationProvider.getApplicationContext(), options);
    GestureRecognizerResult expectedResult =
        getExpectedGestureRecognizerResult(THUMB_UP_LANDMARKS, THUMB_UP_LABEL);
    for (int i = 0; i < 3; i++) {
      GestureRecognizerResult actualResult =
          gestureRecognizer.recognizeForVideo(
              getImageFromAsset(THUMB_UP_IMAGE), /*timestampsMs=*/ i);
      assertActualResultApproximatelyEqualsToExpectedResult(actualResult, expectedResult);
    }
  }

  @Test
  public void recognize_failsWithOutOfOrderInputTimestamps() throws Exception {
    MPImage image = getImageFromAsset(THUMB_UP_IMAGE);
    GestureRecognizerResult expectedResult =
        getExpectedGestureRecognizerResult(THUMB_UP_LANDMARKS, THUMB_UP_LABEL);
    GestureRecognizerOptions options =
        GestureRecognizerOptions.builder()
            .setBaseOptions(
                BaseOptions.builder()
                    .setModelAssetPath(GESTURE_RECOGNIZER_BUNDLE_ASSET_FILE)
                    .build())
            .setRunningMode(RunningMode.LIVE_STREAM)
            .setResultListener(
                (actualResult, inputImage) -> {
                  assertActualResultApproximatelyEqualsToExpectedResult(
                      actualResult, expectedResult);
                  assertImageSizeIsExpected(inputImage);
                })
            .build();
    try (GestureRecognizer gestureRecognizer =
        GestureRecognizer.createFromOptions(ApplicationProvider.getApplicationContext(), options)) {
      gestureRecognizer.recognizeAsync(image, /*timestampsMs=*/ 1);
      MediaPipeException exception =
          assertThrows(
              MediaPipeException.class,
              () -> gestureRecognizer.recognizeAsync(image, /*timestampsMs=*/ 0));
      assertThat(exception)
          .hasMessageThat()
          .contains("having a smaller timestamp than the processed timestamp");
    }
  }

  @Test
  public void recognize_successWithLiveSteamMode() throws Exception {
    MPImage image = getImageFromAsset(THUMB_UP_IMAGE);
    GestureRecognizerResult expectedResult =
        getExpectedGestureRecognizerResult(THUMB_UP_LANDMARKS, THUMB_UP_LABEL);
    GestureRecognizerOptions options =
        GestureRecognizerOptions.builder()
            .setBaseOptions(
                BaseOptions.builder()
                    .setModelAssetPath(GESTURE_RECOGNIZER_BUNDLE_ASSET_FILE)
                    .build())
            .setRunningMode(RunningMode.LIVE_STREAM)
            .setResultListener(
                (actualResult, inputImage) -> {
                  assertActualResultApproximatelyEqualsToExpectedResult(
                      actualResult, expectedResult);
                  assertImageSizeIsExpected(inputImage);
                })
            .build();
    try (GestureRecognizer gestureRecognizer =
        GestureRecognizer.createFromOptions(ApplicationProvider.getApplicationContext(), options)) {
      for (int i = 0; i < 3; i++) {
        gestureRecognizer.recognizeAsync(image, /*timestampsMs=*/ i);
      }
    }
  }

  private static MPImage getImageFromAsset(String filePath) throws Exception {
    AssetManager assetManager = ApplicationProvider.getApplicationContext().getAssets();
    InputStream istr = assetManager.open(filePath);
    return new BitmapImageBuilder(BitmapFactory.decodeStream(istr)).build();
  }

  private static GestureRecognizerResult getExpectedGestureRecognizerResult(
      String filePath, String gestureLabel) throws Exception {
    AssetManager assetManager = ApplicationProvider.getApplicationContext().getAssets();
    InputStream istr = assetManager.open(filePath);
    LandmarksDetectionResult landmarksDetectionResultProto =
        LandmarksDetectionResult.parser().parseFrom(istr);
    ClassificationProto.ClassificationList gesturesProto =
        ClassificationProto.ClassificationList.newBuilder()
            .addClassification(
                ClassificationProto.Classification.newBuilder().setLabel(gestureLabel))
            .build();
    return GestureRecognizerResult.create(
        Arrays.asList(landmarksDetectionResultProto.getLandmarks()),
        Arrays.asList(landmarksDetectionResultProto.getWorldLandmarks()),
        Arrays.asList(landmarksDetectionResultProto.getClassifications()),
        Arrays.asList(gesturesProto),
        /*timestampMs=*/ 0);
  }

  private static void assertActualResultApproximatelyEqualsToExpectedResult(
      GestureRecognizerResult actualResult, GestureRecognizerResult expectedResult) {
    // Expects to have the same number of hands detected.
    assertThat(actualResult.landmarks()).hasSize(expectedResult.landmarks().size());
    assertThat(actualResult.worldLandmarks()).hasSize(expectedResult.worldLandmarks().size());
    assertThat(actualResult.handedness()).hasSize(expectedResult.handedness().size());
    assertThat(actualResult.gestures()).hasSize(expectedResult.gestures().size());

    // Actual landmarks match expected landmarks.
    assertThat(actualResult.landmarks().get(0))
        .comparingElementsUsing(
            Correspondence.from(
                (Correspondence.BinaryPredicate<NormalizedLandmark, NormalizedLandmark>)
                    (actual, expected) -> {
                      return Correspondence.tolerance(LANDMARKS_ERROR_TOLERANCE)
                              .compare(actual.x(), expected.x())
                          && Correspondence.tolerance(LANDMARKS_ERROR_TOLERANCE)
                              .compare(actual.y(), expected.y());
                    },
                "landmarks approximately equal to"))
        .containsExactlyElementsIn(expectedResult.landmarks().get(0));

    // Actual handedness matches expected handedness.
    Category actualTopHandedness = actualResult.handedness().get(0).get(0);
    Category expectedTopHandedness = expectedResult.handedness().get(0).get(0);
    assertThat(actualTopHandedness.index()).isEqualTo(expectedTopHandedness.index());
    assertThat(actualTopHandedness.categoryName()).isEqualTo(expectedTopHandedness.categoryName());

    // Actual gesture with top score matches expected gesture.
    Category actualTopGesture = actualResult.gestures().get(0).get(0);
    Category expectedTopGesture = expectedResult.gestures().get(0).get(0);
    assertActualGestureEqualExpectedGesture(actualTopGesture, expectedTopGesture);
  }

  private static void assertActualGestureEqualExpectedGesture(
      Category actualGesture, Category expectedGesture) {
    assertThat(actualGesture.categoryName()).isEqualTo(expectedGesture.categoryName());
    assertThat(actualGesture.index()).isEqualTo(GESTURE_EXPECTED_INDEX);
  }

  private static void assertImageSizeIsExpected(MPImage inputImage) {
    assertThat(inputImage).isNotNull();
    assertThat(inputImage.getWidth()).isEqualTo(IMAGE_WIDTH);
    assertThat(inputImage.getHeight()).isEqualTo(IMAGE_HEIGHT);
  }
}
