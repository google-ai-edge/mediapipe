// Copyright 2022 The MediaPipe Authors. All Rights Reserved.
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
import androidx.test.core.app.ApplicationProvider;
import androidx.test.ext.junit.runners.AndroidJUnit4;
import com.google.common.truth.Correspondence;
import com.google.mediapipe.formats.proto.ClassificationProto;
import com.google.mediapipe.framework.MediaPipeException;
import com.google.mediapipe.framework.image.BitmapImageBuilder;
import com.google.mediapipe.framework.image.Image;
import com.google.mediapipe.tasks.components.containers.Category;
import com.google.mediapipe.tasks.components.containers.Landmark;
import com.google.mediapipe.tasks.components.containers.proto.LandmarksDetectionResultProto.LandmarksDetectionResult;
import com.google.mediapipe.tasks.core.BaseOptions;
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
  private static final String HAND_DETECTOR_MODEL_FILE = "palm_detection_full.tflite";
  private static final String HAND_LANDMARKER_MODEL_FILE = "hand_landmark_full.tflite";
  private static final String GESTURE_RECOGNIZER_MODEL_FILE =
      "cg_classifier_screen3d_landmark_features_nn_2022_08_04_base_simple_model.tflite";
  private static final String TWO_HANDS_IMAGE = "right_hands.jpg";
  private static final String THUMB_UP_IMAGE = "thumb_up.jpg";
  private static final String NO_HANDS_IMAGE = "cats_and_dogs.jpg";
  private static final String THUMB_UP_LANDMARKS = "thumb_up_landmarks.pb";
  private static final String TAG = "Gesture Recognizer Test";
  private static final String THUMB_UP_LABEL = "Thumb_Up";
  private static final int THUMB_UP_INDEX = 5;
  private static final float LANDMARKS_ERROR_TOLERANCE = 0.03f;
  private static final int IMAGE_WIDTH = 382;
  private static final int IMAGE_HEIGHT = 406;

  @RunWith(AndroidJUnit4.class)
  public static final class General extends GestureRecognizerTest {

    @Test
    public void recognize_successWithValidModels() throws Exception {
      GestureRecognizerOptions options =
          GestureRecognizerOptions.builder()
              .setBaseOptions(
                  BaseOptions.builder().setModelAssetPath(GESTURE_RECOGNIZER_MODEL_FILE).build())
              .setBaseOptionsHandDetector(
                  BaseOptions.builder().setModelAssetPath(HAND_DETECTOR_MODEL_FILE).build())
              .setBaseOptionsHandLandmarker(
                  BaseOptions.builder().setModelAssetPath(HAND_LANDMARKER_MODEL_FILE).build())
              .setBaseOptionsGestureRecognizer(
                  BaseOptions.builder().setModelAssetPath(GESTURE_RECOGNIZER_MODEL_FILE).build())
              .build();
      GestureRecognizer gestureRecognizer =
          GestureRecognizer.createFromOptions(ApplicationProvider.getApplicationContext(), options);
      GestureRecognitionResult actualResult =
          gestureRecognizer.recognize(getImageFromAsset(THUMB_UP_IMAGE));
      GestureRecognitionResult expectedResult =
          getExpectedGestureRecognitionResult(THUMB_UP_LANDMARKS, THUMB_UP_LABEL, THUMB_UP_INDEX);
      assertActualResultApproximatelyEqualsToExpectedResult(actualResult, expectedResult);
    }

    @Test
    public void recognize_successWithEmptyResult() throws Exception {
      GestureRecognizerOptions options =
          GestureRecognizerOptions.builder()
              .setBaseOptions(
                  BaseOptions.builder().setModelAssetPath(GESTURE_RECOGNIZER_MODEL_FILE).build())
              .setBaseOptionsHandDetector(
                  BaseOptions.builder().setModelAssetPath(HAND_DETECTOR_MODEL_FILE).build())
              .setBaseOptionsHandLandmarker(
                  BaseOptions.builder().setModelAssetPath(HAND_LANDMARKER_MODEL_FILE).build())
              .setBaseOptionsGestureRecognizer(
                  BaseOptions.builder().setModelAssetPath(GESTURE_RECOGNIZER_MODEL_FILE).build())
              .build();
      GestureRecognizer gestureRecognizer =
          GestureRecognizer.createFromOptions(ApplicationProvider.getApplicationContext(), options);
      GestureRecognitionResult actualResult =
          gestureRecognizer.recognize(getImageFromAsset(NO_HANDS_IMAGE));
      assertThat(actualResult.landmarks()).isEmpty();
      assertThat(actualResult.worldLandmarks()).isEmpty();
      assertThat(actualResult.handednesses()).isEmpty();
      assertThat(actualResult.gestures()).isEmpty();
    }

    @Test
    public void recognize_successWithMinGestureConfidence() throws Exception {
      GestureRecognizerOptions options =
          GestureRecognizerOptions.builder()
              .setBaseOptions(
                  BaseOptions.builder().setModelAssetPath(GESTURE_RECOGNIZER_MODEL_FILE).build())
              .setBaseOptionsHandDetector(
                  BaseOptions.builder().setModelAssetPath(HAND_DETECTOR_MODEL_FILE).build())
              .setBaseOptionsHandLandmarker(
                  BaseOptions.builder().setModelAssetPath(HAND_LANDMARKER_MODEL_FILE).build())
              .setBaseOptionsGestureRecognizer(
                  BaseOptions.builder().setModelAssetPath(GESTURE_RECOGNIZER_MODEL_FILE).build())
              // TODO update the confidence to be in range [0,1] after embedding model
              // and scoring calculator is integrated.
              .setMinGestureConfidence(3.0f)
              .build();
      GestureRecognizer gestureRecognizer =
          GestureRecognizer.createFromOptions(ApplicationProvider.getApplicationContext(), options);
      GestureRecognitionResult actualResult =
          gestureRecognizer.recognize(getImageFromAsset(THUMB_UP_IMAGE));
      GestureRecognitionResult expectedResult =
          getExpectedGestureRecognitionResult(THUMB_UP_LANDMARKS, THUMB_UP_LABEL, THUMB_UP_INDEX);
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
                  BaseOptions.builder().setModelAssetPath(GESTURE_RECOGNIZER_MODEL_FILE).build())
              .setBaseOptionsHandDetector(
                  BaseOptions.builder().setModelAssetPath(HAND_DETECTOR_MODEL_FILE).build())
              .setBaseOptionsHandLandmarker(
                  BaseOptions.builder().setModelAssetPath(HAND_LANDMARKER_MODEL_FILE).build())
              .setBaseOptionsGestureRecognizer(
                  BaseOptions.builder().setModelAssetPath(GESTURE_RECOGNIZER_MODEL_FILE).build())
              .setNumHands(2)
              .build();
      GestureRecognizer gestureRecognizer =
          GestureRecognizer.createFromOptions(ApplicationProvider.getApplicationContext(), options);
      GestureRecognitionResult actualResult =
          gestureRecognizer.recognize(getImageFromAsset(TWO_HANDS_IMAGE));
      assertThat(actualResult.handednesses()).hasSize(2);
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
                                .setModelAssetPath(GESTURE_RECOGNIZER_MODEL_FILE)
                                .build())
                        .setBaseOptionsHandDetector(
                            BaseOptions.builder()
                                .setModelAssetPath(HAND_DETECTOR_MODEL_FILE)
                                .build())
                        .setBaseOptionsHandLandmarker(
                            BaseOptions.builder()
                                .setModelAssetPath(HAND_LANDMARKER_MODEL_FILE)
                                .build())
                        .setBaseOptionsGestureRecognizer(
                            BaseOptions.builder()
                                .setModelAssetPath(GESTURE_RECOGNIZER_MODEL_FILE)
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
                            .setModelAssetPath(GESTURE_RECOGNIZER_MODEL_FILE)
                            .build())
                    .setBaseOptionsHandDetector(
                        BaseOptions.builder().setModelAssetPath(HAND_DETECTOR_MODEL_FILE).build())
                    .setBaseOptionsHandLandmarker(
                        BaseOptions.builder().setModelAssetPath(HAND_LANDMARKER_MODEL_FILE).build())
                    .setBaseOptionsGestureRecognizer(
                        BaseOptions.builder()
                            .setModelAssetPath(GESTURE_RECOGNIZER_MODEL_FILE)
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
                BaseOptions.builder().setModelAssetPath(GESTURE_RECOGNIZER_MODEL_FILE).build())
            .setBaseOptionsHandDetector(
                BaseOptions.builder().setModelAssetPath(HAND_DETECTOR_MODEL_FILE).build())
            .setBaseOptionsHandLandmarker(
                BaseOptions.builder().setModelAssetPath(HAND_LANDMARKER_MODEL_FILE).build())
            .setBaseOptionsGestureRecognizer(
                BaseOptions.builder().setModelAssetPath(GESTURE_RECOGNIZER_MODEL_FILE).build())
            .setRunningMode(RunningMode.IMAGE)
            .build();

    GestureRecognizer gestureRecognizer =
        GestureRecognizer.createFromOptions(ApplicationProvider.getApplicationContext(), options);
    MediaPipeException exception =
        assertThrows(
            MediaPipeException.class,
            () -> gestureRecognizer.recognizeForVideo(getImageFromAsset(THUMB_UP_IMAGE), 0));
    assertThat(exception).hasMessageThat().contains("not initialized with the video mode");
    exception =
        assertThrows(
            MediaPipeException.class,
            () -> gestureRecognizer.recognizeAsync(getImageFromAsset(THUMB_UP_IMAGE), 0));
    assertThat(exception).hasMessageThat().contains("not initialized with the live stream mode");
  }

  @Test
  public void recognize_failsWithCallingWrongApiInVideoMode() throws Exception {
    GestureRecognizerOptions options =
        GestureRecognizerOptions.builder()
            .setBaseOptions(
                BaseOptions.builder().setModelAssetPath(GESTURE_RECOGNIZER_MODEL_FILE).build())
            .setBaseOptionsHandDetector(
                BaseOptions.builder().setModelAssetPath(HAND_DETECTOR_MODEL_FILE).build())
            .setBaseOptionsHandLandmarker(
                BaseOptions.builder().setModelAssetPath(HAND_LANDMARKER_MODEL_FILE).build())
            .setBaseOptionsGestureRecognizer(
                BaseOptions.builder().setModelAssetPath(GESTURE_RECOGNIZER_MODEL_FILE).build())
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
            () -> gestureRecognizer.recognizeAsync(getImageFromAsset(THUMB_UP_IMAGE), 0));
    assertThat(exception).hasMessageThat().contains("not initialized with the live stream mode");
  }

  @Test
  public void recognize_failsWithCallingWrongApiInLiveSteamMode() throws Exception {
    GestureRecognizerOptions options =
        GestureRecognizerOptions.builder()
            .setBaseOptions(
                BaseOptions.builder().setModelAssetPath(GESTURE_RECOGNIZER_MODEL_FILE).build())
            .setBaseOptionsHandDetector(
                BaseOptions.builder().setModelAssetPath(HAND_DETECTOR_MODEL_FILE).build())
            .setBaseOptionsHandLandmarker(
                BaseOptions.builder().setModelAssetPath(HAND_LANDMARKER_MODEL_FILE).build())
            .setBaseOptionsGestureRecognizer(
                BaseOptions.builder().setModelAssetPath(GESTURE_RECOGNIZER_MODEL_FILE).build())
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
            () -> gestureRecognizer.recognizeForVideo(getImageFromAsset(THUMB_UP_IMAGE), 0));
    assertThat(exception).hasMessageThat().contains("not initialized with the video mode");
  }

  @Test
  public void recognize_successWithImageMode() throws Exception {
    GestureRecognizerOptions options =
        GestureRecognizerOptions.builder()
            .setBaseOptions(
                BaseOptions.builder().setModelAssetPath(GESTURE_RECOGNIZER_MODEL_FILE).build())
            .setBaseOptionsHandDetector(
                BaseOptions.builder().setModelAssetPath(HAND_DETECTOR_MODEL_FILE).build())
            .setBaseOptionsHandLandmarker(
                BaseOptions.builder().setModelAssetPath(HAND_LANDMARKER_MODEL_FILE).build())
            .setBaseOptionsGestureRecognizer(
                BaseOptions.builder().setModelAssetPath(GESTURE_RECOGNIZER_MODEL_FILE).build())
            .setRunningMode(RunningMode.IMAGE)
            .build();

    GestureRecognizer gestureRecognizer =
        GestureRecognizer.createFromOptions(ApplicationProvider.getApplicationContext(), options);
    GestureRecognitionResult actualResult =
        gestureRecognizer.recognize(getImageFromAsset(THUMB_UP_IMAGE));
    GestureRecognitionResult expectedResult =
        getExpectedGestureRecognitionResult(THUMB_UP_LANDMARKS, THUMB_UP_LABEL, THUMB_UP_INDEX);
    assertActualResultApproximatelyEqualsToExpectedResult(actualResult, expectedResult);
  }

  @Test
  public void recognize_successWithVideoMode() throws Exception {
    GestureRecognizerOptions options =
        GestureRecognizerOptions.builder()
            .setBaseOptions(
                BaseOptions.builder().setModelAssetPath(GESTURE_RECOGNIZER_MODEL_FILE).build())
            .setBaseOptionsHandDetector(
                BaseOptions.builder().setModelAssetPath(HAND_DETECTOR_MODEL_FILE).build())
            .setBaseOptionsHandLandmarker(
                BaseOptions.builder().setModelAssetPath(HAND_LANDMARKER_MODEL_FILE).build())
            .setBaseOptionsGestureRecognizer(
                BaseOptions.builder().setModelAssetPath(GESTURE_RECOGNIZER_MODEL_FILE).build())
            .setRunningMode(RunningMode.VIDEO)
            .build();
    GestureRecognizer gestureRecognizer =
        GestureRecognizer.createFromOptions(ApplicationProvider.getApplicationContext(), options);
    GestureRecognitionResult expectedResult =
        getExpectedGestureRecognitionResult(THUMB_UP_LANDMARKS, THUMB_UP_LABEL, THUMB_UP_INDEX);
    for (int i = 0; i < 3; i++) {
      GestureRecognitionResult actualResult =
          gestureRecognizer.recognizeForVideo(getImageFromAsset(THUMB_UP_IMAGE), i);
      assertActualResultApproximatelyEqualsToExpectedResult(actualResult, expectedResult);
    }
  }

  @Test
  public void recognize_failsWithOutOfOrderInputTimestamps() throws Exception {
    Image image = getImageFromAsset(THUMB_UP_IMAGE);
    GestureRecognitionResult expectedResult =
        getExpectedGestureRecognitionResult(THUMB_UP_LANDMARKS, THUMB_UP_LABEL, THUMB_UP_INDEX);
    GestureRecognizerOptions options =
        GestureRecognizerOptions.builder()
            .setBaseOptions(
                BaseOptions.builder().setModelAssetPath(GESTURE_RECOGNIZER_MODEL_FILE).build())
            .setBaseOptionsHandDetector(
                BaseOptions.builder().setModelAssetPath(HAND_DETECTOR_MODEL_FILE).build())
            .setBaseOptionsHandLandmarker(
                BaseOptions.builder().setModelAssetPath(HAND_LANDMARKER_MODEL_FILE).build())
            .setBaseOptionsGestureRecognizer(
                BaseOptions.builder().setModelAssetPath(GESTURE_RECOGNIZER_MODEL_FILE).build())
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
      gestureRecognizer.recognizeAsync(image, 1);
      MediaPipeException exception =
          assertThrows(MediaPipeException.class, () -> gestureRecognizer.recognizeAsync(image, 0));
      assertThat(exception)
          .hasMessageThat()
          .contains("having a smaller timestamp than the processed timestamp");
    }
  }

  @Test
  public void recognize_successWithLiveSteamMode() throws Exception {
    Image image = getImageFromAsset(THUMB_UP_IMAGE);
    GestureRecognitionResult expectedResult =
        getExpectedGestureRecognitionResult(THUMB_UP_LANDMARKS, THUMB_UP_LABEL, THUMB_UP_INDEX);
    GestureRecognizerOptions options =
        GestureRecognizerOptions.builder()
            .setBaseOptions(
                BaseOptions.builder().setModelAssetPath(GESTURE_RECOGNIZER_MODEL_FILE).build())
            .setBaseOptionsHandDetector(
                BaseOptions.builder().setModelAssetPath(HAND_DETECTOR_MODEL_FILE).build())
            .setBaseOptionsHandLandmarker(
                BaseOptions.builder().setModelAssetPath(HAND_LANDMARKER_MODEL_FILE).build())
            .setBaseOptionsGestureRecognizer(
                BaseOptions.builder().setModelAssetPath(GESTURE_RECOGNIZER_MODEL_FILE).build())
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
        gestureRecognizer.recognizeAsync(image, i);
      }
    }
  }

  private static Image getImageFromAsset(String filePath) throws Exception {
    AssetManager assetManager = ApplicationProvider.getApplicationContext().getAssets();
    InputStream istr = assetManager.open(filePath);
    return new BitmapImageBuilder(BitmapFactory.decodeStream(istr)).build();
  }

  private static GestureRecognitionResult getExpectedGestureRecognitionResult(
      String filePath, String gestureLabel, int gestureIndex) throws Exception {
    AssetManager assetManager = ApplicationProvider.getApplicationContext().getAssets();
    InputStream istr = assetManager.open(filePath);
    LandmarksDetectionResult landmarksDetectionResultProto =
        LandmarksDetectionResult.parser().parseFrom(istr);
    ClassificationProto.ClassificationList gesturesProto =
        ClassificationProto.ClassificationList.newBuilder()
            .addClassification(
                ClassificationProto.Classification.newBuilder()
                    .setLabel(gestureLabel)
                    .setIndex(gestureIndex))
            .build();
    return GestureRecognitionResult.create(
        Arrays.asList(landmarksDetectionResultProto.getLandmarks()),
        Arrays.asList(landmarksDetectionResultProto.getWorldLandmarks()),
        Arrays.asList(landmarksDetectionResultProto.getClassifications()),
        Arrays.asList(gesturesProto),
        /*timestampMs=*/ 0);
  }

  private static void assertActualResultApproximatelyEqualsToExpectedResult(
      GestureRecognitionResult actualResult, GestureRecognitionResult expectedResult) {
    // Expects to have the same number of hands detected.
    assertThat(actualResult.landmarks()).hasSize(expectedResult.landmarks().size());
    assertThat(actualResult.worldLandmarks()).hasSize(expectedResult.worldLandmarks().size());
    assertThat(actualResult.handednesses()).hasSize(expectedResult.handednesses().size());
    assertThat(actualResult.gestures()).hasSize(expectedResult.gestures().size());

    // Actual landmarks match expected landmarks.
    assertThat(actualResult.landmarks().get(0))
        .comparingElementsUsing(
            Correspondence.from(
                (Correspondence.BinaryPredicate<Landmark, Landmark>)
                    (actual, expected) -> {
                      return Correspondence.tolerance(LANDMARKS_ERROR_TOLERANCE)
                              .compare(actual.x(), expected.x())
                          && Correspondence.tolerance(LANDMARKS_ERROR_TOLERANCE)
                              .compare(actual.y(), expected.y());
                    },
                "landmarks approximately equal to"))
        .containsExactlyElementsIn(expectedResult.landmarks().get(0));

    // Actual handedness matches expected handedness.
    Category actualTopHandedness = actualResult.handednesses().get(0).get(0);
    Category expectedTopHandedness = expectedResult.handednesses().get(0).get(0);
    assertThat(actualTopHandedness.index()).isEqualTo(expectedTopHandedness.index());
    assertThat(actualTopHandedness.categoryName()).isEqualTo(expectedTopHandedness.categoryName());

    // Actual gesture with top score matches expected gesture.
    Category actualTopGesture = actualResult.gestures().get(0).get(0);
    Category expectedTopGesture = expectedResult.gestures().get(0).get(0);
    assertActualGestureEqualExpectedGesture(actualTopGesture, expectedTopGesture);
  }

  private static void assertActualGestureEqualExpectedGesture(
      Category actualGesture, Category expectedGesture) {
    assertThat(actualGesture.index()).isEqualTo(actualGesture.index());
    assertThat(expectedGesture.categoryName()).isEqualTo(expectedGesture.categoryName());
  }

  private static void assertImageSizeIsExpected(Image inputImage) {
    assertThat(inputImage).isNotNull();
    assertThat(inputImage.getWidth()).isEqualTo(IMAGE_WIDTH);
    assertThat(inputImage.getHeight()).isEqualTo(IMAGE_HEIGHT);
  }
}
