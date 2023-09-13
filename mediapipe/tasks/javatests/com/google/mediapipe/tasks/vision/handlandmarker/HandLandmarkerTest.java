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

package com.google.mediapipe.tasks.vision.handlandmarker;

import static com.google.common.truth.Truth.assertThat;
import static org.junit.Assert.assertThrows;

import android.content.res.AssetManager;
import android.graphics.BitmapFactory;
import android.graphics.RectF;
import androidx.test.core.app.ApplicationProvider;
import androidx.test.ext.junit.runners.AndroidJUnit4;
import com.google.common.truth.Correspondence;
import com.google.mediapipe.framework.MediaPipeException;
import com.google.mediapipe.framework.image.BitmapImageBuilder;
import com.google.mediapipe.framework.image.MPImage;
import com.google.mediapipe.tasks.components.containers.Category;
import com.google.mediapipe.tasks.components.containers.NormalizedLandmark;
import com.google.mediapipe.tasks.components.containers.proto.LandmarksDetectionResultProto.LandmarksDetectionResult;
import com.google.mediapipe.tasks.core.BaseOptions;
import com.google.mediapipe.tasks.vision.core.ImageProcessingOptions;
import com.google.mediapipe.tasks.vision.core.RunningMode;
import com.google.mediapipe.tasks.vision.handlandmarker.HandLandmarker.HandLandmarkerOptions;
import java.io.InputStream;
import java.util.Arrays;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Suite;
import org.junit.runners.Suite.SuiteClasses;

/** Test for {@link HandLandmarker}. */
@RunWith(Suite.class)
@SuiteClasses({HandLandmarkerTest.General.class, HandLandmarkerTest.RunningModeTest.class})
public class HandLandmarkerTest {
  private static final String HAND_LANDMARKER_BUNDLE_ASSET_FILE = "hand_landmarker.task";
  private static final String TWO_HANDS_IMAGE = "right_hands.jpg";
  private static final String THUMB_UP_IMAGE = "thumb_up.jpg";
  private static final String POINTING_UP_ROTATED_IMAGE = "pointing_up_rotated.jpg";
  private static final String NO_HANDS_IMAGE = "cats_and_dogs.jpg";
  private static final String THUMB_UP_LANDMARKS = "thumb_up_landmarks.pb";
  private static final String POINTING_UP_ROTATED_LANDMARKS = "pointing_up_rotated_landmarks.pb";
  private static final String TAG = "Hand Landmarker Test";
  private static final float LANDMARKS_ERROR_TOLERANCE = 0.03f;
  private static final int IMAGE_WIDTH = 382;
  private static final int IMAGE_HEIGHT = 406;

  @RunWith(AndroidJUnit4.class)
  public static final class General extends HandLandmarkerTest {

    @Test
    public void detect_successWithValidModels() throws Exception {
      HandLandmarkerOptions options =
          HandLandmarkerOptions.builder()
              .setBaseOptions(
                  BaseOptions.builder()
                      .setModelAssetPath(HAND_LANDMARKER_BUNDLE_ASSET_FILE)
                      .build())
              .build();
      HandLandmarker handLandmarker =
          HandLandmarker.createFromOptions(ApplicationProvider.getApplicationContext(), options);
      HandLandmarkerResult actualResult =
          handLandmarker.detect(getImageFromAsset(THUMB_UP_IMAGE));
      HandLandmarkerResult expectedResult =
          getExpectedHandLandmarkerResult(THUMB_UP_LANDMARKS);
      assertActualResultApproximatelyEqualsToExpectedResult(actualResult, expectedResult);
    }

    @Test
    public void detect_successWithEmptyResult() throws Exception {
      HandLandmarkerOptions options =
          HandLandmarkerOptions.builder()
              .setBaseOptions(
                  BaseOptions.builder()
                      .setModelAssetPath(HAND_LANDMARKER_BUNDLE_ASSET_FILE)
                      .build())
              .build();
      HandLandmarker handLandmarker =
          HandLandmarker.createFromOptions(ApplicationProvider.getApplicationContext(), options);
      HandLandmarkerResult actualResult =
          handLandmarker.detect(getImageFromAsset(NO_HANDS_IMAGE));
      assertThat(actualResult.landmarks()).isEmpty();
      assertThat(actualResult.worldLandmarks()).isEmpty();
      assertThat(actualResult.handedness()).isEmpty();
    }

    @Test
    public void detect_successWithNumHands() throws Exception {
      HandLandmarkerOptions options =
          HandLandmarkerOptions.builder()
              .setBaseOptions(
                  BaseOptions.builder()
                      .setModelAssetPath(HAND_LANDMARKER_BUNDLE_ASSET_FILE)
                      .build())
              .setNumHands(2)
              .build();
      HandLandmarker handLandmarker =
          HandLandmarker.createFromOptions(ApplicationProvider.getApplicationContext(), options);
      HandLandmarkerResult actualResult =
          handLandmarker.detect(getImageFromAsset(TWO_HANDS_IMAGE));
      assertThat(actualResult.handedness()).hasSize(2);
    }

    @Test
    public void recognize_successWithRotation() throws Exception {
      HandLandmarkerOptions options =
          HandLandmarkerOptions.builder()
              .setBaseOptions(
                  BaseOptions.builder()
                      .setModelAssetPath(HAND_LANDMARKER_BUNDLE_ASSET_FILE)
                      .build())
              .setNumHands(1)
              .build();
      HandLandmarker handLandmarker =
          HandLandmarker.createFromOptions(ApplicationProvider.getApplicationContext(), options);
      ImageProcessingOptions imageProcessingOptions =
          ImageProcessingOptions.builder().setRotationDegrees(-90).build();
      HandLandmarkerResult actualResult =
          handLandmarker.detect(
              getImageFromAsset(POINTING_UP_ROTATED_IMAGE), imageProcessingOptions);
      HandLandmarkerResult expectedResult =
          getExpectedHandLandmarkerResult(POINTING_UP_ROTATED_LANDMARKS);
      assertActualResultApproximatelyEqualsToExpectedResult(actualResult, expectedResult);
    }

    @Test
    public void recognize_failsWithRegionOfInterest() throws Exception {
      HandLandmarkerOptions options =
          HandLandmarkerOptions.builder()
              .setBaseOptions(
                  BaseOptions.builder()
                      .setModelAssetPath(HAND_LANDMARKER_BUNDLE_ASSET_FILE)
                      .build())
              .setNumHands(1)
              .build();
      HandLandmarker handLandmarker =
          HandLandmarker.createFromOptions(ApplicationProvider.getApplicationContext(), options);
      ImageProcessingOptions imageProcessingOptions =
          ImageProcessingOptions.builder().setRegionOfInterest(new RectF(0, 0, 1, 1)).build();
      IllegalArgumentException exception =
          assertThrows(
              IllegalArgumentException.class,
              () ->
                  handLandmarker.detect(getImageFromAsset(THUMB_UP_IMAGE), imageProcessingOptions));
      assertThat(exception)
          .hasMessageThat()
          .contains("HandLandmarker doesn't support region-of-interest");
    }
  }

  @RunWith(AndroidJUnit4.class)
  public static final class RunningModeTest extends HandLandmarkerTest {
    @Test
    public void create_failsWithIllegalResultListenerInNonLiveStreamMode() throws Exception {
      for (RunningMode mode : new RunningMode[] {RunningMode.IMAGE, RunningMode.VIDEO}) {
        IllegalArgumentException exception =
            assertThrows(
                IllegalArgumentException.class,
                () ->
                    HandLandmarkerOptions.builder()
                        .setBaseOptions(
                            BaseOptions.builder()
                                .setModelAssetPath(HAND_LANDMARKER_BUNDLE_ASSET_FILE)
                                .build())
                        .setRunningMode(mode)
                        .setResultListener((HandLandmarkerResults, inputImage) -> {})
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
                HandLandmarkerOptions.builder()
                    .setBaseOptions(
                        BaseOptions.builder()
                            .setModelAssetPath(HAND_LANDMARKER_BUNDLE_ASSET_FILE)
                            .build())
                    .setRunningMode(RunningMode.LIVE_STREAM)
                    .build());
    assertThat(exception)
        .hasMessageThat()
        .contains("a user-defined result listener must be provided");
  }

  @Test
  public void recognize_failsWithCallingWrongApiInImageMode() throws Exception {
    HandLandmarkerOptions options =
        HandLandmarkerOptions.builder()
            .setBaseOptions(
                BaseOptions.builder().setModelAssetPath(HAND_LANDMARKER_BUNDLE_ASSET_FILE).build())
            .setRunningMode(RunningMode.IMAGE)
            .build();

    HandLandmarker handLandmarker =
        HandLandmarker.createFromOptions(ApplicationProvider.getApplicationContext(), options);
    MediaPipeException exception =
        assertThrows(
            MediaPipeException.class,
            () ->
                handLandmarker.detectForVideo(
                    getImageFromAsset(THUMB_UP_IMAGE), /*timestampsMs=*/ 0));
    assertThat(exception).hasMessageThat().contains("not initialized with the video mode");
    exception =
        assertThrows(
            MediaPipeException.class,
            () ->
                handLandmarker.detectAsync(getImageFromAsset(THUMB_UP_IMAGE), /*timestampsMs=*/ 0));
    assertThat(exception).hasMessageThat().contains("not initialized with the live stream mode");
  }

  @Test
  public void recognize_failsWithCallingWrongApiInVideoMode() throws Exception {
    HandLandmarkerOptions options =
        HandLandmarkerOptions.builder()
            .setBaseOptions(
                BaseOptions.builder().setModelAssetPath(HAND_LANDMARKER_BUNDLE_ASSET_FILE).build())
            .setRunningMode(RunningMode.VIDEO)
            .build();

    HandLandmarker handLandmarker =
        HandLandmarker.createFromOptions(ApplicationProvider.getApplicationContext(), options);
    MediaPipeException exception =
        assertThrows(
            MediaPipeException.class,
            () -> handLandmarker.detect(getImageFromAsset(THUMB_UP_IMAGE)));
    assertThat(exception).hasMessageThat().contains("not initialized with the image mode");
    exception =
        assertThrows(
            MediaPipeException.class,
            () ->
                handLandmarker.detectAsync(getImageFromAsset(THUMB_UP_IMAGE), /*timestampsMs=*/ 0));
    assertThat(exception).hasMessageThat().contains("not initialized with the live stream mode");
  }

  @Test
  public void recognize_failsWithCallingWrongApiInLiveSteamMode() throws Exception {
    HandLandmarkerOptions options =
        HandLandmarkerOptions.builder()
            .setBaseOptions(
                BaseOptions.builder().setModelAssetPath(HAND_LANDMARKER_BUNDLE_ASSET_FILE).build())
            .setRunningMode(RunningMode.LIVE_STREAM)
            .setResultListener((HandLandmarkerResults, inputImage) -> {})
            .build();

    HandLandmarker handLandmarker =
        HandLandmarker.createFromOptions(ApplicationProvider.getApplicationContext(), options);
    MediaPipeException exception =
        assertThrows(
            MediaPipeException.class,
            () -> handLandmarker.detect(getImageFromAsset(THUMB_UP_IMAGE)));
    assertThat(exception).hasMessageThat().contains("not initialized with the image mode");
    exception =
        assertThrows(
            MediaPipeException.class,
            () ->
                handLandmarker.detectForVideo(
                    getImageFromAsset(THUMB_UP_IMAGE), /*timestampsMs=*/ 0));
    assertThat(exception).hasMessageThat().contains("not initialized with the video mode");
  }

  @Test
  public void recognize_successWithImageMode() throws Exception {
    HandLandmarkerOptions options =
        HandLandmarkerOptions.builder()
            .setBaseOptions(
                BaseOptions.builder().setModelAssetPath(HAND_LANDMARKER_BUNDLE_ASSET_FILE).build())
            .setRunningMode(RunningMode.IMAGE)
            .build();

    HandLandmarker handLandmarker =
        HandLandmarker.createFromOptions(ApplicationProvider.getApplicationContext(), options);
    HandLandmarkerResult actualResult =
        handLandmarker.detect(getImageFromAsset(THUMB_UP_IMAGE));
    HandLandmarkerResult expectedResult =
        getExpectedHandLandmarkerResult(THUMB_UP_LANDMARKS);
    assertActualResultApproximatelyEqualsToExpectedResult(actualResult, expectedResult);
  }

  @Test
  public void recognize_successWithVideoMode() throws Exception {
    HandLandmarkerOptions options =
        HandLandmarkerOptions.builder()
            .setBaseOptions(
                BaseOptions.builder().setModelAssetPath(HAND_LANDMARKER_BUNDLE_ASSET_FILE).build())
            .setRunningMode(RunningMode.VIDEO)
            .build();
    HandLandmarker handLandmarker =
        HandLandmarker.createFromOptions(ApplicationProvider.getApplicationContext(), options);
    HandLandmarkerResult expectedResult =
        getExpectedHandLandmarkerResult(THUMB_UP_LANDMARKS);
    for (int i = 0; i < 3; i++) {
      HandLandmarkerResult actualResult =
          handLandmarker.detectForVideo(getImageFromAsset(THUMB_UP_IMAGE), /*timestampsMs=*/ i);
      assertActualResultApproximatelyEqualsToExpectedResult(actualResult, expectedResult);
    }
  }

  @Test
  public void recognize_failsWithOutOfOrderInputTimestamps() throws Exception {
    MPImage image = getImageFromAsset(THUMB_UP_IMAGE);
    HandLandmarkerResult expectedResult =
        getExpectedHandLandmarkerResult(THUMB_UP_LANDMARKS);
    HandLandmarkerOptions options =
        HandLandmarkerOptions.builder()
            .setBaseOptions(
                BaseOptions.builder().setModelAssetPath(HAND_LANDMARKER_BUNDLE_ASSET_FILE).build())
            .setRunningMode(RunningMode.LIVE_STREAM)
            .setResultListener(
                (actualResult, inputImage) -> {
                  assertActualResultApproximatelyEqualsToExpectedResult(
                      actualResult, expectedResult);
                  assertImageSizeIsExpected(inputImage);
                })
            .build();
    try (HandLandmarker handLandmarker =
        HandLandmarker.createFromOptions(ApplicationProvider.getApplicationContext(), options)) {
      handLandmarker.detectAsync(image, /*timestampsMs=*/ 1);
      MediaPipeException exception =
          assertThrows(
              MediaPipeException.class,
              () -> handLandmarker.detectAsync(image, /*timestampsMs=*/ 0));
      assertThat(exception)
          .hasMessageThat()
          .contains("having a smaller timestamp than the processed timestamp");
    }
  }

  @Test
  public void recognize_successWithLiveSteamMode() throws Exception {
    MPImage image = getImageFromAsset(THUMB_UP_IMAGE);
    HandLandmarkerResult expectedResult =
        getExpectedHandLandmarkerResult(THUMB_UP_LANDMARKS);
    HandLandmarkerOptions options =
        HandLandmarkerOptions.builder()
            .setBaseOptions(
                BaseOptions.builder().setModelAssetPath(HAND_LANDMARKER_BUNDLE_ASSET_FILE).build())
            .setRunningMode(RunningMode.LIVE_STREAM)
            .setResultListener(
                (actualResult, inputImage) -> {
                  assertActualResultApproximatelyEqualsToExpectedResult(
                      actualResult, expectedResult);
                  assertImageSizeIsExpected(inputImage);
                })
            .build();
    try (HandLandmarker handLandmarker =
        HandLandmarker.createFromOptions(ApplicationProvider.getApplicationContext(), options)) {
      for (int i = 0; i < 3; i++) {
        handLandmarker.detectAsync(image, /*timestampsMs=*/ i);
      }
    }
  }

  private static MPImage getImageFromAsset(String filePath) throws Exception {
    AssetManager assetManager = ApplicationProvider.getApplicationContext().getAssets();
    InputStream istr = assetManager.open(filePath);
    return new BitmapImageBuilder(BitmapFactory.decodeStream(istr)).build();
  }

  private static HandLandmarkerResult getExpectedHandLandmarkerResult(
      String filePath) throws Exception {
    AssetManager assetManager = ApplicationProvider.getApplicationContext().getAssets();
    InputStream istr = assetManager.open(filePath);
    LandmarksDetectionResult landmarksDetectionResultProto =
        LandmarksDetectionResult.parser().parseFrom(istr);
    return HandLandmarkerResult.create(
        Arrays.asList(landmarksDetectionResultProto.getLandmarks()),
        Arrays.asList(landmarksDetectionResultProto.getWorldLandmarks()),
        Arrays.asList(landmarksDetectionResultProto.getClassifications()),
        /*timestampMs=*/ 0);
  }

  private static void assertActualResultApproximatelyEqualsToExpectedResult(
      HandLandmarkerResult actualResult, HandLandmarkerResult expectedResult) {
    // Expects to have the same number of hands detected.
    assertThat(actualResult.landmarks()).hasSize(expectedResult.landmarks().size());
    assertThat(actualResult.worldLandmarks()).hasSize(expectedResult.worldLandmarks().size());
    assertThat(actualResult.handedness()).hasSize(expectedResult.handedness().size());

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
  }

  private static void assertImageSizeIsExpected(MPImage inputImage) {
    assertThat(inputImage).isNotNull();
    assertThat(inputImage.getWidth()).isEqualTo(IMAGE_WIDTH);
    assertThat(inputImage.getHeight()).isEqualTo(IMAGE_HEIGHT);
  }
}
