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

package com.google.mediapipe.tasks.vision.poselandmarker;

import static com.google.common.truth.Truth.assertThat;
import static com.google.common.truth.Truth.assertWithMessage;
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
import com.google.mediapipe.tasks.components.containers.Landmark;
import com.google.mediapipe.tasks.components.containers.NormalizedLandmark;
import com.google.mediapipe.tasks.components.containers.proto.LandmarksDetectionResultProto.LandmarksDetectionResult;
import com.google.mediapipe.tasks.core.BaseOptions;
import com.google.mediapipe.tasks.vision.core.ImageProcessingOptions;
import com.google.mediapipe.tasks.vision.core.RunningMode;
import com.google.mediapipe.tasks.vision.poselandmarker.PoseLandmarker.PoseLandmarkerOptions;
import java.io.InputStream;
import java.util.Arrays;
import java.util.List;
import java.util.Optional;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Suite;
import org.junit.runners.Suite.SuiteClasses;

/** Test for {@link PoseLandmarker}. */
@RunWith(Suite.class)
@SuiteClasses({PoseLandmarkerTest.General.class, PoseLandmarkerTest.RunningModeTest.class})
public class PoseLandmarkerTest {
  private static final String POSE_LANDMARKER_BUNDLE_ASSET_FILE = "pose_landmarker.task";
  private static final String POSE_IMAGE = "pose.jpg";
  private static final String POSE_LANDMARKS = "pose_landmarks.pb";
  private static final String NO_POSES_IMAGE = "burger.jpg";
  private static final String TAG = "Pose Landmarker Test";
  private static final float LANDMARKS_ERROR_TOLERANCE = 0.03f;
  private static final float VISIBILITY_TOLERANCE = 0.9f;
  private static final float PRESENCE_TOLERANCE = 0.9f;
  private static final int IMAGE_WIDTH = 1000;
  private static final int IMAGE_HEIGHT = 667;

  @RunWith(AndroidJUnit4.class)
  public static final class General extends PoseLandmarkerTest {

    @Test
    public void detect_successWithValidModels() throws Exception {
      PoseLandmarkerOptions options =
          PoseLandmarkerOptions.builder()
              .setBaseOptions(
                  BaseOptions.builder()
                      .setModelAssetPath(POSE_LANDMARKER_BUNDLE_ASSET_FILE)
                      .build())
              .build();
      PoseLandmarker poseLandmarker =
          PoseLandmarker.createFromOptions(ApplicationProvider.getApplicationContext(), options);
      PoseLandmarkerResult actualResult = poseLandmarker.detect(getImageFromAsset(POSE_IMAGE));
      PoseLandmarkerResult expectedResult = getExpectedPoseLandmarkerResult(POSE_LANDMARKS);
      assertActualResultApproximatelyEqualsToExpectedResult(actualResult, expectedResult);
      assertAllLandmarksAreVisibleAndPresent(
          actualResult, VISIBILITY_TOLERANCE, PRESENCE_TOLERANCE);
    }

    @Test
    public void detect_successWithEmptyResult() throws Exception {
      PoseLandmarkerOptions options =
          PoseLandmarkerOptions.builder()
              .setBaseOptions(
                  BaseOptions.builder()
                      .setModelAssetPath(POSE_LANDMARKER_BUNDLE_ASSET_FILE)
                      .build())
              .build();
      PoseLandmarker poseLandmarker =
          PoseLandmarker.createFromOptions(ApplicationProvider.getApplicationContext(), options);
      PoseLandmarkerResult actualResult = poseLandmarker.detect(getImageFromAsset(NO_POSES_IMAGE));
      assertThat(actualResult.landmarks()).isEmpty();
      assertThat(actualResult.worldLandmarks()).isEmpty();
      // TODO: Add additional tests for MP Tasks Pose Graphs
      // Add tests for segmentation masks.
    }

    @Test
    public void recognize_failsWithRegionOfInterest() throws Exception {
      PoseLandmarkerOptions options =
          PoseLandmarkerOptions.builder()
              .setBaseOptions(
                  BaseOptions.builder()
                      .setModelAssetPath(POSE_LANDMARKER_BUNDLE_ASSET_FILE)
                      .build())
              .setNumPoses(1)
              .build();
      PoseLandmarker poseLandmarker =
          PoseLandmarker.createFromOptions(ApplicationProvider.getApplicationContext(), options);
      ImageProcessingOptions imageProcessingOptions =
          ImageProcessingOptions.builder().setRegionOfInterest(new RectF(0, 0, 1, 1)).build();
      IllegalArgumentException exception =
          assertThrows(
              IllegalArgumentException.class,
              () -> poseLandmarker.detect(getImageFromAsset(POSE_IMAGE), imageProcessingOptions));
      assertThat(exception)
          .hasMessageThat()
          .contains("PoseLandmarker doesn't support region-of-interest");
    }
  }

  @RunWith(AndroidJUnit4.class)
  public static final class RunningModeTest extends PoseLandmarkerTest {
    @Test
    public void create_failsWithIllegalResultListenerInNonLiveStreamMode() throws Exception {
      for (RunningMode mode : new RunningMode[] {RunningMode.IMAGE, RunningMode.VIDEO}) {
        IllegalArgumentException exception =
            assertThrows(
                IllegalArgumentException.class,
                () ->
                    PoseLandmarkerOptions.builder()
                        .setBaseOptions(
                            BaseOptions.builder()
                                .setModelAssetPath(POSE_LANDMARKER_BUNDLE_ASSET_FILE)
                                .build())
                        .setRunningMode(mode)
                        .setResultListener((PoseLandmarkerResults, inputImage) -> {})
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
                PoseLandmarkerOptions.builder()
                    .setBaseOptions(
                        BaseOptions.builder()
                            .setModelAssetPath(POSE_LANDMARKER_BUNDLE_ASSET_FILE)
                            .build())
                    .setRunningMode(RunningMode.LIVE_STREAM)
                    .build());
    assertThat(exception)
        .hasMessageThat()
        .contains("a user-defined result listener must be provided");
  }

  @Test
  public void recognize_failsWithCallingWrongApiInImageMode() throws Exception {
    PoseLandmarkerOptions options =
        PoseLandmarkerOptions.builder()
            .setBaseOptions(
                BaseOptions.builder().setModelAssetPath(POSE_LANDMARKER_BUNDLE_ASSET_FILE).build())
            .setRunningMode(RunningMode.IMAGE)
            .build();

    PoseLandmarker poseLandmarker =
        PoseLandmarker.createFromOptions(ApplicationProvider.getApplicationContext(), options);
    MediaPipeException exception =
        assertThrows(
            MediaPipeException.class,
            () ->
                poseLandmarker.detectForVideo(
                    getImageFromAsset(POSE_IMAGE), /* timestampsMs= */ 0));
    assertThat(exception).hasMessageThat().contains("not initialized with the video mode");
    exception =
        assertThrows(
            MediaPipeException.class,
            () -> poseLandmarker.detectAsync(getImageFromAsset(POSE_IMAGE), /* timestampsMs= */ 0));
    assertThat(exception).hasMessageThat().contains("not initialized with the live stream mode");
  }

  @Test
  public void recognize_failsWithCallingWrongApiInVideoMode() throws Exception {
    PoseLandmarkerOptions options =
        PoseLandmarkerOptions.builder()
            .setBaseOptions(
                BaseOptions.builder().setModelAssetPath(POSE_LANDMARKER_BUNDLE_ASSET_FILE).build())
            .setRunningMode(RunningMode.VIDEO)
            .build();

    PoseLandmarker poseLandmarker =
        PoseLandmarker.createFromOptions(ApplicationProvider.getApplicationContext(), options);
    MediaPipeException exception =
        assertThrows(
            MediaPipeException.class, () -> poseLandmarker.detect(getImageFromAsset(POSE_IMAGE)));
    assertThat(exception).hasMessageThat().contains("not initialized with the image mode");
    exception =
        assertThrows(
            MediaPipeException.class,
            () -> poseLandmarker.detectAsync(getImageFromAsset(POSE_IMAGE), /* timestampsMs= */ 0));
    assertThat(exception).hasMessageThat().contains("not initialized with the live stream mode");
  }

  @Test
  public void recognize_failsWithCallingWrongApiInLiveSteamMode() throws Exception {
    PoseLandmarkerOptions options =
        PoseLandmarkerOptions.builder()
            .setBaseOptions(
                BaseOptions.builder().setModelAssetPath(POSE_LANDMARKER_BUNDLE_ASSET_FILE).build())
            .setRunningMode(RunningMode.LIVE_STREAM)
            .setResultListener((PoseLandmarkerResults, inputImage) -> {})
            .build();

    PoseLandmarker poseLandmarker =
        PoseLandmarker.createFromOptions(ApplicationProvider.getApplicationContext(), options);
    MediaPipeException exception =
        assertThrows(
            MediaPipeException.class, () -> poseLandmarker.detect(getImageFromAsset(POSE_IMAGE)));
    assertThat(exception).hasMessageThat().contains("not initialized with the image mode");
    exception =
        assertThrows(
            MediaPipeException.class,
            () ->
                poseLandmarker.detectForVideo(
                    getImageFromAsset(POSE_IMAGE), /* timestampsMs= */ 0));
    assertThat(exception).hasMessageThat().contains("not initialized with the video mode");
  }

  @Test
  public void recognize_successWithImageMode() throws Exception {
    PoseLandmarkerOptions options =
        PoseLandmarkerOptions.builder()
            .setBaseOptions(
                BaseOptions.builder().setModelAssetPath(POSE_LANDMARKER_BUNDLE_ASSET_FILE).build())
            .setRunningMode(RunningMode.IMAGE)
            .build();

    PoseLandmarker poseLandmarker =
        PoseLandmarker.createFromOptions(ApplicationProvider.getApplicationContext(), options);
    PoseLandmarkerResult actualResult = poseLandmarker.detect(getImageFromAsset(POSE_IMAGE));
    PoseLandmarkerResult expectedResult = getExpectedPoseLandmarkerResult(POSE_LANDMARKS);
    assertActualResultApproximatelyEqualsToExpectedResult(actualResult, expectedResult);
  }

  @Test
  public void recognize_successWithVideoMode() throws Exception {
    PoseLandmarkerOptions options =
        PoseLandmarkerOptions.builder()
            .setBaseOptions(
                BaseOptions.builder().setModelAssetPath(POSE_LANDMARKER_BUNDLE_ASSET_FILE).build())
            .setRunningMode(RunningMode.VIDEO)
            .build();
    PoseLandmarker poseLandmarker =
        PoseLandmarker.createFromOptions(ApplicationProvider.getApplicationContext(), options);
    PoseLandmarkerResult expectedResult = getExpectedPoseLandmarkerResult(POSE_LANDMARKS);
    for (int i = 0; i < 3; i++) {
      PoseLandmarkerResult actualResult =
          poseLandmarker.detectForVideo(getImageFromAsset(POSE_IMAGE), /* timestampsMs= */ i);
      assertActualResultApproximatelyEqualsToExpectedResult(actualResult, expectedResult);
    }
  }

  @Test
  public void recognize_failsWithOutOfOrderInputTimestamps() throws Exception {
    MPImage image = getImageFromAsset(POSE_IMAGE);
    PoseLandmarkerResult expectedResult = getExpectedPoseLandmarkerResult(POSE_LANDMARKS);
    PoseLandmarkerOptions options =
        PoseLandmarkerOptions.builder()
            .setBaseOptions(
                BaseOptions.builder().setModelAssetPath(POSE_LANDMARKER_BUNDLE_ASSET_FILE).build())
            .setRunningMode(RunningMode.LIVE_STREAM)
            .setResultListener(
                (actualResult, inputImage) -> {
                  assertActualResultApproximatelyEqualsToExpectedResult(
                      actualResult, expectedResult);
                  assertImageSizeIsExpected(inputImage);
                })
            .build();
    try (PoseLandmarker poseLandmarker =
        PoseLandmarker.createFromOptions(ApplicationProvider.getApplicationContext(), options)) {
      poseLandmarker.detectAsync(image, /* timestampsMs= */ 1);
      MediaPipeException exception =
          assertThrows(
              MediaPipeException.class,
              () -> poseLandmarker.detectAsync(image, /* timestampsMs= */ 0));
      assertThat(exception)
          .hasMessageThat()
          .contains("having a smaller timestamp than the processed timestamp");
    }
  }

  @Test
  public void recognize_successWithLiveSteamMode() throws Exception {
    MPImage image = getImageFromAsset(POSE_IMAGE);
    PoseLandmarkerResult expectedResult = getExpectedPoseLandmarkerResult(POSE_LANDMARKS);
    PoseLandmarkerOptions options =
        PoseLandmarkerOptions.builder()
            .setBaseOptions(
                BaseOptions.builder().setModelAssetPath(POSE_LANDMARKER_BUNDLE_ASSET_FILE).build())
            .setRunningMode(RunningMode.LIVE_STREAM)
            .setResultListener(
                (actualResult, inputImage) -> {
                  assertActualResultApproximatelyEqualsToExpectedResult(
                      actualResult, expectedResult);
                  assertImageSizeIsExpected(inputImage);
                })
            .build();
    try (PoseLandmarker poseLandmarker =
        PoseLandmarker.createFromOptions(ApplicationProvider.getApplicationContext(), options)) {
      for (int i = 0; i < 3; i++) {
        poseLandmarker.detectAsync(image, /* timestampsMs= */ i);
      }
    }
  }

  private static MPImage getImageFromAsset(String filePath) throws Exception {
    AssetManager assetManager = ApplicationProvider.getApplicationContext().getAssets();
    InputStream istr = assetManager.open(filePath);
    return new BitmapImageBuilder(BitmapFactory.decodeStream(istr)).build();
  }

  private static PoseLandmarkerResult getExpectedPoseLandmarkerResult(String filePath)
      throws Exception {
    AssetManager assetManager = ApplicationProvider.getApplicationContext().getAssets();
    InputStream istr = assetManager.open(filePath);
    LandmarksDetectionResult landmarksDetectionResultProto =
        LandmarksDetectionResult.parser().parseFrom(istr);
    return PoseLandmarkerResult.create(
        Arrays.asList(landmarksDetectionResultProto.getLandmarks()),
        Arrays.asList(landmarksDetectionResultProto.getWorldLandmarks()),
        Optional.empty(),
        /* timestampMs= */ 0);
  }

  private static void assertActualResultApproximatelyEqualsToExpectedResult(
      PoseLandmarkerResult actualResult, PoseLandmarkerResult expectedResult) {
    // TODO: Add additional tests for MP Tasks Pose Graphs
    // Add additional tests for auxiliary, world landmarks and segmentation masks.
    // Expects to have the same number of poses detected.
    assertThat(actualResult.landmarks()).hasSize(expectedResult.landmarks().size());

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
  }

  private static void assertImageSizeIsExpected(MPImage inputImage) {
    assertThat(inputImage).isNotNull();
    assertThat(inputImage.getWidth()).isEqualTo(IMAGE_WIDTH);
    assertThat(inputImage.getHeight()).isEqualTo(IMAGE_HEIGHT);
  }

  private static void assertAllLandmarksAreVisibleAndPresent(
      PoseLandmarkerResult result, float visbilityThreshold, float presenceThreshold) {
    for (int i = 0; i < result.landmarks().size(); i++) {
      List<NormalizedLandmark> landmarks = result.landmarks().get(i);
      for (int j = 0; j < landmarks.size(); j++) {
        NormalizedLandmark landmark = landmarks.get(j);
        String landmarkMessage = "Landmark List " + i + " landmark " + j + ": " + landmark;
        landmark
            .visibility()
            .ifPresent(
                val ->
                    assertWithMessage(landmarkMessage).that(val).isAtLeast((visbilityThreshold)));
        landmark
            .presence()
            .ifPresent(
                val -> assertWithMessage(landmarkMessage).that(val).isAtLeast((presenceThreshold)));
      }
    }
    for (int i = 0; i < result.worldLandmarks().size(); i++) {
      List<Landmark> landmarks = result.worldLandmarks().get(i);
      for (int j = 0; j < landmarks.size(); j++) {
        Landmark landmark = landmarks.get(j);
        String landmarkMessage = "World Landmark List " + i + " landmark " + j + ": " + landmark;
        landmark
            .visibility()
            .ifPresent(
                val ->
                    assertWithMessage(landmarkMessage).that(val).isAtLeast((visbilityThreshold)));
        landmark
            .presence()
            .ifPresent(
                val -> assertWithMessage(landmarkMessage).that(val).isAtLeast((presenceThreshold)));
      }
    }
  }
}
