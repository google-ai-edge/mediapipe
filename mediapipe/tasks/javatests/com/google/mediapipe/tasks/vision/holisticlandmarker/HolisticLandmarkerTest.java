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

package com.google.mediapipe.tasks.vision.holisticlandmarker;

import static com.google.common.truth.Truth.assertThat;
import static org.junit.Assert.assertThrows;

import android.content.res.AssetManager;
import android.graphics.BitmapFactory;
import android.graphics.RectF;
import androidx.test.core.app.ApplicationProvider;
import androidx.test.ext.junit.runners.AndroidJUnit4;
import com.google.common.truth.Correspondence;
import com.google.mediapipe.formats.proto.LandmarkProto.LandmarkList;
import com.google.mediapipe.formats.proto.ClassificationProto.ClassificationList;
import com.google.mediapipe.framework.MediaPipeException;
import com.google.mediapipe.framework.image.BitmapImageBuilder;
import com.google.mediapipe.framework.image.ByteBufferImageBuilder;
import com.google.mediapipe.framework.image.MPImage;
import com.google.mediapipe.tasks.components.containers.Category;
import com.google.mediapipe.tasks.components.containers.NormalizedLandmark;
import com.google.mediapipe.tasks.core.BaseOptions;
import com.google.mediapipe.tasks.vision.core.ImageProcessingOptions;
import com.google.mediapipe.tasks.vision.core.RunningMode;
import com.google.mediapipe.tasks.vision.holisticlandmarker.HolisticLandmarker.HolisticLandmarkerOptions;
import com.google.mediapipe.tasks.vision.holisticlandmarker.HolisticResultProto.HolisticResult;
import java.io.InputStream;
import java.nio.ByteBuffer;
import java.util.Optional;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Suite;
import org.junit.runners.Suite.SuiteClasses;

/** Test for {@link HolisticLandmarker}. */
@RunWith(Suite.class)
@SuiteClasses({HolisticLandmarkerTest.General.class, HolisticLandmarkerTest.RunningModeTest.class})
public class HolisticLandmarkerTest {
  private static final String HOLISTIC_LANDMARKER_BUNDLE_ASSET_FILE = "holistic_landmarker.task";
  private static final String POSE_IMAGE = "male_full_height_hands.jpg";
  private static final String CAT_IMAGE = "cat.jpg";
  private static final String FACE_IMAGE = "portrait.jpg";
  private static final String HOLISTIC_RESULT = "male_full_height_hands_result_cpu.pb";
  private static final String TAG = "Holistic Landmarker Test";
  private static final float FACE_LANDMARKS_ERROR_TOLERANCE = 0.03f;
  private static final float FACE_BLENDSHAPES_ERROR_TOLERANCE = 0.13f;
  private static final MPImage PLACEHOLDER_MASK =
      new ByteBufferImageBuilder(
              ByteBuffer.allocate(0), /* widht= */ 0, /* height= */ 0, MPImage.IMAGE_FORMAT_VEC32F1)
          .build();
  private static final int IMAGE_WIDTH = 638;
  private static final int IMAGE_HEIGHT = 1000;

  private static final Correspondence<NormalizedLandmark, NormalizedLandmark> VALIDATE_LANDMARRKS =
      Correspondence.from(
          (Correspondence.BinaryPredicate<NormalizedLandmark, NormalizedLandmark>)
              (actual, expected) -> {
                return Correspondence.tolerance(FACE_LANDMARKS_ERROR_TOLERANCE)
                        .compare(actual.x(), expected.x())
                    && Correspondence.tolerance(FACE_LANDMARKS_ERROR_TOLERANCE)
                        .compare(actual.y(), expected.y());
              },
          "landmarks approximately equal to");

  private static final Correspondence<Category, Category> VALIDATE_BLENDSHAPES =
      Correspondence.from(
          (Correspondence.BinaryPredicate<Category, Category>)
              (actual, expected) ->
                  Correspondence.tolerance(FACE_BLENDSHAPES_ERROR_TOLERANCE)
                          .compare(actual.score(), expected.score())
                      && actual.index() == expected.index()
                      && actual.categoryName().equals(expected.categoryName()),
          "face blendshapes approximately equal to");

  @RunWith(AndroidJUnit4.class)
  public static final class General extends HolisticLandmarkerTest {

    @Test
    public void detect_successWithValidModels() throws Exception {
      HolisticLandmarkerOptions options =
          HolisticLandmarkerOptions.builder()
              .setBaseOptions(
                  BaseOptions.builder()
                      .setModelAssetPath(HOLISTIC_LANDMARKER_BUNDLE_ASSET_FILE)
                      .build())
              .build();
      HolisticLandmarker holisticLandmarker =
          HolisticLandmarker.createFromOptions(
              ApplicationProvider.getApplicationContext(), options);
      HolisticLandmarkerResult actualResult =
          holisticLandmarker.detect(getImageFromAsset(POSE_IMAGE));
      HolisticLandmarkerResult expectedResult =
          getExpectedHolisticLandmarkerResult(
              HOLISTIC_RESULT, /* hasFaceBlendshapes= */ false, /* hasSegmentationMask= */ false);
      assertActualResultApproximatelyEqualsToExpectedResult(actualResult, expectedResult);
    }

    @Test
    public void detect_successWithBlendshapes() throws Exception {
      HolisticLandmarkerOptions options =
          HolisticLandmarkerOptions.builder()
              .setBaseOptions(
                  BaseOptions.builder()
                      .setModelAssetPath(HOLISTIC_LANDMARKER_BUNDLE_ASSET_FILE)
                      .build())
              .setOutputFaceBlendshapes(true)
              .build();
      HolisticLandmarker holisticLandmarker =
          HolisticLandmarker.createFromOptions(
              ApplicationProvider.getApplicationContext(), options);
      HolisticLandmarkerResult actualResult =
          holisticLandmarker.detect(getImageFromAsset(POSE_IMAGE));
      HolisticLandmarkerResult expectedResult =
          getExpectedHolisticLandmarkerResult(
              HOLISTIC_RESULT, /* hasFaceBlendshapes= */ true, /* hasSegmentationMask= */ false);
      assertActualResultApproximatelyEqualsToExpectedResult(actualResult, expectedResult);
    }

    @Test
    public void detect_successWithSegmentationMasks() throws Exception {
      HolisticLandmarkerOptions options =
          HolisticLandmarkerOptions.builder()
              .setBaseOptions(
                  BaseOptions.builder()
                      .setModelAssetPath(HOLISTIC_LANDMARKER_BUNDLE_ASSET_FILE)
                      .build())
              .setOutputPoseSegmentationMasks(true)
              .build();
      HolisticLandmarker holisticLandmarker =
          HolisticLandmarker.createFromOptions(
              ApplicationProvider.getApplicationContext(), options);
      HolisticLandmarkerResult actualResult =
          holisticLandmarker.detect(getImageFromAsset(POSE_IMAGE));
      HolisticLandmarkerResult expectedResult =
          getExpectedHolisticLandmarkerResult(
              HOLISTIC_RESULT, /* hasFaceBlendshapes= */ false, /* hasSegmentationMask= */ true);
      assertActualResultApproximatelyEqualsToExpectedResult(actualResult, expectedResult);
    }

    @Test
    public void detect_successWithEmptyResult() throws Exception {
      HolisticLandmarkerOptions options =
          HolisticLandmarkerOptions.builder()
              .setBaseOptions(
                  BaseOptions.builder()
                      .setModelAssetPath(HOLISTIC_LANDMARKER_BUNDLE_ASSET_FILE)
                      .build())
              .build();
      HolisticLandmarker holisticLandmarker =
          HolisticLandmarker.createFromOptions(
              ApplicationProvider.getApplicationContext(), options);
      HolisticLandmarkerResult actualResult =
          holisticLandmarker.detect(getImageFromAsset(CAT_IMAGE));
      assertThat(actualResult.faceLandmarks()).isEmpty();
    }

    @Test
    public void detect_successWithFace() throws Exception {
      HolisticLandmarkerOptions options =
          HolisticLandmarkerOptions.builder()
              .setBaseOptions(
                  BaseOptions.builder()
                      .setModelAssetPath(HOLISTIC_LANDMARKER_BUNDLE_ASSET_FILE)
                      .build())
              .build();
      HolisticLandmarker holisticLandmarker =
          HolisticLandmarker.createFromOptions(
              ApplicationProvider.getApplicationContext(), options);
      HolisticLandmarkerResult actualResult =
          holisticLandmarker.detect(getImageFromAsset(FACE_IMAGE));
      assertThat(actualResult.faceLandmarks()).isNotEmpty();
    }

    @Test
    public void detect_failsWithRegionOfInterest() throws Exception {
      HolisticLandmarkerOptions options =
          HolisticLandmarkerOptions.builder()
              .setBaseOptions(
                  BaseOptions.builder()
                      .setModelAssetPath(HOLISTIC_LANDMARKER_BUNDLE_ASSET_FILE)
                      .build())
              .build();
      HolisticLandmarker holisticLandmarker =
          HolisticLandmarker.createFromOptions(
              ApplicationProvider.getApplicationContext(), options);
      ImageProcessingOptions imageProcessingOptions =
          ImageProcessingOptions.builder().setRegionOfInterest(new RectF(0, 0, 1, 1)).build();
      IllegalArgumentException exception =
          assertThrows(
              IllegalArgumentException.class,
              () ->
                  holisticLandmarker.detect(getImageFromAsset(POSE_IMAGE), imageProcessingOptions));
      assertThat(exception)
          .hasMessageThat()
          .contains("HolisticLandmarker doesn't support region-of-interest");
    }
  }

  @RunWith(AndroidJUnit4.class)
  public static final class RunningModeTest extends HolisticLandmarkerTest {
    private void assertCreationFailsWithResultListenerInNonLiveStreamMode(RunningMode runningMode)
        throws Exception {
      IllegalArgumentException exception =
          assertThrows(
              IllegalArgumentException.class,
              () ->
                  HolisticLandmarkerOptions.builder()
                      .setBaseOptions(
                          BaseOptions.builder()
                              .setModelAssetPath(HOLISTIC_LANDMARKER_BUNDLE_ASSET_FILE)
                              .build())
                      .setRunningMode(runningMode)
                      .setResultListener((HolisticLandmarkerResult, inputImage) -> {})
                      .build());
      assertThat(exception)
          .hasMessageThat()
          .contains("a user-defined result listener shouldn't be provided");
    }

    @Test
    public void create_failsWithIllegalResultListenerInVideoMode() throws Exception {
      assertCreationFailsWithResultListenerInNonLiveStreamMode(RunningMode.VIDEO);
    }

    @Test
    public void create_failsWithIllegalResultListenerInImageMode() throws Exception {
      assertCreationFailsWithResultListenerInNonLiveStreamMode(RunningMode.IMAGE);
    }

    @Test
    public void create_failsWithMissingResultListenerInLiveSteamMode() throws Exception {
      IllegalArgumentException exception =
          assertThrows(
              IllegalArgumentException.class,
              () ->
                  HolisticLandmarkerOptions.builder()
                      .setBaseOptions(
                          BaseOptions.builder()
                              .setModelAssetPath(HOLISTIC_LANDMARKER_BUNDLE_ASSET_FILE)
                              .build())
                      .setRunningMode(RunningMode.LIVE_STREAM)
                      .build());
      assertThat(exception)
          .hasMessageThat()
          .contains("a user-defined result listener must be provided");
    }

    @Test
    public void detect_failsWithCallingWrongApiInImageMode() throws Exception {
      HolisticLandmarkerOptions options =
          HolisticLandmarkerOptions.builder()
              .setBaseOptions(
                  BaseOptions.builder()
                      .setModelAssetPath(HOLISTIC_LANDMARKER_BUNDLE_ASSET_FILE)
                      .build())
              .setRunningMode(RunningMode.IMAGE)
              .build();

      HolisticLandmarker holisticLandmarker =
          HolisticLandmarker.createFromOptions(
              ApplicationProvider.getApplicationContext(), options);
      MediaPipeException exception =
          assertThrows(
              MediaPipeException.class,
              () ->
                  holisticLandmarker.detectForVideo(
                      getImageFromAsset(POSE_IMAGE), /* timestampsMs= */ 0));
      assertThat(exception).hasMessageThat().contains("not initialized with the video mode");
      exception =
          assertThrows(
              MediaPipeException.class,
              () ->
                  holisticLandmarker.detectAsync(
                      getImageFromAsset(POSE_IMAGE), /* timestampsMs= */ 0));
      assertThat(exception).hasMessageThat().contains("not initialized with the live stream mode");
    }

    @Test
    public void detect_failsWithCallingWrongApiInVideoMode() throws Exception {
      HolisticLandmarkerOptions options =
          HolisticLandmarkerOptions.builder()
              .setBaseOptions(
                  BaseOptions.builder()
                      .setModelAssetPath(HOLISTIC_LANDMARKER_BUNDLE_ASSET_FILE)
                      .build())
              .setRunningMode(RunningMode.VIDEO)
              .build();

      HolisticLandmarker holisticLandmarker =
          HolisticLandmarker.createFromOptions(
              ApplicationProvider.getApplicationContext(), options);
      MediaPipeException exception =
          assertThrows(
              MediaPipeException.class,
              () -> holisticLandmarker.detect(getImageFromAsset(POSE_IMAGE)));
      assertThat(exception).hasMessageThat().contains("not initialized with the image mode");
      exception =
          assertThrows(
              MediaPipeException.class,
              () ->
                  holisticLandmarker.detectAsync(
                      getImageFromAsset(POSE_IMAGE), /* timestampsMs= */ 0));
      assertThat(exception).hasMessageThat().contains("not initialized with the live stream mode");
    }

    @Test
    public void detect_failsWithCallingWrongApiInLiveSteamMode() throws Exception {
      HolisticLandmarkerOptions options =
          HolisticLandmarkerOptions.builder()
              .setBaseOptions(
                  BaseOptions.builder()
                      .setModelAssetPath(HOLISTIC_LANDMARKER_BUNDLE_ASSET_FILE)
                      .build())
              .setRunningMode(RunningMode.LIVE_STREAM)
              .setResultListener((HolisticLandmarkerResult, inputImage) -> {})
              .build();

      HolisticLandmarker holisticLandmarker =
          HolisticLandmarker.createFromOptions(
              ApplicationProvider.getApplicationContext(), options);
      MediaPipeException exception =
          assertThrows(
              MediaPipeException.class,
              () -> holisticLandmarker.detect(getImageFromAsset(POSE_IMAGE)));
      assertThat(exception).hasMessageThat().contains("not initialized with the image mode");
      exception =
          assertThrows(
              MediaPipeException.class,
              () ->
                  holisticLandmarker.detectForVideo(
                      getImageFromAsset(POSE_IMAGE), /* timestampsMs= */ 0));
      assertThat(exception).hasMessageThat().contains("not initialized with the video mode");
    }

    @Test
    public void detect_successWithImageMode() throws Exception {
      HolisticLandmarkerOptions options =
          HolisticLandmarkerOptions.builder()
              .setBaseOptions(
                  BaseOptions.builder()
                      .setModelAssetPath(HOLISTIC_LANDMARKER_BUNDLE_ASSET_FILE)
                      .build())
              .setRunningMode(RunningMode.IMAGE)
              .build();

      HolisticLandmarker holisticLandmarker =
          HolisticLandmarker.createFromOptions(
              ApplicationProvider.getApplicationContext(), options);
      HolisticLandmarkerResult actualResult =
          holisticLandmarker.detect(getImageFromAsset(POSE_IMAGE));
      HolisticLandmarkerResult expectedResult =
          getExpectedHolisticLandmarkerResult(
              HOLISTIC_RESULT, /* hasFaceBlendshapes= */ false, /* hasSegmentationMask= */ false);
      assertActualResultApproximatelyEqualsToExpectedResult(actualResult, expectedResult);
    }

    @Test
    public void detect_successWithVideoMode() throws Exception {
      HolisticLandmarkerOptions options =
          HolisticLandmarkerOptions.builder()
              .setBaseOptions(
                  BaseOptions.builder()
                      .setModelAssetPath(HOLISTIC_LANDMARKER_BUNDLE_ASSET_FILE)
                      .build())
              .setRunningMode(RunningMode.VIDEO)
              .build();
      HolisticLandmarker holisticLandmarker =
          HolisticLandmarker.createFromOptions(
              ApplicationProvider.getApplicationContext(), options);
      HolisticLandmarkerResult expectedResult =
          getExpectedHolisticLandmarkerResult(
              HOLISTIC_RESULT, /* hasFaceBlendshapes= */ false, /* hasSegmentationMask= */ false);
      for (int i = 0; i < 3; i++) {
        HolisticLandmarkerResult actualResult =
            holisticLandmarker.detectForVideo(getImageFromAsset(POSE_IMAGE), /* timestampsMs= */ i);
        assertActualResultApproximatelyEqualsToExpectedResult(actualResult, expectedResult);
      }
    }

    @Test
    public void detect_failsWithOutOfOrderInputTimestamps() throws Exception {
      MPImage image = getImageFromAsset(POSE_IMAGE);
      HolisticLandmarkerOptions options =
          HolisticLandmarkerOptions.builder()
              .setBaseOptions(
                  BaseOptions.builder()
                      .setModelAssetPath(HOLISTIC_LANDMARKER_BUNDLE_ASSET_FILE)
                      .build())
              .setRunningMode(RunningMode.LIVE_STREAM)
              .setResultListener((actualResult, inputImage) -> {})
              .build();
      try (HolisticLandmarker holisticLandmarker =
          HolisticLandmarker.createFromOptions(
              ApplicationProvider.getApplicationContext(), options)) {
        holisticLandmarker.detectAsync(image, /* timestampsMs= */ 1);
        MediaPipeException exception =
            assertThrows(
                MediaPipeException.class,
                () -> holisticLandmarker.detectAsync(image, /* timestampsMs= */ 0));
        assertThat(exception)
            .hasMessageThat()
            .contains("having a smaller timestamp than the processed timestamp");
      }
    }

    @Test
    public void detect_successWithLiveSteamMode() throws Exception {
      MPImage image = getImageFromAsset(POSE_IMAGE);
      HolisticLandmarkerResult expectedResult =
          getExpectedHolisticLandmarkerResult(
              HOLISTIC_RESULT, /* hasFaceBlendshapes= */ false, /* hasSegmentationMask= */ false);
      HolisticLandmarkerOptions options =
          HolisticLandmarkerOptions.builder()
              .setBaseOptions(
                  BaseOptions.builder()
                      .setModelAssetPath(HOLISTIC_LANDMARKER_BUNDLE_ASSET_FILE)
                      .build())
              .setRunningMode(RunningMode.LIVE_STREAM)
              .setResultListener(
                  (actualResult, inputImage) -> {
                    assertActualResultApproximatelyEqualsToExpectedResult(
                        actualResult, expectedResult);
                    assertImageSizeIsExpected(inputImage);
                  })
              .build();
      try (HolisticLandmarker holisticLandmarker =
          HolisticLandmarker.createFromOptions(
              ApplicationProvider.getApplicationContext(), options)) {
        for (int i = 0; i < 3; i++) {
          holisticLandmarker.detectAsync(image, /* timestampsMs= */ i);
        }
      }
    }
  }

  private static MPImage getImageFromAsset(String filePath) throws Exception {
    AssetManager assetManager = ApplicationProvider.getApplicationContext().getAssets();
    InputStream istr = assetManager.open(filePath);
    return new BitmapImageBuilder(BitmapFactory.decodeStream(istr)).build();
  }

  private static HolisticLandmarkerResult getExpectedHolisticLandmarkerResult(
      String resultPath, boolean hasFaceBlendshapes, boolean hasSegmentationMask) throws Exception {
    AssetManager assetManager = ApplicationProvider.getApplicationContext().getAssets();

    HolisticResult holisticResult = HolisticResult.parseFrom(assetManager.open(resultPath));

    Optional<ClassificationList> blendshapes =
        hasFaceBlendshapes
            ? Optional.of(holisticResult.getFaceBlendshapes())
            : Optional.<ClassificationList>empty();
    Optional<MPImage> segmentationMask =
        hasSegmentationMask ? Optional.of(PLACEHOLDER_MASK) : Optional.<MPImage>empty();

    return HolisticLandmarkerResult.create(
        holisticResult.getFaceLandmarks(),
        blendshapes,
        holisticResult.getPoseLandmarks(),
        LandmarkList.getDefaultInstance(),
        segmentationMask,
        holisticResult.getLeftHandLandmarks(),
        LandmarkList.getDefaultInstance(),
        holisticResult.getRightHandLandmarks(),
        LandmarkList.getDefaultInstance(),
        /* timestampMs= */ 0);
  }

  private static void assertActualResultApproximatelyEqualsToExpectedResult(
      HolisticLandmarkerResult actualResult, HolisticLandmarkerResult expectedResult) {
    // Expects to have the same number of holistics detected.
    assertThat(actualResult.faceLandmarks()).hasSize(expectedResult.faceLandmarks().size());
    assertThat(actualResult.faceBlendshapes().isPresent())
        .isEqualTo(expectedResult.faceBlendshapes().isPresent());
    assertThat(actualResult.poseLandmarks()).hasSize(expectedResult.poseLandmarks().size());
    assertThat(actualResult.segmentationMask().isPresent())
        .isEqualTo(expectedResult.segmentationMask().isPresent());
    assertThat(actualResult.leftHandLandmarks()).hasSize(expectedResult.leftHandLandmarks().size());
    assertThat(actualResult.rightHandLandmarks())
        .hasSize(expectedResult.rightHandLandmarks().size());

    // Actual face landmarks match expected face landmarks.
    assertThat(actualResult.faceLandmarks())
        .comparingElementsUsing(VALIDATE_LANDMARRKS)
        .containsExactlyElementsIn(expectedResult.faceLandmarks());

    // Actual face blendshapes match expected face blendshapes.
    if (actualResult.faceBlendshapes().isPresent()) {
      assertThat(actualResult.faceBlendshapes().get())
          .comparingElementsUsing(VALIDATE_BLENDSHAPES)
          .containsExactlyElementsIn(expectedResult.faceBlendshapes().get());
    }

    // Actual pose landmarks match expected pose landmarks.
    assertThat(actualResult.poseLandmarks())
        .comparingElementsUsing(VALIDATE_LANDMARRKS)
        .containsExactlyElementsIn(expectedResult.poseLandmarks());

    if (actualResult.segmentationMask().isPresent()) {
      assertImageSizeIsExpected(actualResult.segmentationMask().get());
    }

    // Actual left hand landmarks match expected left hand landmarks.
    assertThat(actualResult.leftHandLandmarks())
        .comparingElementsUsing(VALIDATE_LANDMARRKS)
        .containsExactlyElementsIn(expectedResult.leftHandLandmarks());

    // Actual right hand landmarks match expected right hand landmarks.
    assertThat(actualResult.rightHandLandmarks())
        .comparingElementsUsing(VALIDATE_LANDMARRKS)
        .containsExactlyElementsIn(expectedResult.rightHandLandmarks());
  }

  private static void assertImageSizeIsExpected(MPImage inputImage) {
    assertThat(inputImage).isNotNull();
    assertThat(inputImage.getWidth()).isEqualTo(IMAGE_WIDTH);
    assertThat(inputImage.getHeight()).isEqualTo(IMAGE_HEIGHT);
  }
}
