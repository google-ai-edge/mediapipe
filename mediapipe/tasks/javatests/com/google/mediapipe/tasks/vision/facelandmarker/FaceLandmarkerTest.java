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

package com.google.mediapipe.tasks.vision.facelandmarker;

import static com.google.common.truth.Truth.assertThat;
import static org.junit.Assert.assertThrows;

import android.content.res.AssetManager;
import android.graphics.BitmapFactory;
import android.graphics.RectF;
import androidx.test.core.app.ApplicationProvider;
import androidx.test.ext.junit.runners.AndroidJUnit4;
import com.google.common.truth.Correspondence;
import com.google.mediapipe.formats.proto.LandmarkProto.NormalizedLandmarkList;
import com.google.mediapipe.formats.proto.ClassificationProto.ClassificationList;
import com.google.mediapipe.framework.MediaPipeException;
import com.google.mediapipe.framework.image.BitmapImageBuilder;
import com.google.mediapipe.framework.image.MPImage;
import com.google.mediapipe.tasks.components.containers.Category;
import com.google.mediapipe.tasks.components.containers.NormalizedLandmark;
import com.google.mediapipe.tasks.core.BaseOptions;
import com.google.mediapipe.tasks.vision.core.ImageProcessingOptions;
import com.google.mediapipe.tasks.vision.core.RunningMode;
import com.google.mediapipe.tasks.vision.facegeometry.proto.FaceGeometryProto.FaceGeometry;
import com.google.mediapipe.tasks.vision.facelandmarker.FaceLandmarker.FaceLandmarkerOptions;
import com.google.mediapipe.formats.proto.MatrixDataProto.MatrixData;
import java.io.InputStream;
import java.util.Arrays;
import java.util.List;
import java.util.Optional;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Suite;
import org.junit.runners.Suite.SuiteClasses;

/** Test for {@link FaceLandmarker}. */
@RunWith(Suite.class)
@SuiteClasses({FaceLandmarkerTest.General.class, FaceLandmarkerTest.RunningModeTest.class})
public class FaceLandmarkerTest {
  private static final String FACE_LANDMARKER_BUNDLE_ASSET_FILE =
      "face_landmarker_v2_with_blendshapes.task";
  private static final String PORTRAIT_IMAGE = "portrait.jpg";
  private static final String CAT_IMAGE = "cat.jpg";
  private static final String PORTRAIT_FACE_LANDMARKS = "portrait_expected_face_landmarks.pb";
  private static final String PORTRAIT_FACE_BLENDSHAPES = "portrait_expected_blendshapes.pb";
  private static final String PORTRAIT_FACE_GEOMETRY = "portrait_expected_face_geometry.pb";
  private static final String TAG = "Face Landmarker Test";
  private static final float FACE_LANDMARKS_ERROR_TOLERANCE = 0.01f;
  private static final float FACE_BLENDSHAPES_ERROR_TOLERANCE = 0.13f;
  private static final float FACIAL_TRANSFORMATION_MATRIX_ERROR_TOLERANCE = 0.02f;
  private static final int IMAGE_WIDTH = 820;
  private static final int IMAGE_HEIGHT = 1024;

  @RunWith(AndroidJUnit4.class)
  public static final class General extends FaceLandmarkerTest {

    @Test
    public void detect_successWithValidModels() throws Exception {
      FaceLandmarkerOptions options =
          FaceLandmarkerOptions.builder()
              .setBaseOptions(
                  BaseOptions.builder()
                      .setModelAssetPath(FACE_LANDMARKER_BUNDLE_ASSET_FILE)
                      .build())
              .build();
      FaceLandmarker faceLandmarker =
          FaceLandmarker.createFromOptions(ApplicationProvider.getApplicationContext(), options);
      FaceLandmarkerResult actualResult = faceLandmarker.detect(getImageFromAsset(PORTRAIT_IMAGE));
      FaceLandmarkerResult expectedResult =
          getExpectedFaceLandmarkerResult(
              PORTRAIT_FACE_LANDMARKS, Optional.empty(), Optional.empty());
      assertActualResultApproximatelyEqualsToExpectedResult(actualResult, expectedResult);
    }

    @Test
    public void detect_successWithBlendshapes() throws Exception {
      FaceLandmarkerOptions options =
          FaceLandmarkerOptions.builder()
              .setBaseOptions(
                  BaseOptions.builder()
                      .setModelAssetPath(FACE_LANDMARKER_BUNDLE_ASSET_FILE)
                      .build())
              .setOutputFaceBlendshapes(true)
              .build();
      FaceLandmarker faceLandmarker =
          FaceLandmarker.createFromOptions(ApplicationProvider.getApplicationContext(), options);
      FaceLandmarkerResult actualResult = faceLandmarker.detect(getImageFromAsset(PORTRAIT_IMAGE));
      FaceLandmarkerResult expectedResult =
          getExpectedFaceLandmarkerResult(
              PORTRAIT_FACE_LANDMARKS, Optional.of(PORTRAIT_FACE_BLENDSHAPES), Optional.empty());
      assertActualResultApproximatelyEqualsToExpectedResult(actualResult, expectedResult);
    }

    @Test
    public void detect_successWithFacialTransformationMatrix() throws Exception {
      FaceLandmarkerOptions options =
          FaceLandmarkerOptions.builder()
              .setBaseOptions(
                  BaseOptions.builder()
                      .setModelAssetPath(FACE_LANDMARKER_BUNDLE_ASSET_FILE)
                      .build())
              .setOutputFacialTransformationMatrixes(true)
              .build();
      FaceLandmarker faceLandmarker =
          FaceLandmarker.createFromOptions(ApplicationProvider.getApplicationContext(), options);
      FaceLandmarkerResult actualResult = faceLandmarker.detect(getImageFromAsset(PORTRAIT_IMAGE));
      FaceLandmarkerResult expectedResult =
          getExpectedFaceLandmarkerResult(
              PORTRAIT_FACE_LANDMARKS, Optional.empty(), Optional.of(PORTRAIT_FACE_GEOMETRY));
      assertActualResultApproximatelyEqualsToExpectedResult(actualResult, expectedResult);
    }

    @Test
    public void detect_successWithBlendshapesWithFacialTransformationMatrix() throws Exception {
      FaceLandmarkerOptions options =
          FaceLandmarkerOptions.builder()
              .setBaseOptions(
                  BaseOptions.builder()
                      .setModelAssetPath(FACE_LANDMARKER_BUNDLE_ASSET_FILE)
                      .build())
              .setOutputFaceBlendshapes(true)
              .setOutputFacialTransformationMatrixes(true)
              .build();
      FaceLandmarker faceLandmarker =
          FaceLandmarker.createFromOptions(ApplicationProvider.getApplicationContext(), options);
      FaceLandmarkerResult actualResult = faceLandmarker.detect(getImageFromAsset(PORTRAIT_IMAGE));
      FaceLandmarkerResult expectedResult =
          getExpectedFaceLandmarkerResult(
              PORTRAIT_FACE_LANDMARKS,
              Optional.of(PORTRAIT_FACE_BLENDSHAPES),
              Optional.of(PORTRAIT_FACE_GEOMETRY));
      assertActualResultApproximatelyEqualsToExpectedResult(actualResult, expectedResult);
    }

    @Test
    public void detect_successWithEmptyResult() throws Exception {
      FaceLandmarkerOptions options =
          FaceLandmarkerOptions.builder()
              .setBaseOptions(
                  BaseOptions.builder()
                      .setModelAssetPath(FACE_LANDMARKER_BUNDLE_ASSET_FILE)
                      .build())
              .build();
      FaceLandmarker faceLandmarker =
          FaceLandmarker.createFromOptions(ApplicationProvider.getApplicationContext(), options);
      FaceLandmarkerResult actualResult = faceLandmarker.detect(getImageFromAsset(CAT_IMAGE));
      assertThat(actualResult.faceLandmarks()).isEmpty();
    }

    @Test
    public void detect_failsWithRegionOfInterest() throws Exception {
      FaceLandmarkerOptions options =
          FaceLandmarkerOptions.builder()
              .setBaseOptions(
                  BaseOptions.builder()
                      .setModelAssetPath(FACE_LANDMARKER_BUNDLE_ASSET_FILE)
                      .build())
              .setNumFaces(1)
              .build();
      FaceLandmarker faceLandmarker =
          FaceLandmarker.createFromOptions(ApplicationProvider.getApplicationContext(), options);
      ImageProcessingOptions imageProcessingOptions =
          ImageProcessingOptions.builder().setRegionOfInterest(new RectF(0, 0, 1, 1)).build();
      IllegalArgumentException exception =
          assertThrows(
              IllegalArgumentException.class,
              () ->
                  faceLandmarker.detect(getImageFromAsset(PORTRAIT_IMAGE), imageProcessingOptions));
      assertThat(exception)
          .hasMessageThat()
          .contains("FaceLandmarker doesn't support region-of-interest");
    }
  }

  @RunWith(AndroidJUnit4.class)
  public static final class RunningModeTest extends FaceLandmarkerTest {
    @Test
    public void create_failsWithIllegalResultListenerInNonLiveStreamMode() throws Exception {
      for (RunningMode mode : new RunningMode[] {RunningMode.IMAGE, RunningMode.VIDEO}) {
        IllegalArgumentException exception =
            assertThrows(
                IllegalArgumentException.class,
                () ->
                    FaceLandmarkerOptions.builder()
                        .setBaseOptions(
                            BaseOptions.builder()
                                .setModelAssetPath(FACE_LANDMARKER_BUNDLE_ASSET_FILE)
                                .build())
                        .setRunningMode(mode)
                        .setResultListener((FaceLandmarkerResult, inputImage) -> {})
                        .build());
        assertThat(exception)
            .hasMessageThat()
            .contains("a user-defined result listener shouldn't be provided");
      }
    }

    @Test
    public void create_failsWithMissingResultListenerInLiveSteamMode() throws Exception {
      IllegalArgumentException exception =
          assertThrows(
              IllegalArgumentException.class,
              () ->
                  FaceLandmarkerOptions.builder()
                      .setBaseOptions(
                          BaseOptions.builder()
                              .setModelAssetPath(FACE_LANDMARKER_BUNDLE_ASSET_FILE)
                              .build())
                      .setRunningMode(RunningMode.LIVE_STREAM)
                      .build());
      assertThat(exception)
          .hasMessageThat()
          .contains("a user-defined result listener must be provided");
    }

    @Test
    public void detect_failsWithCallingWrongApiInImageMode() throws Exception {
      FaceLandmarkerOptions options =
          FaceLandmarkerOptions.builder()
              .setBaseOptions(
                  BaseOptions.builder()
                      .setModelAssetPath(FACE_LANDMARKER_BUNDLE_ASSET_FILE)
                      .build())
              .setRunningMode(RunningMode.IMAGE)
              .build();

      FaceLandmarker faceLandmarker =
          FaceLandmarker.createFromOptions(ApplicationProvider.getApplicationContext(), options);
      MediaPipeException exception =
          assertThrows(
              MediaPipeException.class,
              () ->
                  faceLandmarker.detectForVideo(
                      getImageFromAsset(PORTRAIT_IMAGE), /* timestampsMs= */ 0));
      assertThat(exception).hasMessageThat().contains("not initialized with the video mode");
      exception =
          assertThrows(
              MediaPipeException.class,
              () ->
                  faceLandmarker.detectAsync(
                      getImageFromAsset(PORTRAIT_IMAGE), /* timestampsMs= */ 0));
      assertThat(exception).hasMessageThat().contains("not initialized with the live stream mode");
    }

    @Test
    public void detect_failsWithCallingWrongApiInVideoMode() throws Exception {
      FaceLandmarkerOptions options =
          FaceLandmarkerOptions.builder()
              .setBaseOptions(
                  BaseOptions.builder()
                      .setModelAssetPath(FACE_LANDMARKER_BUNDLE_ASSET_FILE)
                      .build())
              .setRunningMode(RunningMode.VIDEO)
              .build();

      FaceLandmarker faceLandmarker =
          FaceLandmarker.createFromOptions(ApplicationProvider.getApplicationContext(), options);
      MediaPipeException exception =
          assertThrows(
              MediaPipeException.class,
              () -> faceLandmarker.detect(getImageFromAsset(PORTRAIT_IMAGE)));
      assertThat(exception).hasMessageThat().contains("not initialized with the image mode");
      exception =
          assertThrows(
              MediaPipeException.class,
              () ->
                  faceLandmarker.detectAsync(
                      getImageFromAsset(PORTRAIT_IMAGE), /* timestampsMs= */ 0));
      assertThat(exception).hasMessageThat().contains("not initialized with the live stream mode");
    }

    @Test
    public void detect_failsWithCallingWrongApiInLiveSteamMode() throws Exception {
      FaceLandmarkerOptions options =
          FaceLandmarkerOptions.builder()
              .setBaseOptions(
                  BaseOptions.builder()
                      .setModelAssetPath(FACE_LANDMARKER_BUNDLE_ASSET_FILE)
                      .build())
              .setRunningMode(RunningMode.LIVE_STREAM)
              .setResultListener((FaceLandmarkerResult, inputImage) -> {})
              .build();

      FaceLandmarker faceLandmarker =
          FaceLandmarker.createFromOptions(ApplicationProvider.getApplicationContext(), options);
      MediaPipeException exception =
          assertThrows(
              MediaPipeException.class,
              () -> faceLandmarker.detect(getImageFromAsset(PORTRAIT_IMAGE)));
      assertThat(exception).hasMessageThat().contains("not initialized with the image mode");
      exception =
          assertThrows(
              MediaPipeException.class,
              () ->
                  faceLandmarker.detectForVideo(
                      getImageFromAsset(PORTRAIT_IMAGE), /* timestampsMs= */ 0));
      assertThat(exception).hasMessageThat().contains("not initialized with the video mode");
    }

    @Test
    public void detect_successWithImageMode() throws Exception {
      FaceLandmarkerOptions options =
          FaceLandmarkerOptions.builder()
              .setBaseOptions(
                  BaseOptions.builder()
                      .setModelAssetPath(FACE_LANDMARKER_BUNDLE_ASSET_FILE)
                      .build())
              .setOutputFaceBlendshapes(true)
              .setOutputFacialTransformationMatrixes(true)
              .setRunningMode(RunningMode.IMAGE)
              .build();

      FaceLandmarker faceLandmarker =
          FaceLandmarker.createFromOptions(ApplicationProvider.getApplicationContext(), options);
      FaceLandmarkerResult actualResult = faceLandmarker.detect(getImageFromAsset(PORTRAIT_IMAGE));
      FaceLandmarkerResult expectedResult =
          getExpectedFaceLandmarkerResult(
              PORTRAIT_FACE_LANDMARKS,
              Optional.of(PORTRAIT_FACE_BLENDSHAPES),
              Optional.of(PORTRAIT_FACE_GEOMETRY));
      assertActualResultApproximatelyEqualsToExpectedResult(actualResult, expectedResult);
    }

    @Test
    public void detect_successWithVideoMode() throws Exception {
      FaceLandmarkerOptions options =
          FaceLandmarkerOptions.builder()
              .setBaseOptions(
                  BaseOptions.builder()
                      .setModelAssetPath(FACE_LANDMARKER_BUNDLE_ASSET_FILE)
                      .build())
              .setOutputFaceBlendshapes(true)
              .setRunningMode(RunningMode.VIDEO)
              .build();
      FaceLandmarker faceLandmarker =
          FaceLandmarker.createFromOptions(ApplicationProvider.getApplicationContext(), options);
      FaceLandmarkerResult expectedResult =
          getExpectedFaceLandmarkerResult(
              PORTRAIT_FACE_LANDMARKS, Optional.of(PORTRAIT_FACE_BLENDSHAPES), Optional.empty());
      for (int i = 0; i < 3; i++) {
        FaceLandmarkerResult actualResult =
            faceLandmarker.detectForVideo(getImageFromAsset(PORTRAIT_IMAGE), /* timestampsMs= */ i);
        assertActualResultApproximatelyEqualsToExpectedResult(actualResult, expectedResult);
      }
    }

    @Test
    public void detect_failsWithOutOfOrderInputTimestamps() throws Exception {
      MPImage image = getImageFromAsset(PORTRAIT_IMAGE);
      FaceLandmarkerResult expectedResult =
          getExpectedFaceLandmarkerResult(
              PORTRAIT_FACE_LANDMARKS, Optional.of(PORTRAIT_FACE_BLENDSHAPES), Optional.empty());
      FaceLandmarkerOptions options =
          FaceLandmarkerOptions.builder()
              .setBaseOptions(
                  BaseOptions.builder()
                      .setModelAssetPath(FACE_LANDMARKER_BUNDLE_ASSET_FILE)
                      .build())
              .setOutputFaceBlendshapes(true)
              .setRunningMode(RunningMode.LIVE_STREAM)
              .setResultListener(
                  (actualResult, inputImage) -> {
                    assertActualResultApproximatelyEqualsToExpectedResult(
                        actualResult, expectedResult);
                    assertImageSizeIsExpected(inputImage);
                  })
              .build();
      try (FaceLandmarker faceLandmarker =
          FaceLandmarker.createFromOptions(ApplicationProvider.getApplicationContext(), options)) {
        faceLandmarker.detectAsync(image, /* timestampsMs= */ 1);
        MediaPipeException exception =
            assertThrows(
                MediaPipeException.class,
                () -> faceLandmarker.detectAsync(image, /* timestampsMs= */ 0));
        assertThat(exception)
            .hasMessageThat()
            .contains("having a smaller timestamp than the processed timestamp");
      }
    }

    @Test
    public void detect_successWithLiveSteamMode() throws Exception {
      MPImage image = getImageFromAsset(PORTRAIT_IMAGE);
      FaceLandmarkerResult expectedResult =
          getExpectedFaceLandmarkerResult(
              PORTRAIT_FACE_LANDMARKS, Optional.of(PORTRAIT_FACE_BLENDSHAPES), Optional.empty());
      FaceLandmarkerOptions options =
          FaceLandmarkerOptions.builder()
              .setBaseOptions(
                  BaseOptions.builder()
                      .setModelAssetPath(FACE_LANDMARKER_BUNDLE_ASSET_FILE)
                      .build())
              .setOutputFaceBlendshapes(true)
              .setRunningMode(RunningMode.LIVE_STREAM)
              .setResultListener(
                  (actualResult, inputImage) -> {
                    assertActualResultApproximatelyEqualsToExpectedResult(
                        actualResult, expectedResult);
                    assertImageSizeIsExpected(inputImage);
                  })
              .build();
      try (FaceLandmarker faceLandmarker =
          FaceLandmarker.createFromOptions(ApplicationProvider.getApplicationContext(), options)) {
        for (int i = 0; i < 3; i++) {
          faceLandmarker.detectAsync(image, /* timestampsMs= */ i);
        }
      }
    }
  }

  private static MPImage getImageFromAsset(String filePath) throws Exception {
    AssetManager assetManager = ApplicationProvider.getApplicationContext().getAssets();
    InputStream istr = assetManager.open(filePath);
    return new BitmapImageBuilder(BitmapFactory.decodeStream(istr)).build();
  }

  private static FaceLandmarkerResult getExpectedFaceLandmarkerResult(
      String faceLandmarksFilePath,
      Optional<String> faceBlendshapesFilePath,
      Optional<String> faceGeometryFilePath)
      throws Exception {
    AssetManager assetManager = ApplicationProvider.getApplicationContext().getAssets();

    List<NormalizedLandmarkList> faceLandmarks =
        Arrays.asList(
            NormalizedLandmarkList.parser().parseFrom(assetManager.open(faceLandmarksFilePath)));
    Optional<List<ClassificationList>> faceBlendshapes = Optional.empty();
    if (faceBlendshapesFilePath.isPresent()) {
      faceBlendshapes =
          Optional.of(
              Arrays.asList(
                  ClassificationList.parser()
                      .parseFrom(assetManager.open(faceBlendshapesFilePath.get()))));
    }
    Optional<List<MatrixData>> facialTransformationMatrixes = Optional.empty();
    if (faceGeometryFilePath.isPresent()) {
      FaceGeometry faceGeometry =
          FaceGeometry.parser().parseFrom(assetManager.open(faceGeometryFilePath.get()));
      facialTransformationMatrixes =
          Optional.of(Arrays.asList(faceGeometry.getPoseTransformMatrix()));
    }

    return FaceLandmarkerResult.create(
        faceLandmarks, faceBlendshapes, facialTransformationMatrixes, /* timestampMs= */ 0);
  }

  private static void assertActualResultApproximatelyEqualsToExpectedResult(
      FaceLandmarkerResult actualResult, FaceLandmarkerResult expectedResult) {
    // Expects to have the same number of faces detected.
    assertThat(actualResult.faceLandmarks()).hasSize(expectedResult.faceLandmarks().size());
    assertThat(actualResult.faceBlendshapes().isPresent())
        .isEqualTo(expectedResult.faceBlendshapes().isPresent());
    assertThat(actualResult.facialTransformationMatrixes().isPresent())
        .isEqualTo(expectedResult.facialTransformationMatrixes().isPresent());

    // Actual face landmarks match expected face landmarks.
    assertThat(actualResult.faceLandmarks().get(0))
        .comparingElementsUsing(
            Correspondence.from(
                (Correspondence.BinaryPredicate<NormalizedLandmark, NormalizedLandmark>)
                    (actual, expected) -> {
                      return Correspondence.tolerance(FACE_LANDMARKS_ERROR_TOLERANCE)
                              .compare(actual.x(), expected.x())
                          && Correspondence.tolerance(FACE_LANDMARKS_ERROR_TOLERANCE)
                              .compare(actual.y(), expected.y());
                    },
                "face landmarks approximately equal to"))
        .containsExactlyElementsIn(expectedResult.faceLandmarks().get(0));

    // Actual face blendshapes match expected face blendshapes.
    if (actualResult.faceBlendshapes().isPresent()) {
      assertThat(actualResult.faceBlendshapes().get().get(0))
          .comparingElementsUsing(
              Correspondence.from(
                  (Correspondence.BinaryPredicate<Category, Category>)
                      (actual, expected) -> {
                        return Correspondence.tolerance(FACE_BLENDSHAPES_ERROR_TOLERANCE)
                                .compare(actual.score(), expected.score())
                            && actual.index() == expected.index()
                            && actual.categoryName().equals(expected.categoryName());
                      },
                  "face blendshapes approximately equal to"))
          .containsExactlyElementsIn(expectedResult.faceBlendshapes().get().get(0));
    }

    // Actual transformation matrix match expected transformation matrix;
    if (actualResult.facialTransformationMatrixes().isPresent()) {
      assertThat(actualResult.facialTransformationMatrixes().get().get(0))
          .usingTolerance(FACIAL_TRANSFORMATION_MATRIX_ERROR_TOLERANCE)
          .containsExactly(expectedResult.facialTransformationMatrixes().get().get(0));
    }
  }

  private static void assertImageSizeIsExpected(MPImage inputImage) {
    assertThat(inputImage).isNotNull();
    assertThat(inputImage.getWidth()).isEqualTo(IMAGE_WIDTH);
    assertThat(inputImage.getHeight()).isEqualTo(IMAGE_HEIGHT);
  }
}
