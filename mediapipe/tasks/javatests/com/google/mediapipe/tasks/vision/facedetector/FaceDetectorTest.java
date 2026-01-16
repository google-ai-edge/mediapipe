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

package com.google.mediapipe.tasks.vision.facedetector;

import static com.google.common.truth.Truth.assertThat;
import static org.junit.Assert.assertThrows;

import android.content.res.AssetManager;
import android.graphics.BitmapFactory;
import android.graphics.RectF;
import androidx.test.core.app.ApplicationProvider;
import androidx.test.ext.junit.runners.AndroidJUnit4;
import com.google.mediapipe.framework.MediaPipeException;
import com.google.mediapipe.framework.image.BitmapImageBuilder;
import com.google.mediapipe.framework.image.MPImage;
import com.google.mediapipe.tasks.components.containers.NormalizedKeypoint;
import com.google.mediapipe.tasks.core.BaseOptions;
import com.google.mediapipe.tasks.core.TestUtils;
import com.google.mediapipe.tasks.vision.core.ImageProcessingOptions;
import com.google.mediapipe.tasks.vision.core.RunningMode;
import com.google.mediapipe.tasks.vision.facedetector.FaceDetector.FaceDetectorOptions;
import java.io.InputStream;
import java.nio.ByteBuffer;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Suite;
import org.junit.runners.Suite.SuiteClasses;

/** Test for {@link FaceDetector}. */
@RunWith(Suite.class)
@SuiteClasses({FaceDetectorTest.General.class, FaceDetectorTest.RunningModeTest.class})
public class FaceDetectorTest {
  private static final String MODEL_FILE = "face_detection_short_range.tflite";
  private static final String CAT_IMAGE = "cat.jpg";
  private static final String PORTRAIT_IMAGE = "portrait.jpg";
  private static final String PORTRAIT_ROTATED_IMAGE = "portrait_rotated.jpg";
  private static final float KEYPOINTS_DIFF_TOLERANCE = 0.01f;
  private static final float PIXEL_DIFF_TOLERANCE = 5.0f;
  private static final RectF PORTRAIT_FACE_BOUNDING_BOX = new RectF(283, 115, 514, 349);
  private static final List<NormalizedKeypoint> PORTRAIT_FACE_KEYPOINTS =
      Collections.unmodifiableList(
          Arrays.asList(
              NormalizedKeypoint.create(0.44416f, 0.17643f),
              NormalizedKeypoint.create(0.55514f, 0.17731f),
              NormalizedKeypoint.create(0.50467f, 0.22657f),
              NormalizedKeypoint.create(0.50227f, 0.27199f),
              NormalizedKeypoint.create(0.36063f, 0.20143f),
              NormalizedKeypoint.create(0.60841f, 0.20409f)));
  private static final RectF PORTRAIT_ROTATED_FACE_BOUNDING_BOX = new RectF(674, 283, 910, 519);
  private static final List<NormalizedKeypoint> PORTRAIT_ROTATED_FACE_KEYPOINTS =
      Collections.unmodifiableList(
          Arrays.asList(
              NormalizedKeypoint.create(0.82075f, 0.44679f),
              NormalizedKeypoint.create(0.81965f, 0.56261f),
              NormalizedKeypoint.create(0.76194f, 0.51719f),
              NormalizedKeypoint.create(0.71993f, 0.51360f),
              NormalizedKeypoint.create(0.80700f, 0.36298f),
              NormalizedKeypoint.create(0.80882f, 0.61204f)));

  @RunWith(AndroidJUnit4.class)
  public static final class General extends FaceDetectorTest {

    @Test
    public void detect_successWithValidModels() throws Exception {
      FaceDetectorOptions options =
          FaceDetectorOptions.builder()
              .setBaseOptions(BaseOptions.builder().setModelAssetPath(MODEL_FILE).build())
              .build();
      FaceDetector faceDetector =
          FaceDetector.createFromOptions(ApplicationProvider.getApplicationContext(), options);
      FaceDetectorResult results = faceDetector.detect(getImageFromAsset(PORTRAIT_IMAGE));
      assertContainsSinglePortraitFace(
          results, PORTRAIT_FACE_BOUNDING_BOX, PORTRAIT_FACE_KEYPOINTS);
    }

    @Test
    public void detect_succeedsWithMinDetectionConfidence() throws Exception {
      FaceDetectorOptions options =
          FaceDetectorOptions.builder()
              .setBaseOptions(BaseOptions.builder().setModelAssetPath(MODEL_FILE).build())
              .setMinDetectionConfidence(1.0f)
              .build();
      FaceDetector faceDetector =
          FaceDetector.createFromOptions(ApplicationProvider.getApplicationContext(), options);
      FaceDetectorResult results = faceDetector.detect(getImageFromAsset(PORTRAIT_IMAGE));
      // Set minDetectionConfidence to 1.0, so the detected face should be all filtered out.
      assertThat(results.detections().isEmpty()).isTrue();
    }

    @Test
    public void detect_succeedsWithEmptyFace() throws Exception {
      FaceDetectorOptions options =
          FaceDetectorOptions.builder()
              .setBaseOptions(BaseOptions.builder().setModelAssetPath(MODEL_FILE).build())
              .setMinDetectionConfidence(1.0f)
              .build();
      FaceDetector faceDetector =
          FaceDetector.createFromOptions(ApplicationProvider.getApplicationContext(), options);
      FaceDetectorResult results = faceDetector.detect(getImageFromAsset(CAT_IMAGE));
      assertThat(results.detections().isEmpty()).isTrue();
    }

    @Test
    public void detect_succeedsWithModelFileObject() throws Exception {
      FaceDetector faceDetector =
          FaceDetector.createFromFile(
              ApplicationProvider.getApplicationContext(),
              TestUtils.loadFile(ApplicationProvider.getApplicationContext(), MODEL_FILE));
      FaceDetectorResult results = faceDetector.detect(getImageFromAsset(PORTRAIT_IMAGE));
      assertContainsSinglePortraitFace(
          results, PORTRAIT_FACE_BOUNDING_BOX, PORTRAIT_FACE_KEYPOINTS);
    }

    @Test
    public void detect_succeedsWithModelBuffer() throws Exception {
      FaceDetector faceDetector =
          FaceDetector.createFromBuffer(
              ApplicationProvider.getApplicationContext(),
              TestUtils.loadToDirectByteBuffer(
                  ApplicationProvider.getApplicationContext(), MODEL_FILE));
      FaceDetectorResult results = faceDetector.detect(getImageFromAsset(PORTRAIT_IMAGE));
      assertContainsSinglePortraitFace(
          results, PORTRAIT_FACE_BOUNDING_BOX, PORTRAIT_FACE_KEYPOINTS);
    }

    @Test
    public void detect_succeedsWithModelBufferAndOptions() throws Exception {
      FaceDetectorOptions options =
          FaceDetectorOptions.builder()
              .setBaseOptions(
                  BaseOptions.builder()
                      .setModelAssetBuffer(
                          TestUtils.loadToDirectByteBuffer(
                              ApplicationProvider.getApplicationContext(), MODEL_FILE))
                      .build())
              .build();
      FaceDetector faceDetector =
          FaceDetector.createFromOptions(ApplicationProvider.getApplicationContext(), options);
      FaceDetectorResult results = faceDetector.detect(getImageFromAsset(PORTRAIT_IMAGE));
      assertContainsSinglePortraitFace(
          results, PORTRAIT_FACE_BOUNDING_BOX, PORTRAIT_FACE_KEYPOINTS);
    }

    @Test
    public void create_failsWithMissingModel() throws Exception {
      String nonexistentFile = "/path/to/non/existent/file";
      MediaPipeException exception =
          assertThrows(
              MediaPipeException.class,
              () ->
                  FaceDetector.createFromFile(
                      ApplicationProvider.getApplicationContext(), nonexistentFile));
      assertThat(exception).hasMessageThat().contains(nonexistentFile);
    }

    @Test
    public void create_failsWithInvalidModelBuffer() throws Exception {
      // Create a non-direct model ByteBuffer.
      ByteBuffer modelBuffer =
          TestUtils.loadToNonDirectByteBuffer(
              ApplicationProvider.getApplicationContext(), MODEL_FILE);

      IllegalArgumentException exception =
          assertThrows(
              IllegalArgumentException.class,
              () ->
                  FaceDetector.createFromBuffer(
                      ApplicationProvider.getApplicationContext(), modelBuffer));

      assertThat(exception)
          .hasMessageThat()
          .contains("The model buffer should be either a direct ByteBuffer or a MappedByteBuffer.");
    }

    @Test
    public void detect_succeedsWithRotation() throws Exception {
      FaceDetectorOptions options =
          FaceDetectorOptions.builder()
              .setBaseOptions(BaseOptions.builder().setModelAssetPath(MODEL_FILE).build())
              .build();
      FaceDetector faceDetector =
          FaceDetector.createFromOptions(ApplicationProvider.getApplicationContext(), options);
      ImageProcessingOptions imageProcessingOptions =
          ImageProcessingOptions.builder().setRotationDegrees(-90).build();
      FaceDetectorResult results =
          faceDetector.detect(getImageFromAsset(PORTRAIT_ROTATED_IMAGE), imageProcessingOptions);
      assertContainsSinglePortraitFace(
          results, PORTRAIT_ROTATED_FACE_BOUNDING_BOX, PORTRAIT_ROTATED_FACE_KEYPOINTS);
    }

    @Test
    public void detect_failsWithRegionOfInterest() throws Exception {
      FaceDetectorOptions options =
          FaceDetectorOptions.builder()
              .setBaseOptions(BaseOptions.builder().setModelAssetPath(MODEL_FILE).build())
              .build();
      FaceDetector faceDetector =
          FaceDetector.createFromOptions(ApplicationProvider.getApplicationContext(), options);
      ImageProcessingOptions imageProcessingOptions =
          ImageProcessingOptions.builder().setRegionOfInterest(new RectF(0, 0, 1, 1)).build();
      IllegalArgumentException exception =
          assertThrows(
              IllegalArgumentException.class,
              () -> faceDetector.detect(getImageFromAsset(PORTRAIT_IMAGE), imageProcessingOptions));
      assertThat(exception)
          .hasMessageThat()
          .contains("FaceDetector doesn't support region-of-interest");
    }
  }

  @RunWith(AndroidJUnit4.class)
  public static final class RunningModeTest extends FaceDetectorTest {

    @Test
    public void create_failsWithIllegalResultListenerInNonLiveStreamMode() throws Exception {
      for (RunningMode mode : new RunningMode[] {RunningMode.IMAGE, RunningMode.VIDEO}) {
        IllegalArgumentException exception =
            assertThrows(
                IllegalArgumentException.class,
                () ->
                    FaceDetectorOptions.builder()
                        .setBaseOptions(BaseOptions.builder().setModelAssetPath(MODEL_FILE).build())
                        .setRunningMode(mode)
                        .setResultListener((faceDetectorResult, inputImage) -> {})
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
                  FaceDetectorOptions.builder()
                      .setBaseOptions(BaseOptions.builder().setModelAssetPath(MODEL_FILE).build())
                      .setRunningMode(RunningMode.LIVE_STREAM)
                      .build());
      assertThat(exception)
          .hasMessageThat()
          .contains("a user-defined result listener must be provided");
    }

    @Test
    public void detect_failsWithCallingWrongApiInImageMode() throws Exception {
      FaceDetectorOptions options =
          FaceDetectorOptions.builder()
              .setBaseOptions(BaseOptions.builder().setModelAssetPath(MODEL_FILE).build())
              .setRunningMode(RunningMode.IMAGE)
              .build();

      FaceDetector faceDetector =
          FaceDetector.createFromOptions(ApplicationProvider.getApplicationContext(), options);
      MediaPipeException exception =
          assertThrows(
              MediaPipeException.class,
              () ->
                  faceDetector.detectForVideo(
                      getImageFromAsset(PORTRAIT_IMAGE), /* timestampsMs= */ 0));
      assertThat(exception).hasMessageThat().contains("not initialized with the video mode");
      exception =
          assertThrows(
              MediaPipeException.class,
              () ->
                  faceDetector.detectAsync(
                      getImageFromAsset(PORTRAIT_IMAGE), /* timestampsMs= */ 0));
      assertThat(exception).hasMessageThat().contains("not initialized with the live stream mode");
    }

    @Test
    public void detect_failsWithCallingWrongApiInVideoMode() throws Exception {
      FaceDetectorOptions options =
          FaceDetectorOptions.builder()
              .setBaseOptions(BaseOptions.builder().setModelAssetPath(MODEL_FILE).build())
              .setRunningMode(RunningMode.VIDEO)
              .build();

      FaceDetector faceDetector =
          FaceDetector.createFromOptions(ApplicationProvider.getApplicationContext(), options);
      MediaPipeException exception =
          assertThrows(
              MediaPipeException.class,
              () -> faceDetector.detect(getImageFromAsset(PORTRAIT_IMAGE)));
      assertThat(exception).hasMessageThat().contains("not initialized with the image mode");
      exception =
          assertThrows(
              MediaPipeException.class,
              () ->
                  faceDetector.detectAsync(
                      getImageFromAsset(PORTRAIT_IMAGE), /* timestampsMs= */ 0));
      assertThat(exception).hasMessageThat().contains("not initialized with the live stream mode");
    }

    @Test
    public void detect_failsWithCallingWrongApiInLiveSteamMode() throws Exception {
      FaceDetectorOptions options =
          FaceDetectorOptions.builder()
              .setBaseOptions(BaseOptions.builder().setModelAssetPath(MODEL_FILE).build())
              .setRunningMode(RunningMode.LIVE_STREAM)
              .setResultListener((faceDetectorResult, inputImage) -> {})
              .build();

      FaceDetector faceDetector =
          FaceDetector.createFromOptions(ApplicationProvider.getApplicationContext(), options);

      MediaPipeException exception =
          assertThrows(
              MediaPipeException.class,
              () -> faceDetector.detect(getImageFromAsset(PORTRAIT_IMAGE)));
      assertThat(exception).hasMessageThat().contains("not initialized with the image mode");
      exception =
          assertThrows(
              MediaPipeException.class,
              () ->
                  faceDetector.detectForVideo(
                      getImageFromAsset(PORTRAIT_IMAGE), /* timestampsMs= */ 0));
      assertThat(exception).hasMessageThat().contains("not initialized with the video mode");
    }

    @Test
    public void detect_successWithImageMode() throws Exception {
      FaceDetectorOptions options =
          FaceDetectorOptions.builder()
              .setBaseOptions(BaseOptions.builder().setModelAssetPath(MODEL_FILE).build())
              .setRunningMode(RunningMode.IMAGE)
              .build();
      FaceDetector faceDetector =
          FaceDetector.createFromOptions(ApplicationProvider.getApplicationContext(), options);
      FaceDetectorResult results = faceDetector.detect(getImageFromAsset(PORTRAIT_IMAGE));
      assertContainsSinglePortraitFace(
          results, PORTRAIT_FACE_BOUNDING_BOX, PORTRAIT_FACE_KEYPOINTS);
    }

    @Test
    public void detect_successWithVideoMode() throws Exception {
      FaceDetectorOptions options =
          FaceDetectorOptions.builder()
              .setBaseOptions(BaseOptions.builder().setModelAssetPath(MODEL_FILE).build())
              .setRunningMode(RunningMode.VIDEO)
              .build();
      FaceDetector faceDetector =
          FaceDetector.createFromOptions(ApplicationProvider.getApplicationContext(), options);
      for (int i = 0; i < 3; i++) {
        FaceDetectorResult results =
            faceDetector.detectForVideo(getImageFromAsset(PORTRAIT_IMAGE), /* timestampsMs= */ i);
        assertContainsSinglePortraitFace(
            results, PORTRAIT_FACE_BOUNDING_BOX, PORTRAIT_FACE_KEYPOINTS);
      }
    }

    @Test
    public void detect_failsWithOutOfOrderInputTimestamps() throws Exception {
      MPImage image = getImageFromAsset(PORTRAIT_IMAGE);
      FaceDetectorOptions options =
          FaceDetectorOptions.builder()
              .setBaseOptions(BaseOptions.builder().setModelAssetPath(MODEL_FILE).build())
              .setRunningMode(RunningMode.LIVE_STREAM)
              .setResultListener(
                  (faceDetectorResult, inputImage) -> {
                    assertContainsSinglePortraitFace(
                        faceDetectorResult, PORTRAIT_FACE_BOUNDING_BOX, PORTRAIT_FACE_KEYPOINTS);
                  })
              .build();
      try (FaceDetector faceDetector =
          FaceDetector.createFromOptions(ApplicationProvider.getApplicationContext(), options)) {
        faceDetector.detectAsync(image, /* timestampsMs= */ 1);
        MediaPipeException exception =
            assertThrows(
                MediaPipeException.class,
                () -> faceDetector.detectAsync(image, /* timestampsMs= */ 0));
        assertThat(exception)
            .hasMessageThat()
            .contains("having a smaller timestamp than the processed timestamp");
      }
    }

    @Test
    public void detect_successWithLiveSteamMode() throws Exception {
      MPImage image = getImageFromAsset(PORTRAIT_IMAGE);
      FaceDetectorOptions options =
          FaceDetectorOptions.builder()
              .setBaseOptions(BaseOptions.builder().setModelAssetPath(MODEL_FILE).build())
              .setRunningMode(RunningMode.LIVE_STREAM)
              .setResultListener(
                  (faceDetectorResult, inputImage) -> {
                    assertContainsSinglePortraitFace(
                        faceDetectorResult, PORTRAIT_FACE_BOUNDING_BOX, PORTRAIT_FACE_KEYPOINTS);
                  })
              .build();
      try (FaceDetector faceDetector =
          FaceDetector.createFromOptions(ApplicationProvider.getApplicationContext(), options)) {
        for (int i = 0; i < 3; i++) {
          faceDetector.detectAsync(image, /* timestampsMs= */ i);
        }
      }
    }
  }

  private static MPImage getImageFromAsset(String filePath) throws Exception {
    AssetManager assetManager = ApplicationProvider.getApplicationContext().getAssets();
    InputStream istr = assetManager.open(filePath);
    return new BitmapImageBuilder(BitmapFactory.decodeStream(istr)).build();
  }

  private static void assertContainsSinglePortraitFace(
      FaceDetectorResult results,
      RectF expectedboundingBox,
      List<NormalizedKeypoint> expectedKeypoints) {
    assertThat(results.detections()).hasSize(1);
    assertApproximatelyEqualBoundingBoxes(
        results.detections().get(0).boundingBox(), expectedboundingBox);
    assertThat(results.detections().get(0).keypoints().isPresent()).isTrue();
    assertApproximatelyEqualKeypoints(
        results.detections().get(0).keypoints().get(), expectedKeypoints);
  }

  private static void assertApproximatelyEqualBoundingBoxes(
      RectF boundingBox1, RectF boundingBox2) {
    assertThat(boundingBox1.left).isWithin(PIXEL_DIFF_TOLERANCE).of(boundingBox2.left);
    assertThat(boundingBox1.top).isWithin(PIXEL_DIFF_TOLERANCE).of(boundingBox2.top);
    assertThat(boundingBox1.right).isWithin(PIXEL_DIFF_TOLERANCE).of(boundingBox2.right);
    assertThat(boundingBox1.bottom).isWithin(PIXEL_DIFF_TOLERANCE).of(boundingBox2.bottom);
  }

  private static void assertApproximatelyEqualKeypoints(
      List<NormalizedKeypoint> keypoints1, List<NormalizedKeypoint> keypoints2) {
    assertThat(keypoints1.size()).isEqualTo(keypoints2.size());
    for (int i = 0; i < keypoints1.size(); i++) {
      assertThat(keypoints1.get(i).x())
          .isWithin(KEYPOINTS_DIFF_TOLERANCE)
          .of(keypoints2.get(i).x());
      assertThat(keypoints1.get(i).y())
          .isWithin(KEYPOINTS_DIFF_TOLERANCE)
          .of(keypoints2.get(i).y());
    }
  }
}
