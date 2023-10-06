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

package com.google.mediapipe.tasks.vision.objectdetector;

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
import com.google.mediapipe.tasks.components.containers.Category;
import com.google.mediapipe.tasks.components.containers.Detection;
import com.google.mediapipe.tasks.core.BaseOptions;
import com.google.mediapipe.tasks.core.TestUtils;
import com.google.mediapipe.tasks.vision.core.ImageProcessingOptions;
import com.google.mediapipe.tasks.vision.core.RunningMode;
import com.google.mediapipe.tasks.vision.objectdetector.ObjectDetector.ObjectDetectorOptions;
import java.io.InputStream;
import java.nio.ByteBuffer;
import java.util.Arrays;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Suite;
import org.junit.runners.Suite.SuiteClasses;

/** Test for {@link ObjectDetector}. */
@RunWith(Suite.class)
@SuiteClasses({ObjectDetectorTest.General.class, ObjectDetectorTest.RunningModeTest.class})
public class ObjectDetectorTest {
  private static final String MODEL_FILE = "coco_ssd_mobilenet_v1_1.0_quant_2018_06_29.tflite";
  private static final String NO_NMS_MODEL_FILE = "efficientdet_lite0_fp16_no_nms.tflite";
  private static final String CAT_AND_DOG_IMAGE = "cats_and_dogs.jpg";
  private static final String CAT_AND_DOG_ROTATED_IMAGE = "cats_and_dogs_rotated.jpg";
  private static final int IMAGE_WIDTH = 1200;
  private static final int IMAGE_HEIGHT = 600;
  private static final float CAT_SCORE = 0.69f;
  private static final RectF CAT_BOUNDING_BOX = new RectF(611, 164, 993, 596);
  // TODO: Figure out why android_x86 and android_arm tests have slightly different
  // scores (0.6875 vs 0.69921875).
  private static final float SCORE_DIFF_TOLERANCE = 0.01f;
  private static final float PIXEL_DIFF_TOLERANCE = 5.0f;

  @RunWith(AndroidJUnit4.class)
  public static final class General extends ObjectDetectorTest {

    @Test
    public void detect_successWithValidModels() throws Exception {
      ObjectDetectorOptions options =
          ObjectDetectorOptions.builder()
              .setBaseOptions(BaseOptions.builder().setModelAssetPath(MODEL_FILE).build())
              .setMaxResults(1)
              .build();
      ObjectDetector objectDetector =
          ObjectDetector.createFromOptions(ApplicationProvider.getApplicationContext(), options);
      ObjectDetectorResult results = objectDetector.detect(getImageFromAsset(CAT_AND_DOG_IMAGE));
      assertContainsOnlyCat(results, CAT_BOUNDING_BOX, CAT_SCORE);
    }

    @Test
    public void detect_successWithNoOptions() throws Exception {
      ObjectDetector objectDetector =
          ObjectDetector.createFromFile(ApplicationProvider.getApplicationContext(), MODEL_FILE);
      ObjectDetectorResult results = objectDetector.detect(getImageFromAsset(CAT_AND_DOG_IMAGE));
      // Check if the object with the highest score is cat.
      assertIsCat(results.detections().get(0).categories().get(0), CAT_SCORE);
    }

    @Test
    public void detect_succeedsWithMaxResultsOption() throws Exception {
      ObjectDetectorOptions options =
          ObjectDetectorOptions.builder()
              .setBaseOptions(BaseOptions.builder().setModelAssetPath(MODEL_FILE).build())
              .setMaxResults(8)
              .build();
      ObjectDetector objectDetector =
          ObjectDetector.createFromOptions(ApplicationProvider.getApplicationContext(), options);
      ObjectDetectorResult results = objectDetector.detect(getImageFromAsset(CAT_AND_DOG_IMAGE));
      // results should have 8 detected objects because maxResults was set to 8.
      assertThat(results.detections()).hasSize(8);
    }

    @Test
    public void detect_succeedsWithScoreThresholdOption() throws Exception {
      ObjectDetectorOptions options =
          ObjectDetectorOptions.builder()
              .setBaseOptions(BaseOptions.builder().setModelAssetPath(MODEL_FILE).build())
              .setScoreThreshold(0.68f)
              .build();
      ObjectDetector objectDetector =
          ObjectDetector.createFromOptions(ApplicationProvider.getApplicationContext(), options);
      ObjectDetectorResult results = objectDetector.detect(getImageFromAsset(CAT_AND_DOG_IMAGE));
      // The score threshold should block all other other objects, except cat.
      assertContainsOnlyCat(results, CAT_BOUNDING_BOX, CAT_SCORE);
    }

    @Test
    public void detect_succeedsWithNoObjectDetected() throws Exception {
      ObjectDetectorOptions options =
          ObjectDetectorOptions.builder()
              .setBaseOptions(BaseOptions.builder().setModelAssetPath(NO_NMS_MODEL_FILE).build())
              .setScoreThreshold(1.0f)
              .build();
      ObjectDetector objectDetector =
          ObjectDetector.createFromOptions(ApplicationProvider.getApplicationContext(), options);
      ObjectDetectorResult results = objectDetector.detect(getImageFromAsset(CAT_AND_DOG_IMAGE));
      // The score threshold should block objects.
      assertThat(results.detections()).isEmpty();
    }

    @Test
    public void detect_succeedsWithAllowListOption() throws Exception {
      ObjectDetectorOptions options =
          ObjectDetectorOptions.builder()
              .setBaseOptions(BaseOptions.builder().setModelAssetPath(MODEL_FILE).build())
              .setCategoryAllowlist(Arrays.asList("cat"))
              .build();
      ObjectDetector objectDetector =
          ObjectDetector.createFromOptions(ApplicationProvider.getApplicationContext(), options);
      ObjectDetectorResult results = objectDetector.detect(getImageFromAsset(CAT_AND_DOG_IMAGE));
      // Because of the allowlist, results should only contain cat, and there are 6 detected
      // bounding boxes of cats in CAT_AND_DOG_IMAGE.
      assertThat(results.detections()).hasSize(5);
    }

    @Test
    public void detect_succeedsWithDenyListOption() throws Exception {
      ObjectDetectorOptions options =
          ObjectDetectorOptions.builder()
              .setBaseOptions(BaseOptions.builder().setModelAssetPath(MODEL_FILE).build())
              .setCategoryDenylist(Arrays.asList("cat"))
              .build();
      ObjectDetector objectDetector =
          ObjectDetector.createFromOptions(ApplicationProvider.getApplicationContext(), options);
      ObjectDetectorResult results = objectDetector.detect(getImageFromAsset(CAT_AND_DOG_IMAGE));
      // Because of the denylist, the highest result is not cat anymore.
      assertThat(results.detections().get(0).categories().get(0).categoryName())
          .isNotEqualTo("cat");
    }

    @Test
    public void detect_succeedsWithModelFileObject() throws Exception {
      ObjectDetector objectDetector =
          ObjectDetector.createFromFile(
              ApplicationProvider.getApplicationContext(),
              TestUtils.loadFile(ApplicationProvider.getApplicationContext(), MODEL_FILE));
      ObjectDetectorResult results = objectDetector.detect(getImageFromAsset(CAT_AND_DOG_IMAGE));
      // Check if the object with the highest score is cat.
      assertIsCat(results.detections().get(0).categories().get(0), CAT_SCORE);
    }

    @Test
    public void detect_succeedsWithModelBuffer() throws Exception {
      ObjectDetector objectDetector =
          ObjectDetector.createFromBuffer(
              ApplicationProvider.getApplicationContext(),
              TestUtils.loadToDirectByteBuffer(
                  ApplicationProvider.getApplicationContext(), MODEL_FILE));
      ObjectDetectorResult results = objectDetector.detect(getImageFromAsset(CAT_AND_DOG_IMAGE));
      // Check if the object with the highest score is cat.
      assertIsCat(results.detections().get(0).categories().get(0), CAT_SCORE);
    }

    @Test
    public void detect_succeedsWithModelBufferAndOptions() throws Exception {
      ObjectDetectorOptions options =
          ObjectDetectorOptions.builder()
              .setBaseOptions(
                  BaseOptions.builder()
                      .setModelAssetBuffer(
                          TestUtils.loadToDirectByteBuffer(
                              ApplicationProvider.getApplicationContext(), MODEL_FILE))
                      .build())
              .setMaxResults(1)
              .build();
      ObjectDetector objectDetector =
          ObjectDetector.createFromOptions(ApplicationProvider.getApplicationContext(), options);
      ObjectDetectorResult results = objectDetector.detect(getImageFromAsset(CAT_AND_DOG_IMAGE));
      assertContainsOnlyCat(results, CAT_BOUNDING_BOX, CAT_SCORE);
    }

    @Test
    public void create_failsWithMissingModel() throws Exception {
      String nonexistentFile = "/path/to/non/existent/file";
      MediaPipeException exception =
          assertThrows(
              MediaPipeException.class,
              () ->
                  ObjectDetector.createFromFile(
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
                  ObjectDetector.createFromBuffer(
                      ApplicationProvider.getApplicationContext(), modelBuffer));

      assertThat(exception)
          .hasMessageThat()
          .contains("The model buffer should be either a direct ByteBuffer or a MappedByteBuffer.");
    }

    @Test
    public void detect_failsWithBothAllowAndDenyListOption() throws Exception {
      ObjectDetectorOptions options =
          ObjectDetectorOptions.builder()
              .setBaseOptions(BaseOptions.builder().setModelAssetPath(MODEL_FILE).build())
              .setCategoryAllowlist(Arrays.asList("cat"))
              .setCategoryDenylist(Arrays.asList("dog"))
              .build();
      MediaPipeException exception =
          assertThrows(
              MediaPipeException.class,
              () ->
                  ObjectDetector.createFromOptions(
                      ApplicationProvider.getApplicationContext(), options));
      assertThat(exception)
          .hasMessageThat()
          .contains("`category_allowlist` and `category_denylist` are mutually exclusive options.");
    }

    @Test
    public void detect_succeedsWithRotation() throws Exception {
      ObjectDetectorOptions options =
          ObjectDetectorOptions.builder()
              .setBaseOptions(BaseOptions.builder().setModelAssetPath(MODEL_FILE).build())
              .setMaxResults(1)
              .build();
      ObjectDetector objectDetector =
          ObjectDetector.createFromOptions(ApplicationProvider.getApplicationContext(), options);
      ImageProcessingOptions imageProcessingOptions =
          ImageProcessingOptions.builder().setRotationDegrees(-90).build();
      ObjectDetectorResult results =
          objectDetector.detect(
              getImageFromAsset(CAT_AND_DOG_ROTATED_IMAGE), imageProcessingOptions);

      assertContainsOnlyCat(results, new RectF(0.0f, 608.0f, 439.0f, 995.0f), 0.69921875f);
    }

    @Test
    public void detect_failsWithRegionOfInterest() throws Exception {
      ObjectDetectorOptions options =
          ObjectDetectorOptions.builder()
              .setBaseOptions(BaseOptions.builder().setModelAssetPath(MODEL_FILE).build())
              .build();
      ObjectDetector objectDetector =
          ObjectDetector.createFromOptions(ApplicationProvider.getApplicationContext(), options);
      ImageProcessingOptions imageProcessingOptions =
          ImageProcessingOptions.builder().setRegionOfInterest(new RectF(0, 0, 1, 1)).build();
      IllegalArgumentException exception =
          assertThrows(
              IllegalArgumentException.class,
              () ->
                  objectDetector.detect(
                      getImageFromAsset(CAT_AND_DOG_IMAGE), imageProcessingOptions));
      assertThat(exception)
          .hasMessageThat()
          .contains("ObjectDetector doesn't support region-of-interest");
    }

    // TODO: Implement detect_succeedsWithFloatImages, detect_succeedsWithOrientation,
    // detect_succeedsWithNumThreads, detect_successWithNumThreadsFromBaseOptions,
    // detect_failsWithInvalidNegativeNumThreads, detect_failsWithInvalidNumThreadsAsZero.
  }

  @RunWith(AndroidJUnit4.class)
  public static final class RunningModeTest extends ObjectDetectorTest {

    @Test
    public void create_failsWithIllegalResultListenerInNonLiveStreamMode() throws Exception {
      for (RunningMode mode : new RunningMode[] {RunningMode.IMAGE, RunningMode.VIDEO}) {
        IllegalArgumentException exception =
            assertThrows(
                IllegalArgumentException.class,
                () ->
                    ObjectDetectorOptions.builder()
                        .setBaseOptions(BaseOptions.builder().setModelAssetPath(MODEL_FILE).build())
                        .setRunningMode(mode)
                        .setResultListener((ObjectDetectorResult, inputImage) -> {})
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
                  ObjectDetectorOptions.builder()
                      .setBaseOptions(BaseOptions.builder().setModelAssetPath(MODEL_FILE).build())
                      .setRunningMode(RunningMode.LIVE_STREAM)
                      .build());
      assertThat(exception)
          .hasMessageThat()
          .contains("a user-defined result listener must be provided");
    }

    @Test
    public void detect_failsWithCallingWrongApiInImageMode() throws Exception {
      ObjectDetectorOptions options =
          ObjectDetectorOptions.builder()
              .setBaseOptions(BaseOptions.builder().setModelAssetPath(MODEL_FILE).build())
              .setRunningMode(RunningMode.IMAGE)
              .build();

      ObjectDetector objectDetector =
          ObjectDetector.createFromOptions(ApplicationProvider.getApplicationContext(), options);
      MediaPipeException exception =
          assertThrows(
              MediaPipeException.class,
              () ->
                  objectDetector.detectForVideo(
                      getImageFromAsset(CAT_AND_DOG_IMAGE), /* timestampsMs= */ 0));
      assertThat(exception).hasMessageThat().contains("not initialized with the video mode");
      exception =
          assertThrows(
              MediaPipeException.class,
              () ->
                  objectDetector.detectAsync(
                      getImageFromAsset(CAT_AND_DOG_IMAGE), /* timestampsMs= */ 0));
      assertThat(exception).hasMessageThat().contains("not initialized with the live stream mode");
    }

    @Test
    public void detect_failsWithCallingWrongApiInVideoMode() throws Exception {
      ObjectDetectorOptions options =
          ObjectDetectorOptions.builder()
              .setBaseOptions(BaseOptions.builder().setModelAssetPath(MODEL_FILE).build())
              .setRunningMode(RunningMode.VIDEO)
              .build();

      ObjectDetector objectDetector =
          ObjectDetector.createFromOptions(ApplicationProvider.getApplicationContext(), options);
      MediaPipeException exception =
          assertThrows(
              MediaPipeException.class,
              () -> objectDetector.detect(getImageFromAsset(CAT_AND_DOG_IMAGE)));
      assertThat(exception).hasMessageThat().contains("not initialized with the image mode");
      exception =
          assertThrows(
              MediaPipeException.class,
              () ->
                  objectDetector.detectAsync(
                      getImageFromAsset(CAT_AND_DOG_IMAGE), /* timestampsMs= */ 0));
      assertThat(exception).hasMessageThat().contains("not initialized with the live stream mode");
    }

    @Test
    public void detect_failsWithCallingWrongApiInLiveSteamMode() throws Exception {
      ObjectDetectorOptions options =
          ObjectDetectorOptions.builder()
              .setBaseOptions(BaseOptions.builder().setModelAssetPath(MODEL_FILE).build())
              .setRunningMode(RunningMode.LIVE_STREAM)
              .setResultListener((ObjectDetectorResult, inputImage) -> {})
              .build();

      ObjectDetector objectDetector =
          ObjectDetector.createFromOptions(ApplicationProvider.getApplicationContext(), options);

      MediaPipeException exception =
          assertThrows(
              MediaPipeException.class,
              () -> objectDetector.detect(getImageFromAsset(CAT_AND_DOG_IMAGE)));
      assertThat(exception).hasMessageThat().contains("not initialized with the image mode");
      exception =
          assertThrows(
              MediaPipeException.class,
              () ->
                  objectDetector.detectForVideo(
                      getImageFromAsset(CAT_AND_DOG_IMAGE), /* timestampsMs= */ 0));
      assertThat(exception).hasMessageThat().contains("not initialized with the video mode");
    }

    @Test
    public void detect_successWithImageMode() throws Exception {
      ObjectDetectorOptions options =
          ObjectDetectorOptions.builder()
              .setBaseOptions(BaseOptions.builder().setModelAssetPath(MODEL_FILE).build())
              .setRunningMode(RunningMode.IMAGE)
              .setMaxResults(1)
              .build();
      ObjectDetector objectDetector =
          ObjectDetector.createFromOptions(ApplicationProvider.getApplicationContext(), options);
      ObjectDetectorResult results = objectDetector.detect(getImageFromAsset(CAT_AND_DOG_IMAGE));
      assertContainsOnlyCat(results, CAT_BOUNDING_BOX, CAT_SCORE);
    }

    @Test
    public void detect_successWithVideoMode() throws Exception {
      ObjectDetectorOptions options =
          ObjectDetectorOptions.builder()
              .setBaseOptions(BaseOptions.builder().setModelAssetPath(MODEL_FILE).build())
              .setRunningMode(RunningMode.VIDEO)
              .setMaxResults(1)
              .build();
      ObjectDetector objectDetector =
          ObjectDetector.createFromOptions(ApplicationProvider.getApplicationContext(), options);
      for (int i = 0; i < 3; i++) {
        ObjectDetectorResult results =
            objectDetector.detectForVideo(
                getImageFromAsset(CAT_AND_DOG_IMAGE), /* timestampsMs= */ i);
        assertContainsOnlyCat(results, CAT_BOUNDING_BOX, CAT_SCORE);
      }
    }

    @Test
    public void detect_failsWithOutOfOrderInputTimestamps() throws Exception {
      MPImage image = getImageFromAsset(CAT_AND_DOG_IMAGE);
      ObjectDetectorOptions options =
          ObjectDetectorOptions.builder()
              .setBaseOptions(BaseOptions.builder().setModelAssetPath(MODEL_FILE).build())
              .setRunningMode(RunningMode.LIVE_STREAM)
              .setResultListener(
                  (ObjectDetectorResult, inputImage) -> {
                    assertContainsOnlyCat(ObjectDetectorResult, CAT_BOUNDING_BOX, CAT_SCORE);
                    assertImageSizeIsExpected(inputImage);
                  })
              .setMaxResults(1)
              .build();
      try (ObjectDetector objectDetector =
          ObjectDetector.createFromOptions(ApplicationProvider.getApplicationContext(), options)) {
        objectDetector.detectAsync(image, /* timestampsMs= */ 1);
        MediaPipeException exception =
            assertThrows(
                MediaPipeException.class,
                () -> objectDetector.detectAsync(image, /* timestampsMs= */ 0));
        assertThat(exception)
            .hasMessageThat()
            .contains("having a smaller timestamp than the processed timestamp");
      }
    }

    @Test
    public void detect_successWithLiveSteamMode() throws Exception {
      MPImage image = getImageFromAsset(CAT_AND_DOG_IMAGE);
      ObjectDetectorOptions options =
          ObjectDetectorOptions.builder()
              .setBaseOptions(BaseOptions.builder().setModelAssetPath(MODEL_FILE).build())
              .setRunningMode(RunningMode.LIVE_STREAM)
              .setResultListener(
                  (ObjectDetectorResult, inputImage) -> {
                    assertContainsOnlyCat(ObjectDetectorResult, CAT_BOUNDING_BOX, CAT_SCORE);
                    assertImageSizeIsExpected(inputImage);
                  })
              .setMaxResults(1)
              .build();
      try (ObjectDetector objectDetector =
          ObjectDetector.createFromOptions(ApplicationProvider.getApplicationContext(), options)) {
        for (int i = 0; i < 3; i++) {
          objectDetector.detectAsync(image, /* timestampsMs= */ i);
        }
      }
    }
  }

  @Test
  @SuppressWarnings("deprecation")
  public void detect_canUseDeprecatedApi() throws Exception {
    ObjectDetector objectDetector =
        ObjectDetector.createFromFile(ApplicationProvider.getApplicationContext(), MODEL_FILE);
    ObjectDetectionResult results = objectDetector.detect(getImageFromAsset(CAT_AND_DOG_IMAGE));
    // Check if the object with the highest score is cat.
    assertIsCat(results.detections().get(0).categories().get(0), CAT_SCORE);
  }

  private static MPImage getImageFromAsset(String filePath) throws Exception {
    AssetManager assetManager = ApplicationProvider.getApplicationContext().getAssets();
    InputStream istr = assetManager.open(filePath);
    return new BitmapImageBuilder(BitmapFactory.decodeStream(istr)).build();
  }

  // Checks if results has one and only detection result, which is a cat.
  private static void assertContainsOnlyCat(
      ObjectDetectorResult result, RectF expectedBoundingBox, float expectedScore) {
    assertThat(result.detections()).hasSize(1);
    Detection catResult = result.detections().get(0);
    assertApproximatelyEqualBoundingBoxes(catResult.boundingBox(), expectedBoundingBox);
    // We only support one category for each detected object at this point.
    assertIsCat(catResult.categories().get(0), expectedScore);
  }

  private static void assertIsCat(Category category, float expectedScore) {
    assertThat(category.categoryName()).isEqualTo("cat");
    // coco_ssd_mobilenet_v1_1.0_quant_2018_06_29.tflite does not support label locale.
    assertThat(category.displayName()).isEmpty();
    assertThat((double) category.score()).isWithin(SCORE_DIFF_TOLERANCE).of(expectedScore);
    assertThat(category.index()).isEqualTo(-1);
  }

  private static void assertApproximatelyEqualBoundingBoxes(
      RectF boundingBox1, RectF boundingBox2) {
    assertThat(boundingBox1.left).isWithin(PIXEL_DIFF_TOLERANCE).of(boundingBox2.left);
    assertThat(boundingBox1.top).isWithin(PIXEL_DIFF_TOLERANCE).of(boundingBox2.top);
    assertThat(boundingBox1.right).isWithin(PIXEL_DIFF_TOLERANCE).of(boundingBox2.right);
    assertThat(boundingBox1.bottom).isWithin(PIXEL_DIFF_TOLERANCE).of(boundingBox2.bottom);
  }

  private static void assertImageSizeIsExpected(MPImage inputImage) {
    assertThat(inputImage).isNotNull();
    assertThat(inputImage.getWidth()).isEqualTo(IMAGE_WIDTH);
    assertThat(inputImage.getHeight()).isEqualTo(IMAGE_HEIGHT);
  }
}
