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

package com.google.mediapipe.tasks.vision.imageclassifier;

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
import com.google.mediapipe.tasks.core.BaseOptions;
import com.google.mediapipe.tasks.core.TestUtils;
import com.google.mediapipe.tasks.vision.core.ImageProcessingOptions;
import com.google.mediapipe.tasks.vision.core.RunningMode;
import com.google.mediapipe.tasks.vision.imageclassifier.ImageClassifier.ImageClassifierOptions;
import java.io.InputStream;
import java.nio.ByteBuffer;
import java.util.Arrays;
import java.util.List;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Suite;
import org.junit.runners.Suite.SuiteClasses;

/** Test for {@link ImageClassifier}/ */
@RunWith(Suite.class)
@SuiteClasses({ImageClassifierTest.General.class, ImageClassifierTest.RunningModeTest.class})
public class ImageClassifierTest {
  private static final String FLOAT_MODEL_FILE = "mobilenet_v2_1.0_224.tflite";
  private static final String QUANTIZED_MODEL_FILE = "mobilenet_v1_0.25_224_quant.tflite";
  private static final String BURGER_IMAGE = "burger.jpg";
  private static final String BURGER_ROTATED_IMAGE = "burger_rotated.jpg";
  private static final String MULTI_OBJECTS_IMAGE = "multi_objects.jpg";
  private static final String MULTI_OBJECTS_ROTATED_IMAGE = "multi_objects_rotated.jpg";

  @RunWith(AndroidJUnit4.class)
  public static final class General extends ImageClassifierTest {

    @Test
    public void options_failsWithNegativeMaxResults() throws Exception {
      IllegalArgumentException exception =
          assertThrows(
              IllegalArgumentException.class,
              () ->
                  ImageClassifierOptions.builder()
                      .setBaseOptions(
                          BaseOptions.builder().setModelAssetPath(FLOAT_MODEL_FILE).build())
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
                  ImageClassifierOptions.builder()
                      .setBaseOptions(
                          BaseOptions.builder().setModelAssetPath(FLOAT_MODEL_FILE).build())
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
                  ImageClassifier.createFromFile(
                      ApplicationProvider.getApplicationContext(), nonExistentFile));
      assertThat(exception).hasMessageThat().contains(nonExistentFile);
    }

    @Test
    public void create_failsWithInvalidModelBuffer() throws Exception {
      // Create a non-direct model ByteBuffer.
      ByteBuffer modelBuffer =
          TestUtils.loadToNonDirectByteBuffer(
              ApplicationProvider.getApplicationContext(), FLOAT_MODEL_FILE);

      IllegalArgumentException exception =
          assertThrows(
              IllegalArgumentException.class,
              () ->
                  ImageClassifier.createFromBuffer(
                      ApplicationProvider.getApplicationContext(), modelBuffer));

      assertThat(exception)
          .hasMessageThat()
          .contains("The model buffer should be either a direct ByteBuffer or a MappedByteBuffer.");
    }

    @Test
    public void classify_succeedsWithNoOptions() throws Exception {
      ImageClassifier imageClassifier =
          ImageClassifier.createFromFile(
              ApplicationProvider.getApplicationContext(), FLOAT_MODEL_FILE);
      ImageClassifierResult results = imageClassifier.classify(getImageFromAsset(BURGER_IMAGE));

      assertHasOneHead(results);
      assertThat(results.classificationResult().classifications().get(0).categories())
          .hasSize(1001);
      assertThat(results.classificationResult().classifications().get(0).categories().get(0))
          .isEqualTo(Category.create(0.7952058f, 934, "cheeseburger", ""));
    }

    @Test
    public void classify_succeedsWithFloatModel() throws Exception {
      ImageClassifierOptions options =
          ImageClassifierOptions.builder()
              .setBaseOptions(BaseOptions.builder().setModelAssetPath(FLOAT_MODEL_FILE).build())
              .setMaxResults(3)
              .build();
      ImageClassifier imageClassifier =
          ImageClassifier.createFromOptions(ApplicationProvider.getApplicationContext(), options);
      ImageClassifierResult results = imageClassifier.classify(getImageFromAsset(BURGER_IMAGE));

      assertHasOneHead(results);
      assertCategoriesAre(
          results,
          Arrays.asList(
              Category.create(0.7952058f, 934, "cheeseburger", ""),
              Category.create(0.027329788f, 932, "bagel", ""),
              Category.create(0.019334773f, 925, "guacamole", "")));
    }

    @Test
    public void classify_succeedsWithQuantizedModel() throws Exception {
      ImageClassifierOptions options =
          ImageClassifierOptions.builder()
              .setBaseOptions(BaseOptions.builder().setModelAssetPath(QUANTIZED_MODEL_FILE).build())
              .setMaxResults(1)
              .build();
      ImageClassifier imageClassifier =
          ImageClassifier.createFromOptions(ApplicationProvider.getApplicationContext(), options);
      ImageClassifierResult results = imageClassifier.classify(getImageFromAsset(BURGER_IMAGE));

      assertHasOneHead(results);
      assertCategoriesAre(
          results, Arrays.asList(Category.create(0.97265625f, 934, "cheeseburger", "")));
    }

    @Test
    public void classify_succeedsWithScoreThreshold() throws Exception {
      ImageClassifierOptions options =
          ImageClassifierOptions.builder()
              .setBaseOptions(BaseOptions.builder().setModelAssetPath(FLOAT_MODEL_FILE).build())
              .setScoreThreshold(0.02f)
              .build();
      ImageClassifier imageClassifier =
          ImageClassifier.createFromOptions(ApplicationProvider.getApplicationContext(), options);
      ImageClassifierResult results = imageClassifier.classify(getImageFromAsset(BURGER_IMAGE));

      assertHasOneHead(results);
      assertCategoriesAre(
          results,
          Arrays.asList(
              Category.create(0.7952058f, 934, "cheeseburger", ""),
              Category.create(0.027329788f, 932, "bagel", "")));
    }

    @Test
    public void classify_succeedsWithAllowlist() throws Exception {
      ImageClassifierOptions options =
          ImageClassifierOptions.builder()
              .setBaseOptions(BaseOptions.builder().setModelAssetPath(FLOAT_MODEL_FILE).build())
              .setCategoryAllowlist(Arrays.asList("cheeseburger", "guacamole", "meat loaf"))
              .build();
      ImageClassifier imageClassifier =
          ImageClassifier.createFromOptions(ApplicationProvider.getApplicationContext(), options);
      ImageClassifierResult results = imageClassifier.classify(getImageFromAsset(BURGER_IMAGE));

      assertHasOneHead(results);
      assertCategoriesAre(
          results,
          Arrays.asList(
              Category.create(0.7952058f, 934, "cheeseburger", ""),
              Category.create(0.019334773f, 925, "guacamole", ""),
              Category.create(0.006279315f, 963, "meat loaf", "")));
    }

    @Test
    public void classify_succeedsWithDenylist() throws Exception {
      ImageClassifierOptions options =
          ImageClassifierOptions.builder()
              .setBaseOptions(BaseOptions.builder().setModelAssetPath(FLOAT_MODEL_FILE).build())
              .setMaxResults(3)
              .setCategoryDenylist(Arrays.asList("bagel"))
              .build();
      ImageClassifier imageClassifier =
          ImageClassifier.createFromOptions(ApplicationProvider.getApplicationContext(), options);
      ImageClassifierResult results = imageClassifier.classify(getImageFromAsset(BURGER_IMAGE));

      assertHasOneHead(results);
      assertCategoriesAre(
          results,
          Arrays.asList(
              Category.create(0.7952058f, 934, "cheeseburger", ""),
              Category.create(0.019334773f, 925, "guacamole", ""),
              Category.create(0.006279315f, 963, "meat loaf", "")));
    }

    @Test
    public void classify_succeedsWithRegionOfInterest() throws Exception {
      ImageClassifierOptions options =
          ImageClassifierOptions.builder()
              .setBaseOptions(BaseOptions.builder().setModelAssetPath(FLOAT_MODEL_FILE).build())
              .setMaxResults(1)
              .build();
      ImageClassifier imageClassifier =
          ImageClassifier.createFromOptions(ApplicationProvider.getApplicationContext(), options);
      // RectF around the soccer ball.
      RectF roi = new RectF(0.450f, 0.308f, 0.614f, 0.734f);
      ImageProcessingOptions imageProcessingOptions =
          ImageProcessingOptions.builder().setRegionOfInterest(roi).build();
      ImageClassifierResult results =
          imageClassifier.classify(getImageFromAsset(MULTI_OBJECTS_IMAGE), imageProcessingOptions);

      assertHasOneHead(results);
      assertCategoriesAre(
          results, Arrays.asList(Category.create(0.9969325f, 806, "soccer ball", "")));
    }

    @Test
    public void classify_succeedsWithRotation() throws Exception {
      ImageClassifierOptions options =
          ImageClassifierOptions.builder()
              .setBaseOptions(BaseOptions.builder().setModelAssetPath(FLOAT_MODEL_FILE).build())
              .setMaxResults(3)
              .build();
      ImageClassifier imageClassifier =
          ImageClassifier.createFromOptions(ApplicationProvider.getApplicationContext(), options);
      ImageProcessingOptions imageProcessingOptions =
          ImageProcessingOptions.builder().setRotationDegrees(-90).build();
      ImageClassifierResult results =
          imageClassifier.classify(getImageFromAsset(BURGER_ROTATED_IMAGE), imageProcessingOptions);

      assertHasOneHead(results);
      assertCategoriesAre(
          results,
          Arrays.asList(
              Category.create(0.75369555f, 934, "cheeseburger", ""),
              Category.create(0.029219573f, 925, "guacamole", ""),
              Category.create(0.028840661f, 932, "bagel", "")));
    }

    @Test
    public void classify_succeedsWithRegionOfInterestAndRotation() throws Exception {
      ImageClassifierOptions options =
          ImageClassifierOptions.builder()
              .setBaseOptions(BaseOptions.builder().setModelAssetPath(FLOAT_MODEL_FILE).build())
              .setMaxResults(1)
              .build();
      ImageClassifier imageClassifier =
          ImageClassifier.createFromOptions(ApplicationProvider.getApplicationContext(), options);
      // RectF around the soccer ball.
      RectF roi = new RectF(0.2655f, 0.45f, 0.6925f, 0.614f);
      ImageProcessingOptions imageProcessingOptions =
          ImageProcessingOptions.builder().setRegionOfInterest(roi).setRotationDegrees(-90).build();
      ImageClassifierResult results =
          imageClassifier.classify(
              getImageFromAsset(MULTI_OBJECTS_ROTATED_IMAGE), imageProcessingOptions);

      assertHasOneHead(results);
      assertCategoriesAre(
          results, Arrays.asList(Category.create(0.99730396f, 806, "soccer ball", "")));
    }
  }

  @RunWith(AndroidJUnit4.class)
  public static final class RunningModeTest extends ImageClassifierTest {

    @Test
    public void create_failsWithIllegalResultListenerInNonLiveStreamMode() throws Exception {
      for (RunningMode mode : new RunningMode[] {RunningMode.IMAGE, RunningMode.VIDEO}) {
        IllegalArgumentException exception =
            assertThrows(
                IllegalArgumentException.class,
                () ->
                    ImageClassifierOptions.builder()
                        .setBaseOptions(
                            BaseOptions.builder().setModelAssetPath(FLOAT_MODEL_FILE).build())
                        .setRunningMode(mode)
                        .setResultListener((imageClassificationResult, inputImage) -> {})
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
                  ImageClassifierOptions.builder()
                      .setBaseOptions(
                          BaseOptions.builder().setModelAssetPath(FLOAT_MODEL_FILE).build())
                      .setRunningMode(RunningMode.LIVE_STREAM)
                      .build());
      assertThat(exception)
          .hasMessageThat()
          .contains("a user-defined result listener must be provided");
    }

    @Test
    public void classify_failsWithCallingWrongApiInImageMode() throws Exception {
      ImageClassifierOptions options =
          ImageClassifierOptions.builder()
              .setBaseOptions(BaseOptions.builder().setModelAssetPath(FLOAT_MODEL_FILE).build())
              .setRunningMode(RunningMode.IMAGE)
              .build();

      ImageClassifier imageClassifier =
          ImageClassifier.createFromOptions(ApplicationProvider.getApplicationContext(), options);
      MediaPipeException exception =
          assertThrows(
              MediaPipeException.class,
              () ->
                  imageClassifier.classifyForVideo(
                      getImageFromAsset(BURGER_IMAGE), /* timestampMs= */ 0));
      assertThat(exception).hasMessageThat().contains("not initialized with the video mode");
      exception =
          assertThrows(
              MediaPipeException.class,
              () ->
                  imageClassifier.classifyAsync(
                      getImageFromAsset(BURGER_IMAGE), /* timestampMs= */ 0));
      assertThat(exception).hasMessageThat().contains("not initialized with the live stream mode");
    }

    @Test
    public void classify_failsWithCallingWrongApiInVideoMode() throws Exception {
      ImageClassifierOptions options =
          ImageClassifierOptions.builder()
              .setBaseOptions(BaseOptions.builder().setModelAssetPath(FLOAT_MODEL_FILE).build())
              .setRunningMode(RunningMode.VIDEO)
              .build();

      ImageClassifier imageClassifier =
          ImageClassifier.createFromOptions(ApplicationProvider.getApplicationContext(), options);
      MediaPipeException exception =
          assertThrows(
              MediaPipeException.class,
              () -> imageClassifier.classify(getImageFromAsset(BURGER_IMAGE)));
      assertThat(exception).hasMessageThat().contains("not initialized with the image mode");
      exception =
          assertThrows(
              MediaPipeException.class,
              () ->
                  imageClassifier.classifyAsync(
                      getImageFromAsset(BURGER_IMAGE), /* timestampMs= */ 0));
      assertThat(exception).hasMessageThat().contains("not initialized with the live stream mode");
    }

    @Test
    public void classify_failsWithCallingWrongApiInLiveSteamMode() throws Exception {
      ImageClassifierOptions options =
          ImageClassifierOptions.builder()
              .setBaseOptions(BaseOptions.builder().setModelAssetPath(FLOAT_MODEL_FILE).build())
              .setRunningMode(RunningMode.LIVE_STREAM)
              .setResultListener((imageClassificationResult, inputImage) -> {})
              .build();

      ImageClassifier imageClassifier =
          ImageClassifier.createFromOptions(ApplicationProvider.getApplicationContext(), options);

      MediaPipeException exception =
          assertThrows(
              MediaPipeException.class,
              () -> imageClassifier.classify(getImageFromAsset(BURGER_IMAGE)));
      assertThat(exception).hasMessageThat().contains("not initialized with the image mode");
      exception =
          assertThrows(
              MediaPipeException.class,
              () ->
                  imageClassifier.classifyForVideo(
                      getImageFromAsset(BURGER_IMAGE), /* timestampMs= */ 0));
      assertThat(exception).hasMessageThat().contains("not initialized with the video mode");
    }

    @Test
    public void classify_succeedsWithImageMode() throws Exception {
      ImageClassifierOptions options =
          ImageClassifierOptions.builder()
              .setBaseOptions(BaseOptions.builder().setModelAssetPath(FLOAT_MODEL_FILE).build())
              .setMaxResults(1)
              .build();
      ImageClassifier imageClassifier =
          ImageClassifier.createFromOptions(ApplicationProvider.getApplicationContext(), options);
      ImageClassifierResult results = imageClassifier.classify(getImageFromAsset(BURGER_IMAGE));

      assertHasOneHead(results);
      assertCategoriesAre(
          results, Arrays.asList(Category.create(0.7952058f, 934, "cheeseburger", "")));
    }

    @Test
    public void classify_succeedsWithVideoMode() throws Exception {
      MPImage image = getImageFromAsset(BURGER_IMAGE);
      ImageClassifierOptions options =
          ImageClassifierOptions.builder()
              .setBaseOptions(BaseOptions.builder().setModelAssetPath(FLOAT_MODEL_FILE).build())
              .setMaxResults(1)
              .setRunningMode(RunningMode.VIDEO)
              .build();
      ImageClassifier imageClassifier =
          ImageClassifier.createFromOptions(ApplicationProvider.getApplicationContext(), options);
      for (int i = 0; i < 3; i++) {
        ImageClassifierResult results =
            imageClassifier.classifyForVideo(image, /* timestampMs= */ i);
        assertHasOneHead(results);
        assertCategoriesAre(
            results, Arrays.asList(Category.create(0.7952058f, 934, "cheeseburger", "")));
      }
    }

    @Test
    public void classify_failsWithOutOfOrderInputTimestamps() throws Exception {
      MPImage image = getImageFromAsset(BURGER_IMAGE);
      ImageClassifierOptions options =
          ImageClassifierOptions.builder()
              .setBaseOptions(BaseOptions.builder().setModelAssetPath(FLOAT_MODEL_FILE).build())
              .setMaxResults(1)
              .setRunningMode(RunningMode.LIVE_STREAM)
              .setResultListener(
                  (imageClassificationResult, inputImage) -> {
                    assertCategoriesAre(
                        imageClassificationResult,
                        Arrays.asList(Category.create(0.7952058f, 934, "cheeseburger", "")));
                    assertImageSizeIsExpected(inputImage);
                  })
              .build();
      try (ImageClassifier imageClassifier =
          ImageClassifier.createFromOptions(ApplicationProvider.getApplicationContext(), options)) {
        imageClassifier.classifyAsync(getImageFromAsset(BURGER_IMAGE), /* timestampMs= */ 1);
        MediaPipeException exception =
            assertThrows(
                MediaPipeException.class,
                () -> imageClassifier.classifyAsync(image, /* timestampMs= */ 0));
        assertThat(exception)
            .hasMessageThat()
            .contains("having a smaller timestamp than the processed timestamp");
      }
    }

    @Test
    public void classify_succeedsWithLiveStreamMode() throws Exception {
      MPImage image = getImageFromAsset(BURGER_IMAGE);
      ImageClassifierOptions options =
          ImageClassifierOptions.builder()
              .setBaseOptions(BaseOptions.builder().setModelAssetPath(FLOAT_MODEL_FILE).build())
              .setMaxResults(1)
              .setRunningMode(RunningMode.LIVE_STREAM)
              .setResultListener(
                  (imageClassificationResult, inputImage) -> {
                    assertCategoriesAre(
                        imageClassificationResult,
                        Arrays.asList(Category.create(0.7952058f, 934, "cheeseburger", "")));
                    assertImageSizeIsExpected(inputImage);
                  })
              .build();
      try (ImageClassifier imageClassifier =
          ImageClassifier.createFromOptions(ApplicationProvider.getApplicationContext(), options)) {
        for (int i = 0; i < 3; ++i) {
          imageClassifier.classifyAsync(image, /* timestampMs= */ i);
        }
      }
    }
  }

  private static MPImage getImageFromAsset(String filePath) throws Exception {
    AssetManager assetManager = ApplicationProvider.getApplicationContext().getAssets();
    InputStream istr = assetManager.open(filePath);
    return new BitmapImageBuilder(BitmapFactory.decodeStream(istr)).build();
  }

  private static void assertHasOneHead(ImageClassifierResult results) {
    assertThat(results.classificationResult().classifications()).hasSize(1);
    assertThat(results.classificationResult().classifications().get(0).headIndex()).isEqualTo(0);
    assertThat(results.classificationResult().classifications().get(0).headName().get())
        .isEqualTo("probability");
  }

  private static void assertCategoriesAre(
      ImageClassifierResult results, List<Category> categories) {
    assertThat(results.classificationResult().classifications().get(0).categories())
        .isEqualTo(categories);
  }

  private static void assertImageSizeIsExpected(MPImage inputImage) {
    assertThat(inputImage).isNotNull();
    assertThat(inputImage.getWidth()).isEqualTo(480);
    assertThat(inputImage.getHeight()).isEqualTo(325);
  }
}
