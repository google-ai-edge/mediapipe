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

package com.google.mediapipe.tasks.vision.imageembedder;

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
import com.google.mediapipe.tasks.core.BaseOptions;
import com.google.mediapipe.tasks.core.TestUtils;
import com.google.mediapipe.tasks.vision.core.ImageProcessingOptions;
import com.google.mediapipe.tasks.vision.core.RunningMode;
import com.google.mediapipe.tasks.vision.imageembedder.ImageEmbedder.ImageEmbedderOptions;
import java.io.InputStream;
import java.nio.ByteBuffer;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Suite;
import org.junit.runners.Suite.SuiteClasses;

/** Test for {@link ImageEmbedder}/ */
@RunWith(Suite.class)
@SuiteClasses({ImageEmbedderTest.General.class, ImageEmbedderTest.RunningModeTest.class})
public class ImageEmbedderTest {
  private static final String MOBILENET_EMBEDDER = "mobilenet_v3_small_100_224_embedder.tflite";
  private static final String BURGER_IMAGE = "burger.jpg";
  private static final String BURGER_CROP_IMAGE = "burger_crop.jpg";
  private static final String BURGER_ROTATED_IMAGE = "burger_rotated.jpg";

  private static final double DOUBLE_DIFF_TOLERANCE = 1e-4;

  @RunWith(AndroidJUnit4.class)
  public static final class General extends ImageEmbedderTest {

    @Test
    public void create_failsWithMissingModel() throws Exception {
      String nonExistentFile = "/path/to/non/existent/file";
      MediaPipeException exception =
          assertThrows(
              MediaPipeException.class,
              () ->
                  ImageEmbedder.createFromFile(
                      ApplicationProvider.getApplicationContext(), nonExistentFile));
      assertThat(exception).hasMessageThat().contains(nonExistentFile);
    }

    @Test
    public void create_failsWithInvalidModelBuffer() throws Exception {
      // Create a non-direct model ByteBuffer.
      ByteBuffer modelBuffer =
          TestUtils.loadToNonDirectByteBuffer(
              ApplicationProvider.getApplicationContext(), MOBILENET_EMBEDDER);

      IllegalArgumentException exception =
          assertThrows(
              IllegalArgumentException.class,
              () ->
                  ImageEmbedder.createFromBuffer(
                      ApplicationProvider.getApplicationContext(), modelBuffer));

      assertThat(exception)
          .hasMessageThat()
          .contains("The model buffer should be either a direct ByteBuffer or a MappedByteBuffer.");
    }

    @Test
    public void embed_succeedsWithNoOptions() throws Exception {
      ImageEmbedder imageEmbedder =
          ImageEmbedder.createFromFile(
              ApplicationProvider.getApplicationContext(), MOBILENET_EMBEDDER);
      ImageEmbedderResult result = imageEmbedder.embed(getImageFromAsset(BURGER_IMAGE));
      ImageEmbedderResult resultCrop = imageEmbedder.embed(getImageFromAsset(BURGER_CROP_IMAGE));

      // Check results.
      assertHasOneHeadAndCorrectDimension(result, /* quantized= */ false);
      assertHasOneHeadAndCorrectDimension(resultCrop, /* quantized= */ false);
      // Check similarity.
      double similarity =
          ImageEmbedder.cosineSimilarity(
              result.embeddingResult().embeddings().get(0),
              resultCrop.embeddingResult().embeddings().get(0));
      assertThat(similarity).isWithin(DOUBLE_DIFF_TOLERANCE).of(0.925272);
    }

    @Test
    public void embed_succeedsWithL2Normalization() throws Exception {
      BaseOptions baseOptions = BaseOptions.builder().setModelAssetPath(MOBILENET_EMBEDDER).build();
      ImageEmbedderOptions options =
          ImageEmbedderOptions.builder().setBaseOptions(baseOptions).setL2Normalize(true).build();

      ImageEmbedder imageEmbedder =
          ImageEmbedder.createFromOptions(ApplicationProvider.getApplicationContext(), options);
      ImageEmbedderResult result = imageEmbedder.embed(getImageFromAsset(BURGER_IMAGE));
      ImageEmbedderResult resultCrop = imageEmbedder.embed(getImageFromAsset(BURGER_CROP_IMAGE));

      // Check results.
      assertHasOneHeadAndCorrectDimension(result, /* quantized= */ false);
      assertHasOneHeadAndCorrectDimension(resultCrop, /* quantized= */ false);
      // Check similarity.
      double similarity =
          ImageEmbedder.cosineSimilarity(
              result.embeddingResult().embeddings().get(0),
              resultCrop.embeddingResult().embeddings().get(0));
      assertThat(similarity).isWithin(DOUBLE_DIFF_TOLERANCE).of(0.925272);
    }

    @Test
    public void embed_succeedsWithQuantization() throws Exception {
      BaseOptions baseOptions = BaseOptions.builder().setModelAssetPath(MOBILENET_EMBEDDER).build();
      ImageEmbedderOptions options =
          ImageEmbedderOptions.builder().setBaseOptions(baseOptions).setQuantize(true).build();

      ImageEmbedder imageEmbedder =
          ImageEmbedder.createFromOptions(ApplicationProvider.getApplicationContext(), options);
      ImageEmbedderResult result = imageEmbedder.embed(getImageFromAsset(BURGER_IMAGE));
      ImageEmbedderResult resultCrop = imageEmbedder.embed(getImageFromAsset(BURGER_CROP_IMAGE));

      // Check results.
      assertHasOneHeadAndCorrectDimension(result, /* quantized= */ true);
      assertHasOneHeadAndCorrectDimension(resultCrop, /* quantized= */ true);
      // Check similarity.
      double similarity =
          ImageEmbedder.cosineSimilarity(
              result.embeddingResult().embeddings().get(0),
              resultCrop.embeddingResult().embeddings().get(0));
      assertThat(similarity).isWithin(DOUBLE_DIFF_TOLERANCE).of(0.926776);
    }

    @Test
    public void embed_succeedsWithRegionOfInterest() throws Exception {
      ImageEmbedder imageEmbedder =
          ImageEmbedder.createFromFile(
              ApplicationProvider.getApplicationContext(), MOBILENET_EMBEDDER);
      // RectF around the region in "burger.jpg" corresponding to "burger_crop.jpg".
      RectF roi = new RectF(0.0f, 0.0f, 0.833333f, 1.0f);
      ImageProcessingOptions imageProcessingOptions =
          ImageProcessingOptions.builder().setRegionOfInterest(roi).build();
      ImageEmbedderResult resultRoi =
          imageEmbedder.embed(getImageFromAsset(BURGER_IMAGE), imageProcessingOptions);
      ImageEmbedderResult resultCrop = imageEmbedder.embed(getImageFromAsset(BURGER_CROP_IMAGE));

      // Check results.
      assertHasOneHeadAndCorrectDimension(resultRoi, /* quantized= */ false);
      assertHasOneHeadAndCorrectDimension(resultCrop, /* quantized= */ false);
      // Check similarity.
      double similarity =
          ImageEmbedder.cosineSimilarity(
              resultRoi.embeddingResult().embeddings().get(0),
              resultCrop.embeddingResult().embeddings().get(0));
      assertThat(similarity).isWithin(DOUBLE_DIFF_TOLERANCE).of(0.999931f);
    }

    @Test
    public void embed_succeedsWithRotation() throws Exception {
      ImageEmbedder imageEmbedder =
          ImageEmbedder.createFromFile(
              ApplicationProvider.getApplicationContext(), MOBILENET_EMBEDDER);
      ImageEmbedderResult result = imageEmbedder.embed(getImageFromAsset(BURGER_IMAGE));
      ImageProcessingOptions imageProcessingOptions =
          ImageProcessingOptions.builder().setRotationDegrees(-90).build();
      ImageEmbedderResult resultRotated =
          imageEmbedder.embed(getImageFromAsset(BURGER_ROTATED_IMAGE), imageProcessingOptions);

      // Check results.
      assertHasOneHeadAndCorrectDimension(result, /* quantized= */ false);
      assertHasOneHeadAndCorrectDimension(resultRotated, /* quantized= */ false);
      // Check similarity.
      double similarity =
          ImageEmbedder.cosineSimilarity(
              result.embeddingResult().embeddings().get(0),
              resultRotated.embeddingResult().embeddings().get(0));
      assertThat(similarity).isWithin(DOUBLE_DIFF_TOLERANCE).of(0.982316669f);
    }

    @Test
    public void embed_succeedsWithRegionOfInterestAndRotation() throws Exception {
      ImageEmbedder imageEmbedder =
          ImageEmbedder.createFromFile(
              ApplicationProvider.getApplicationContext(), MOBILENET_EMBEDDER);
      // RectF around the region in "burger_rotated.jpg" corresponding to "burger_crop.jpg".
      RectF roi = new RectF(0.0f, 0.0f, 1.0f, 0.833333f);
      ImageProcessingOptions imageProcessingOptions =
          ImageProcessingOptions.builder().setRegionOfInterest(roi).setRotationDegrees(-90).build();
      ImageEmbedderResult resultRoiRotated =
          imageEmbedder.embed(getImageFromAsset(BURGER_ROTATED_IMAGE), imageProcessingOptions);
      ImageEmbedderResult resultCrop = imageEmbedder.embed(getImageFromAsset(BURGER_CROP_IMAGE));

      // Check results.
      assertHasOneHeadAndCorrectDimension(resultRoiRotated, /* quantized= */ false);
      assertHasOneHeadAndCorrectDimension(resultCrop, /* quantized= */ false);
      // Check similarity.
      double similarity =
          ImageEmbedder.cosineSimilarity(
              resultRoiRotated.embeddingResult().embeddings().get(0),
              resultCrop.embeddingResult().embeddings().get(0));
      assertThat(similarity).isWithin(DOUBLE_DIFF_TOLERANCE).of(0.9745944861f);
    }
  }

  @RunWith(AndroidJUnit4.class)
  public static final class RunningModeTest extends ImageEmbedderTest {

    @Test
    public void create_failsWithIllegalResultListenerInNonLiveStreamMode() throws Exception {
      for (RunningMode mode : new RunningMode[] {RunningMode.IMAGE, RunningMode.VIDEO}) {
        IllegalArgumentException exception =
            assertThrows(
                IllegalArgumentException.class,
                () ->
                    ImageEmbedderOptions.builder()
                        .setBaseOptions(
                            BaseOptions.builder().setModelAssetPath(MOBILENET_EMBEDDER).build())
                        .setRunningMode(mode)
                        .setResultListener((result, inputImage) -> {})
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
                  ImageEmbedderOptions.builder()
                      .setBaseOptions(
                          BaseOptions.builder().setModelAssetPath(MOBILENET_EMBEDDER).build())
                      .setRunningMode(RunningMode.LIVE_STREAM)
                      .build());
      assertThat(exception)
          .hasMessageThat()
          .contains("a user-defined result listener must be provided");
    }

    @Test
    public void embed_failsWithCallingWrongApiInImageMode() throws Exception {
      ImageEmbedderOptions options =
          ImageEmbedderOptions.builder()
              .setBaseOptions(BaseOptions.builder().setModelAssetPath(MOBILENET_EMBEDDER).build())
              .setRunningMode(RunningMode.IMAGE)
              .build();

      ImageEmbedder imageEmbedder =
          ImageEmbedder.createFromOptions(ApplicationProvider.getApplicationContext(), options);
      MediaPipeException exception =
          assertThrows(
              MediaPipeException.class,
              () ->
                  imageEmbedder.embedForVideo(
                      getImageFromAsset(BURGER_IMAGE), /* timestampMs= */ 0));
      assertThat(exception).hasMessageThat().contains("not initialized with the video mode");
      exception =
          assertThrows(
              MediaPipeException.class,
              () ->
                  imageEmbedder.embedAsync(getImageFromAsset(BURGER_IMAGE), /* timestampMs= */ 0));
      assertThat(exception).hasMessageThat().contains("not initialized with the live stream mode");
    }

    @Test
    public void embed_failsWithCallingWrongApiInVideoMode() throws Exception {
      ImageEmbedderOptions options =
          ImageEmbedderOptions.builder()
              .setBaseOptions(BaseOptions.builder().setModelAssetPath(MOBILENET_EMBEDDER).build())
              .setRunningMode(RunningMode.VIDEO)
              .build();

      ImageEmbedder imageEmbedder =
          ImageEmbedder.createFromOptions(ApplicationProvider.getApplicationContext(), options);
      MediaPipeException exception =
          assertThrows(
              MediaPipeException.class, () -> imageEmbedder.embed(getImageFromAsset(BURGER_IMAGE)));
      assertThat(exception).hasMessageThat().contains("not initialized with the image mode");
      exception =
          assertThrows(
              MediaPipeException.class,
              () ->
                  imageEmbedder.embedAsync(getImageFromAsset(BURGER_IMAGE), /* timestampMs= */ 0));
      assertThat(exception).hasMessageThat().contains("not initialized with the live stream mode");
    }

    @Test
    public void embed_failsWithCallingWrongApiInLiveSteamMode() throws Exception {
      ImageEmbedderOptions options =
          ImageEmbedderOptions.builder()
              .setBaseOptions(BaseOptions.builder().setModelAssetPath(MOBILENET_EMBEDDER).build())
              .setRunningMode(RunningMode.LIVE_STREAM)
              .setResultListener((imageClassificationResult, inputImage) -> {})
              .build();

      ImageEmbedder imageEmbedder =
          ImageEmbedder.createFromOptions(ApplicationProvider.getApplicationContext(), options);

      MediaPipeException exception =
          assertThrows(
              MediaPipeException.class, () -> imageEmbedder.embed(getImageFromAsset(BURGER_IMAGE)));
      assertThat(exception).hasMessageThat().contains("not initialized with the image mode");
      exception =
          assertThrows(
              MediaPipeException.class,
              () ->
                  imageEmbedder.embedForVideo(
                      getImageFromAsset(BURGER_IMAGE), /* timestampMs= */ 0));
      assertThat(exception).hasMessageThat().contains("not initialized with the video mode");
    }

    @Test
    public void embed_succeedsWithImageMode() throws Exception {
      ImageEmbedder imageEmbedder =
          ImageEmbedder.createFromFile(
              ApplicationProvider.getApplicationContext(), MOBILENET_EMBEDDER);
      ImageEmbedderResult result = imageEmbedder.embed(getImageFromAsset(BURGER_IMAGE));
      ImageEmbedderResult resultCrop = imageEmbedder.embed(getImageFromAsset(BURGER_CROP_IMAGE));

      // Check results.
      assertHasOneHeadAndCorrectDimension(result, /* quantized= */ false);
      assertHasOneHeadAndCorrectDimension(resultCrop, /* quantized= */ false);
      // Check similarity.
      double similarity =
          ImageEmbedder.cosineSimilarity(
              result.embeddingResult().embeddings().get(0),
              resultCrop.embeddingResult().embeddings().get(0));
      assertThat(similarity).isWithin(DOUBLE_DIFF_TOLERANCE).of(0.925272);
    }

    @Test
    public void embed_succeedsWithVideoMode() throws Exception {
      BaseOptions baseOptions = BaseOptions.builder().setModelAssetPath(MOBILENET_EMBEDDER).build();
      ImageEmbedderOptions options =
          ImageEmbedderOptions.builder()
              .setBaseOptions(baseOptions)
              .setRunningMode(RunningMode.VIDEO)
              .build();
      ImageEmbedder imageEmbedder =
          ImageEmbedder.createFromOptions(ApplicationProvider.getApplicationContext(), options);

      for (int i = 0; i < 3; ++i) {
        ImageEmbedderResult result =
            imageEmbedder.embedForVideo(getImageFromAsset(BURGER_IMAGE), /* timestampMs= */ i);
        assertHasOneHeadAndCorrectDimension(result, /* quantized= */ false);
      }
    }

    @Test
    public void embed_failsWithOutOfOrderInputTimestamps() throws Exception {
      MPImage image = getImageFromAsset(BURGER_IMAGE);
      BaseOptions baseOptions = BaseOptions.builder().setModelAssetPath(MOBILENET_EMBEDDER).build();
      ImageEmbedderOptions options =
          ImageEmbedderOptions.builder()
              .setBaseOptions(baseOptions)
              .setRunningMode(RunningMode.LIVE_STREAM)
              .setResultListener(
                  (imageEmbedderResult, inputImage) -> {
                    assertHasOneHeadAndCorrectDimension(
                        imageEmbedderResult, /* quantized= */ false);
                    assertImageSizeIsExpected(inputImage);
                  })
              .build();
      try (ImageEmbedder imageEmbedder =
          ImageEmbedder.createFromOptions(ApplicationProvider.getApplicationContext(), options)) {
        imageEmbedder.embedAsync(getImageFromAsset(BURGER_IMAGE), /* timestampMs= */ 1);
        MediaPipeException exception =
            assertThrows(
                MediaPipeException.class,
                () -> imageEmbedder.embedAsync(image, /* timestampMs= */ 0));
        assertThat(exception)
            .hasMessageThat()
            .contains("having a smaller timestamp than the processed timestamp");
      }
    }

    @Test
    public void embed_succeedsWithLiveStreamMode() throws Exception {
      MPImage image = getImageFromAsset(BURGER_IMAGE);
      BaseOptions baseOptions = BaseOptions.builder().setModelAssetPath(MOBILENET_EMBEDDER).build();
      ImageEmbedderOptions options =
          ImageEmbedderOptions.builder()
              .setBaseOptions(baseOptions)
              .setRunningMode(RunningMode.LIVE_STREAM)
              .setResultListener(
                  (imageEmbedderResult, inputImage) -> {
                    assertHasOneHeadAndCorrectDimension(
                        imageEmbedderResult, /* quantized= */ false);
                    assertImageSizeIsExpected(inputImage);
                  })
              .build();
      try (ImageEmbedder imageEmbedder =
          ImageEmbedder.createFromOptions(ApplicationProvider.getApplicationContext(), options)) {
        for (int i = 0; i < 3; ++i) {
          imageEmbedder.embedAsync(image, /* timestampMs= */ i);
        }
      }
    }
  }

  private static MPImage getImageFromAsset(String filePath) throws Exception {
    AssetManager assetManager = ApplicationProvider.getApplicationContext().getAssets();
    InputStream istr = assetManager.open(filePath);
    return new BitmapImageBuilder(BitmapFactory.decodeStream(istr)).build();
  }

  private static void assertHasOneHeadAndCorrectDimension(
      ImageEmbedderResult result, boolean quantized) {
    assertThat(result.embeddingResult().embeddings()).hasSize(1);
    assertThat(result.embeddingResult().embeddings().get(0).headIndex()).isEqualTo(0);
    assertThat(result.embeddingResult().embeddings().get(0).headName().get()).isEqualTo("feature");
    if (quantized) {
      assertThat(result.embeddingResult().embeddings().get(0).quantizedEmbedding()).hasLength(1024);
    } else {
      assertThat(result.embeddingResult().embeddings().get(0).floatEmbedding()).hasLength(1024);
    }
  }

  private static void assertImageSizeIsExpected(MPImage inputImage) {
    assertThat(inputImage).isNotNull();
    assertThat(inputImage.getWidth()).isEqualTo(480);
    assertThat(inputImage.getHeight()).isEqualTo(325);
  }
}
