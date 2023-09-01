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

package com.google.mediapipe.tasks.vision.imagesegmenter;

import static com.google.common.truth.Truth.assertThat;
import static org.junit.Assert.assertThrows;

import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Color;
import androidx.test.core.app.ApplicationProvider;
import androidx.test.ext.junit.runners.AndroidJUnit4;
import com.google.mediapipe.framework.MediaPipeException;
import com.google.mediapipe.framework.image.BitmapExtractor;
import com.google.mediapipe.framework.image.BitmapImageBuilder;
import com.google.mediapipe.framework.image.ByteBufferExtractor;
import com.google.mediapipe.framework.image.MPImage;
import com.google.mediapipe.tasks.core.BaseOptions;
import com.google.mediapipe.tasks.vision.core.RunningMode;
import com.google.mediapipe.tasks.vision.imagesegmenter.ImageSegmenter.ImageSegmenterOptions;
import com.google.mediapipe.tasks.vision.imagesegmenter.ImageSegmenter.SegmentationOptions;
import java.io.InputStream;
import java.nio.ByteBuffer;
import java.nio.FloatBuffer;
import java.util.Arrays;
import java.util.List;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Suite;
import org.junit.runners.Suite.SuiteClasses;

/** Test for {@link ImageSegmenter}. */
@RunWith(Suite.class)
@SuiteClasses({ImageSegmenterTest.General.class, ImageSegmenterTest.RunningModeTest.class})
public class ImageSegmenterTest {
  private static final String DEEPLAB_MODEL_FILE = "deeplabv3.tflite";
  private static final String SELFIE_128x128_MODEL_FILE = "selfie_segm_128_128_3.tflite";
  private static final String SELFIE_144x256_MODEL_FILE = "selfie_segm_144_256_3.tflite";
  private static final String CAT_IMAGE = "cat.jpg";
  private static final float GOLDEN_MASK_SIMILARITY = 0.96f;
  private static final int MAGNIFICATION_FACTOR = 10;

  @RunWith(AndroidJUnit4.class)
  public static final class General extends ImageSegmenterTest {
    @Test
    public void segment_successWithCategoryMask() throws Exception {
      final String inputImageName = "segmentation_input_rotation0.jpg";
      final String goldenImageName = "segmentation_golden_rotation0.png";
      ImageSegmenterOptions options =
          ImageSegmenterOptions.builder()
              .setBaseOptions(BaseOptions.builder().setModelAssetPath(DEEPLAB_MODEL_FILE).build())
              .setOutputConfidenceMasks(false)
              .setOutputCategoryMask(true)
              .build();
      ImageSegmenter imageSegmenter =
          ImageSegmenter.createFromOptions(ApplicationProvider.getApplicationContext(), options);
      ImageSegmenterResult actualResult = imageSegmenter.segment(getImageFromAsset(inputImageName));
      assertThat(actualResult.categoryMask().isPresent()).isTrue();
      MPImage actualMaskBuffer = actualResult.categoryMask().get();
      MPImage expectedMaskBuffer = getImageFromAsset(goldenImageName);
      verifyCategoryMask(
          actualMaskBuffer, expectedMaskBuffer, GOLDEN_MASK_SIMILARITY, MAGNIFICATION_FACTOR);
    }

    @Test
    public void segment_successWithConfidenceMask() throws Exception {
      final String inputImageName = "cat.jpg";
      final String goldenImageName = "cat_mask.jpg";
      ImageSegmenterOptions options =
          ImageSegmenterOptions.builder()
              .setBaseOptions(BaseOptions.builder().setModelAssetPath(DEEPLAB_MODEL_FILE).build())
              .build();
      ImageSegmenter imageSegmenter =
          ImageSegmenter.createFromOptions(ApplicationProvider.getApplicationContext(), options);
      ImageSegmenterResult actualResult = imageSegmenter.segment(getImageFromAsset(inputImageName));
      List<MPImage> segmentations = actualResult.confidenceMasks().get();
      assertThat(segmentations.size()).isEqualTo(21);
      // Cat category index 8.
      MPImage actualMaskBuffer = segmentations.get(8);
      MPImage expectedMaskBuffer = getImageFromAsset(goldenImageName);
      verifyConfidenceMask(actualMaskBuffer, expectedMaskBuffer, GOLDEN_MASK_SIMILARITY);
    }

    @Test
    public void segment_successWith128x128Segmentation() throws Exception {
      final String inputImageName = "mozart_square.jpg";
      final String goldenImageName = "selfie_segm_128_128_3_expected_mask.jpg";
      ImageSegmenterOptions options =
          ImageSegmenterOptions.builder()
              .setBaseOptions(
                  BaseOptions.builder().setModelAssetPath(SELFIE_128x128_MODEL_FILE).build())
              .build();
      ImageSegmenter imageSegmenter =
          ImageSegmenter.createFromOptions(ApplicationProvider.getApplicationContext(), options);
      ImageSegmenterResult actualResult = imageSegmenter.segment(getImageFromAsset(inputImageName));
      List<MPImage> segmentations = actualResult.confidenceMasks().get();
      assertThat(segmentations.size()).isEqualTo(2);
      // Selfie category index 1.
      MPImage actualMaskBuffer = segmentations.get(1);
      MPImage expectedMaskBuffer = getImageFromAsset(goldenImageName);
      verifyConfidenceMask(actualMaskBuffer, expectedMaskBuffer, GOLDEN_MASK_SIMILARITY);
    }

    @Test
    public void segment_successWith144x256Segmentation() throws Exception {
      final String inputImageName = "mozart_square.jpg";
      final String goldenImageName = "selfie_segm_144_256_3_expected_mask.jpg";
      ImageSegmenterOptions options =
          ImageSegmenterOptions.builder()
              .setBaseOptions(
                  BaseOptions.builder().setModelAssetPath(SELFIE_144x256_MODEL_FILE).build())
              .build();
      ImageSegmenter imageSegmenter =
          ImageSegmenter.createFromOptions(ApplicationProvider.getApplicationContext(), options);
      ImageSegmenterResult actualResult = imageSegmenter.segment(getImageFromAsset(inputImageName));
      List<MPImage> segmentations = actualResult.confidenceMasks().get();
      assertThat(segmentations.size()).isEqualTo(1);
      MPImage actualMaskBuffer = segmentations.get(0);
      MPImage expectedMaskBuffer = getImageFromAsset(goldenImageName);
      verifyConfidenceMask(actualMaskBuffer, expectedMaskBuffer, GOLDEN_MASK_SIMILARITY);
    }

    @Test
    public void getLabels_success() throws Exception {
      final List<String> expectedLabels =
          Arrays.asList(
              "background",
              "aeroplane",
              "bicycle",
              "bird",
              "boat",
              "bottle",
              "bus",
              "car",
              "cat",
              "chair",
              "cow",
              "dining table",
              "dog",
              "horse",
              "motorbike",
              "person",
              "potted plant",
              "sheep",
              "sofa",
              "train",
              "tv");
      ImageSegmenterOptions options =
          ImageSegmenterOptions.builder()
              .setBaseOptions(BaseOptions.builder().setModelAssetPath(DEEPLAB_MODEL_FILE).build())
              .build();
      ImageSegmenter imageSegmenter =
          ImageSegmenter.createFromOptions(ApplicationProvider.getApplicationContext(), options);
      List<String> actualLabels = imageSegmenter.getLabels();
      assertThat(actualLabels.size()).isEqualTo(expectedLabels.size());
      for (int i = 0; i < actualLabels.size(); i++) {
        assertThat(actualLabels.get(i)).isEqualTo(expectedLabels.get(i));
      }
    }
  }

  @RunWith(AndroidJUnit4.class)
  public static final class RunningModeTest extends ImageSegmenterTest {
    @Test
    public void create_failsWithMissingResultListenerInLiveSteamMode() throws Exception {
      IllegalArgumentException exception =
          assertThrows(
              IllegalArgumentException.class,
              () ->
                  ImageSegmenterOptions.builder()
                      .setBaseOptions(
                          BaseOptions.builder().setModelAssetPath(DEEPLAB_MODEL_FILE).build())
                      .setRunningMode(RunningMode.LIVE_STREAM)
                      .build());
      assertThat(exception)
          .hasMessageThat()
          .contains("a user-defined result listener must be provided");
    }

    @Test
    public void segment_failsWithCallingWrongApiInImageMode() throws Exception {
      ImageSegmenterOptions options =
          ImageSegmenterOptions.builder()
              .setBaseOptions(BaseOptions.builder().setModelAssetPath(DEEPLAB_MODEL_FILE).build())
              .setRunningMode(RunningMode.IMAGE)
              .build();
      ImageSegmenter imageSegmenter =
          ImageSegmenter.createFromOptions(ApplicationProvider.getApplicationContext(), options);
      MediaPipeException exception =
          assertThrows(
              MediaPipeException.class,
              () ->
                  imageSegmenter.segmentForVideo(
                      getImageFromAsset(CAT_IMAGE), /* timestampsMs= */ 0));
      assertThat(exception).hasMessageThat().contains("not initialized with the video mode");
      exception =
          assertThrows(
              MediaPipeException.class,
              () ->
                  imageSegmenter.segmentAsync(getImageFromAsset(CAT_IMAGE), /* timestampsMs= */ 0));
      assertThat(exception).hasMessageThat().contains("not initialized with the live stream mode");
      exception =
          assertThrows(
              MediaPipeException.class,
              () -> imageSegmenter.segmentWithResultListener(getImageFromAsset(CAT_IMAGE)));
      assertThat(exception)
          .hasMessageThat()
          .contains("ResultListener is not set in the ImageSegmenterOptions");
    }

    @Test
    public void segment_failsWithCallingWrongApiInVideoMode() throws Exception {
      ImageSegmenterOptions options =
          ImageSegmenterOptions.builder()
              .setBaseOptions(BaseOptions.builder().setModelAssetPath(DEEPLAB_MODEL_FILE).build())
              .setRunningMode(RunningMode.VIDEO)
              .build();
      ImageSegmenter imageSegmenter =
          ImageSegmenter.createFromOptions(ApplicationProvider.getApplicationContext(), options);
      MediaPipeException exception =
          assertThrows(
              MediaPipeException.class, () -> imageSegmenter.segment(getImageFromAsset(CAT_IMAGE)));
      assertThat(exception).hasMessageThat().contains("not initialized with the image mode");
      exception =
          assertThrows(
              MediaPipeException.class,
              () ->
                  imageSegmenter.segmentAsync(getImageFromAsset(CAT_IMAGE), /* timestampsMs= */ 0));
      assertThat(exception).hasMessageThat().contains("not initialized with the live stream mode");
      exception =
          assertThrows(
              MediaPipeException.class,
              () ->
                  imageSegmenter.segmentForVideoWithResultListener(
                      getImageFromAsset(CAT_IMAGE), /* timestampsMs= */ 0));
      assertThat(exception)
          .hasMessageThat()
          .contains("ResultListener is not set in the ImageSegmenterOptions");
    }

    @Test
    public void segment_failsWithCallingWrongApiInLiveSteamMode() throws Exception {
      ImageSegmenterOptions options =
          ImageSegmenterOptions.builder()
              .setBaseOptions(BaseOptions.builder().setModelAssetPath(DEEPLAB_MODEL_FILE).build())
              .setRunningMode(RunningMode.LIVE_STREAM)
              .setResultListener((result, inputImage) -> {})
              .build();
      ImageSegmenter imageSegmenter =
          ImageSegmenter.createFromOptions(ApplicationProvider.getApplicationContext(), options);
      MediaPipeException exception =
          assertThrows(
              MediaPipeException.class,
              () -> imageSegmenter.segmentWithResultListener(getImageFromAsset(CAT_IMAGE)));
      assertThat(exception).hasMessageThat().contains("not initialized with the image mode");
      exception =
          assertThrows(
              MediaPipeException.class,
              () ->
                  imageSegmenter.segmentForVideoWithResultListener(
                      getImageFromAsset(CAT_IMAGE), /* timestampsMs= */ 0));
      assertThat(exception).hasMessageThat().contains("not initialized with the video mode");
    }

    @Test
    public void segment_successWithImageMode() throws Exception {
      final String inputImageName = "cat.jpg";
      final String goldenImageName = "cat_mask.jpg";
      ImageSegmenterOptions options =
          ImageSegmenterOptions.builder()
              .setBaseOptions(BaseOptions.builder().setModelAssetPath(DEEPLAB_MODEL_FILE).build())
              .setRunningMode(RunningMode.IMAGE)
              .build();
      ImageSegmenter imageSegmenter =
          ImageSegmenter.createFromOptions(ApplicationProvider.getApplicationContext(), options);
      ImageSegmenterResult actualResult = imageSegmenter.segment(getImageFromAsset(inputImageName));
      List<MPImage> segmentations = actualResult.confidenceMasks().get();
      assertThat(segmentations.size()).isEqualTo(21);
      // Cat category index 8.
      MPImage actualMaskBuffer = segmentations.get(8);
      MPImage expectedMaskBuffer = getImageFromAsset(goldenImageName);
      verifyConfidenceMask(actualMaskBuffer, expectedMaskBuffer, GOLDEN_MASK_SIMILARITY);
    }

    @Test
    public void segment_successWithImageModeWithResultListener() throws Exception {
      final String inputImageName = "cat.jpg";
      final String goldenImageName = "cat_mask.jpg";
      MPImage expectedResult = getImageFromAsset(goldenImageName);
      ImageSegmenterOptions options =
          ImageSegmenterOptions.builder()
              .setBaseOptions(BaseOptions.builder().setModelAssetPath(DEEPLAB_MODEL_FILE).build())
              .setRunningMode(RunningMode.IMAGE)
              .setResultListener(
                  (segmenterResult, inputImage) -> {
                    verifyConfidenceMask(
                        segmenterResult.confidenceMasks().get().get(8),
                        expectedResult,
                        GOLDEN_MASK_SIMILARITY);
                  })
              .build();
      ImageSegmenter imageSegmenter =
          ImageSegmenter.createFromOptions(ApplicationProvider.getApplicationContext(), options);
      imageSegmenter.segmentWithResultListener(getImageFromAsset(inputImageName));
    }

    @Test
    public void segment_successWithVideoMode() throws Exception {
      final String inputImageName = "cat.jpg";
      final String goldenImageName = "cat_mask.jpg";
      ImageSegmenterOptions options =
          ImageSegmenterOptions.builder()
              .setBaseOptions(BaseOptions.builder().setModelAssetPath(DEEPLAB_MODEL_FILE).build())
              .setRunningMode(RunningMode.VIDEO)
              .build();
      ImageSegmenter imageSegmenter =
          ImageSegmenter.createFromOptions(ApplicationProvider.getApplicationContext(), options);
      MPImage expectedMaskBuffer = getImageFromAsset(goldenImageName);
      for (int i = 0; i < 3; i++) {
        ImageSegmenterResult actualResult =
            imageSegmenter.segmentForVideo(
                getImageFromAsset(inputImageName), /* timestampsMs= */ i);
        List<MPImage> segmentations = actualResult.confidenceMasks().get();
        assertThat(segmentations.size()).isEqualTo(21);
        // Cat category index 8.
        MPImage actualMaskBuffer = segmentations.get(8);
        verifyConfidenceMask(actualMaskBuffer, expectedMaskBuffer, GOLDEN_MASK_SIMILARITY);
      }
    }

    @Test
    public void segment_successWithVideoModeWithResultListener() throws Exception {
      final String inputImageName = "cat.jpg";
      final String goldenImageName = "cat_mask.jpg";
      MPImage expectedResult = getImageFromAsset(goldenImageName);
      ImageSegmenterOptions options =
          ImageSegmenterOptions.builder()
              .setBaseOptions(BaseOptions.builder().setModelAssetPath(DEEPLAB_MODEL_FILE).build())
              .setRunningMode(RunningMode.VIDEO)
              .setResultListener(
                  (segmenterResult, inputImage) -> {
                    verifyConfidenceMask(
                        segmenterResult.confidenceMasks().get().get(8),
                        expectedResult,
                        GOLDEN_MASK_SIMILARITY);
                  })
              .build();
      ImageSegmenter imageSegmenter =
          ImageSegmenter.createFromOptions(ApplicationProvider.getApplicationContext(), options);
      for (int i = 0; i < 3; i++) {
        imageSegmenter.segmentForVideoWithResultListener(
            getImageFromAsset(inputImageName), /* timestampsMs= */ i);
      }
    }

    @Test
    public void segment_successWithLiveStreamMode() throws Exception {
      final String inputImageName = "cat.jpg";
      final String goldenImageName = "cat_mask.jpg";
      MPImage image = getImageFromAsset(inputImageName);
      MPImage expectedResult = getImageFromAsset(goldenImageName);
      ImageSegmenterOptions options =
          ImageSegmenterOptions.builder()
              .setBaseOptions(BaseOptions.builder().setModelAssetPath(DEEPLAB_MODEL_FILE).build())
              .setRunningMode(RunningMode.LIVE_STREAM)
              .setResultListener(
                  (segmenterResult, inputImage) -> {
                    verifyConfidenceMask(
                        segmenterResult.confidenceMasks().get().get(8),
                        expectedResult,
                        GOLDEN_MASK_SIMILARITY);
                  })
              .build();
      try (ImageSegmenter imageSegmenter =
          ImageSegmenter.createFromOptions(ApplicationProvider.getApplicationContext(), options)) {
        for (int i = 0; i < 3; i++) {
          imageSegmenter.segmentAsync(image, /* timestampsMs= */ i);
        }
      }
    }

    @Test
    public void segment_failsWithOutOfOrderInputTimestamps() throws Exception {
      final String inputImageName = "cat.jpg";
      final String goldenImageName = "cat_mask.jpg";
      MPImage image = getImageFromAsset(inputImageName);
      MPImage expectedResult = getImageFromAsset(goldenImageName);
      ImageSegmenterOptions options =
          ImageSegmenterOptions.builder()
              .setBaseOptions(BaseOptions.builder().setModelAssetPath(DEEPLAB_MODEL_FILE).build())
              .setRunningMode(RunningMode.LIVE_STREAM)
              .setResultListener(
                  (segmenterResult, inputImage) -> {
                    verifyConfidenceMask(
                        segmenterResult.confidenceMasks().get().get(8),
                        expectedResult,
                        GOLDEN_MASK_SIMILARITY);
                  })
              .build();
      try (ImageSegmenter imageSegmenter =
          ImageSegmenter.createFromOptions(ApplicationProvider.getApplicationContext(), options)) {
        imageSegmenter.segmentAsync(image, /* timestampsMs= */ 1);
        MediaPipeException exception =
            assertThrows(
                MediaPipeException.class,
                () -> imageSegmenter.segmentAsync(image, /* timestampsMs= */ 0));
        assertThat(exception)
            .hasMessageThat()
            .contains("having a smaller timestamp than the processed timestamp");
      }
    }
  }

  private static void verifyCategoryMask(
      MPImage actualMask, MPImage goldenMask, float similarityThreshold, int magnificationFactor) {
    assertThat(actualMask.getWidth()).isEqualTo(goldenMask.getWidth());
    assertThat(actualMask.getHeight()).isEqualTo(goldenMask.getHeight());
    ByteBuffer actualMaskBuffer = ByteBufferExtractor.extract(actualMask);
    Bitmap goldenMaskBitmap = BitmapExtractor.extract(goldenMask);
    int consistentPixels = 0;
    final int numPixels = actualMask.getWidth() * actualMask.getHeight();
    actualMaskBuffer.rewind();
    for (int y = 0; y < actualMask.getHeight(); y++) {
      for (int x = 0; x < actualMask.getWidth(); x++) {
        // RGB values are the same in the golden mask image.
        consistentPixels +=
            actualMaskBuffer.get() * magnificationFactor
                    == Color.red(goldenMaskBitmap.getPixel(x, y))
                ? 1
                : 0;
      }
    }
    assertThat((float) consistentPixels / numPixels).isGreaterThan(similarityThreshold);
  }

  private static void verifyConfidenceMask(
      MPImage actualMask, MPImage goldenMask, float similarityThreshold) {
    assertThat(actualMask.getWidth()).isEqualTo(goldenMask.getWidth());
    assertThat(actualMask.getHeight()).isEqualTo(goldenMask.getHeight());
    FloatBuffer actualMaskBuffer = ByteBufferExtractor.extract(actualMask).asFloatBuffer();
    Bitmap goldenMaskBitmap = BitmapExtractor.extract(goldenMask);
    FloatBuffer goldenMaskBuffer = getByteBufferFromBitmap(goldenMaskBitmap).asFloatBuffer();
    assertThat(
            calculateSoftIOU(
                actualMaskBuffer, goldenMaskBuffer, actualMask.getWidth() * actualMask.getHeight()))
        .isGreaterThan((double) similarityThreshold);
  }

  private static MPImage getImageFromAsset(String filePath) throws Exception {
    AssetManager assetManager = ApplicationProvider.getApplicationContext().getAssets();
    InputStream istr = assetManager.open(filePath);
    return new BitmapImageBuilder(BitmapFactory.decodeStream(istr)).build();
  }

  private static ByteBuffer getByteBufferFromBitmap(Bitmap bitmap) {
    ByteBuffer byteBuffer = ByteBuffer.allocateDirect(bitmap.getWidth() * bitmap.getHeight() * 4);
    for (int y = 0; y < bitmap.getHeight(); y++) {
      for (int x = 0; x < bitmap.getWidth(); x++) {
        byteBuffer.putFloat((float) Color.red(bitmap.getPixel(x, y)) / 255.f);
      }
    }
    byteBuffer.rewind();
    return byteBuffer;
  }

  private static double calculateSum(FloatBuffer m) {
    m.rewind();
    double sum = 0;
    while (m.hasRemaining()) {
      sum += m.get();
    }
    m.rewind();
    return sum;
  }

  private static FloatBuffer multiply(FloatBuffer m1, FloatBuffer m2, int bufferSize) {
    m1.rewind();
    m2.rewind();
    FloatBuffer buffer = FloatBuffer.allocate(bufferSize);
    while (m1.hasRemaining()) {
      buffer.put(m1.get() * m2.get());
    }
    m1.rewind();
    m2.rewind();
    buffer.rewind();
    return buffer;
  }

  private static double calculateSoftIOU(FloatBuffer m1, FloatBuffer m2, int bufferSize) {
    double intersectionSum = calculateSum(multiply(m1, m2, bufferSize));
    double m1m1 = calculateSum(multiply(m1, m1.duplicate(), bufferSize));
    double m2m2 = calculateSum(multiply(m2, m2.duplicate(), bufferSize));
    double unionSum = m1m1 + m2m2 - intersectionSum;
    return unionSum > 0.0 ? intersectionSum / unionSum : 0.0;
  }
}
