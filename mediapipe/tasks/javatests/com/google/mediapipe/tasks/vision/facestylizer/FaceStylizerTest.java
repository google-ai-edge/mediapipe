// Copyright 2023 The MediaPipe Authors. All Rights Reserved.
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

package com.google.mediapipe.tasks.vision.facestylizer;

import static com.google.common.truth.Truth.assertThat;
import static org.junit.Assert.assertThrows;

import android.content.res.AssetManager;
import android.graphics.BitmapFactory;
import android.graphics.RectF;
import android.util.Pair;
import androidx.test.core.app.ApplicationProvider;
import androidx.test.ext.junit.runners.AndroidJUnit4;
import com.google.mediapipe.framework.MediaPipeException;
import com.google.mediapipe.framework.image.BitmapImageBuilder;
import com.google.mediapipe.framework.image.MPImage;
import com.google.mediapipe.tasks.core.BaseOptions;
import com.google.mediapipe.tasks.vision.core.ImageProcessingOptions;
import com.google.mediapipe.tasks.vision.core.RunningMode;
import com.google.mediapipe.tasks.vision.facestylizer.FaceStylizer.FaceStylizerOptions;
import java.io.InputStream;
import org.junit.After;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Suite;
import org.junit.runners.Suite.SuiteClasses;

/** Test for {@link FaceStylizer}. */
@RunWith(Suite.class)
@SuiteClasses({FaceStylizerTest.General.class, FaceStylizerTest.RunningModeTest.class})
public class FaceStylizerTest {
  private static final String modelFile = "face_stylization_dummy.tflite";
  private static final String testImage = "portrait.jpg";
  private static final int modelImageSize = 512;

  public Pair<Integer, Integer> getRectPixelSize(MPImage originalImage, RectF rect) {
    int width = originalImage.getWidth();
    int height = originalImage.getHeight();
    return new Pair<>(
        (int) ((rect.right - rect.left) * width), (int) ((rect.bottom - rect.top) * height));
  }

  @RunWith(AndroidJUnit4.class)
  public static final class General extends FaceStylizerTest {
    FaceStylizer faceStylizer;

    @After
    public void afterEach() throws Exception {
      if (faceStylizer != null) {
        faceStylizer.close();
      }
    }

    @Test
    public void create_succeeds() throws Exception {
      FaceStylizerOptions options =
          FaceStylizerOptions.builder()
              .setBaseOptions(BaseOptions.builder().setModelAssetPath(modelFile).build())
              .build();
      faceStylizer =
          FaceStylizer.createFromOptions(ApplicationProvider.getApplicationContext(), options);
      assertThat(faceStylizer).isNotNull();
    }

    @Test
    public void create_failsWithMissingModel() throws Exception {
      IllegalArgumentException exception =
          assertThrows(
              IllegalArgumentException.class,
              () ->
                  FaceStylizerOptions.builder()
                      .setBaseOptions(BaseOptions.builder().build())
                      .build());
      assertThat(exception)
          .hasMessageThat()
          .contains(
              "specify only one of the model asset path, the model asset file descriptor, and the"
                  + " model asset buffer");
    }
  }

  @RunWith(AndroidJUnit4.class)
  public static final class RunningModeTest extends FaceStylizerTest {
    FaceStylizer faceStylizer;

    @After
    public void afterEach() throws Exception {
      if (faceStylizer != null) {
        faceStylizer.close();
      }
    }

    @Test
    public void create_failsWithMissingResultListenerInLiveSteamMode() throws Exception {
      IllegalArgumentException exception =
          assertThrows(
              IllegalArgumentException.class,
              () ->
                  FaceStylizerOptions.builder()
                      .setBaseOptions(BaseOptions.builder().setModelAssetPath(modelFile).build())
                      .setRunningMode(RunningMode.LIVE_STREAM)
                      .build());
      assertThat(exception)
          .hasMessageThat()
          .contains("a user-defined result listener must be provided");
    }

    @Test
    public void stylizer_failsWithCallingWrongApiInImageMode() throws Exception {
      FaceStylizerOptions options =
          FaceStylizerOptions.builder()
              .setBaseOptions(BaseOptions.builder().setModelAssetPath(modelFile).build())
              .setRunningMode(RunningMode.IMAGE)
              .build();
      faceStylizer =
          FaceStylizer.createFromOptions(ApplicationProvider.getApplicationContext(), options);
      MediaPipeException exception =
          assertThrows(
              MediaPipeException.class,
              () ->
                  faceStylizer.stylizeForVideo(
                      getImageFromAsset(testImage), /* timestampsMs= */ 0));
      assertThat(exception).hasMessageThat().contains("not initialized with the video mode");
      exception =
          assertThrows(
              MediaPipeException.class,
              () -> faceStylizer.stylizeAsync(getImageFromAsset(testImage), /* timestampsMs= */ 0));
      assertThat(exception).hasMessageThat().contains("not initialized with the live stream mode");
      exception =
          assertThrows(
              MediaPipeException.class,
              () -> faceStylizer.stylizeWithResultListener(getImageFromAsset(testImage)));
      assertThat(exception)
          .hasMessageThat()
          .contains("ResultListener is not set in the FaceStylizerOptions");
    }

    @Test
    public void stylizer_failsWithCallingWrongApiInVideoMode() throws Exception {
      FaceStylizerOptions options =
          FaceStylizerOptions.builder()
              .setBaseOptions(BaseOptions.builder().setModelAssetPath(modelFile).build())
              .setRunningMode(RunningMode.VIDEO)
              .build();

      faceStylizer =
          FaceStylizer.createFromOptions(ApplicationProvider.getApplicationContext(), options);
      MediaPipeException exception =
          assertThrows(
              MediaPipeException.class, () -> faceStylizer.stylize(getImageFromAsset(testImage)));
      assertThat(exception).hasMessageThat().contains("not initialized with the image mode");
      exception =
          assertThrows(
              MediaPipeException.class,
              () -> faceStylizer.stylizeAsync(getImageFromAsset(testImage), /* timestampsMs= */ 0));
      assertThat(exception).hasMessageThat().contains("not initialized with the live stream mode");
      exception =
          assertThrows(
              MediaPipeException.class,
              () ->
                  faceStylizer.stylizeForVideoWithResultListener(
                      getImageFromAsset(testImage), /* timestampsMs= */ 0));
      assertThat(exception)
          .hasMessageThat()
          .contains("ResultListener is not set in the FaceStylizerOptions");
    }

    @Test
    public void stylizer_failsWithCallingWrongApiInLiveSteamMode() throws Exception {
      FaceStylizerOptions options =
          FaceStylizerOptions.builder()
              .setBaseOptions(BaseOptions.builder().setModelAssetPath(modelFile).build())
              .setRunningMode(RunningMode.LIVE_STREAM)
              .setResultListener((result, inputImage) -> {})
              .build();

      faceStylizer =
          FaceStylizer.createFromOptions(ApplicationProvider.getApplicationContext(), options);
      MediaPipeException exception =
          assertThrows(
              MediaPipeException.class,
              () -> faceStylizer.stylizeWithResultListener(getImageFromAsset(testImage)));
      assertThat(exception).hasMessageThat().contains("not initialized with the image mode");
      exception =
          assertThrows(
              MediaPipeException.class,
              () ->
                  faceStylizer.stylizeForVideoWithResultListener(
                      getImageFromAsset(testImage), /* timestampsMs= */ 0));
      assertThat(exception).hasMessageThat().contains("not initialized with the video mode");
    }

    @Test
    public void stylizer_succeedsWithImageMode() throws Exception {
      FaceStylizerOptions options =
          FaceStylizerOptions.builder()
              .setBaseOptions(BaseOptions.builder().setModelAssetPath(modelFile).build())
              .setRunningMode(RunningMode.IMAGE)
              .build();

      faceStylizer =
          FaceStylizer.createFromOptions(ApplicationProvider.getApplicationContext(), options);

      MPImage inputImage = getImageFromAsset(testImage);
      int inputWidth = inputImage.getWidth();
      int inputHeight = inputImage.getHeight();
      float inputAspectRatio = (float) inputWidth / inputHeight;

      FaceStylizerResult actualResult = faceStylizer.stylize(inputImage);
      MPImage stylizedImage = actualResult.stylizedImage();
      assertThat(stylizedImage).isNotNull();
      assertThat(stylizedImage.getWidth()).isEqualTo((int) (modelImageSize * inputAspectRatio));
      assertThat(stylizedImage.getHeight()).isEqualTo(modelImageSize);
    }

    @Test
    public void stylizer_succeedsWithRegionOfInterest() throws Exception {
      FaceStylizerOptions options =
          FaceStylizerOptions.builder()
              .setBaseOptions(BaseOptions.builder().setModelAssetPath(modelFile).build())
              .setRunningMode(RunningMode.IMAGE)
              .build();
      faceStylizer =
          FaceStylizer.createFromOptions(ApplicationProvider.getApplicationContext(), options);

      MPImage inputImage = getImageFromAsset(testImage);

      // Region-of-interest around the face.
      RectF roi =
          new RectF(/* left= */ 0.32f, /* top= */ 0.02f, /* right= */ 0.67f, /* bottom= */ 0.32f);
      ImageProcessingOptions imageProcessingOptions =
          ImageProcessingOptions.builder().setRegionOfInterest(roi).build();

      FaceStylizerResult actualResult = faceStylizer.stylize(inputImage, imageProcessingOptions);
      var rectPixelSize = getRectPixelSize(inputImage, roi);

      MPImage stylizedImage = actualResult.stylizedImage();
      assertThat(stylizedImage).isNotNull();
      assertThat(stylizedImage.getWidth()).isEqualTo(rectPixelSize.first);
      assertThat(stylizedImage.getHeight()).isEqualTo(rectPixelSize.second);
    }

    @Test
    public void stylizer_successWithImageModeWithResultListener() throws Exception {
      MPImage inputImage = getImageFromAsset(testImage);
      int inputWidth = inputImage.getWidth();
      int inputHeight = inputImage.getHeight();
      float inputAspectRatio = (float) inputWidth / inputHeight;

      FaceStylizerOptions options =
          FaceStylizerOptions.builder()
              .setBaseOptions(BaseOptions.builder().setModelAssetPath(modelFile).build())
              .setRunningMode(RunningMode.IMAGE)
              .setResultListener(
                  (result, originalImage) -> {
                    assertThat(originalImage).isEqualTo(inputImage);

                    MPImage stylizedImage = result.stylizedImage();
                    assertThat(stylizedImage).isNotNull();
                    assertThat(stylizedImage.getWidth())
                        .isEqualTo(modelImageSize * inputAspectRatio);
                    assertThat(stylizedImage.getHeight()).isEqualTo(modelImageSize);
                  })
              .build();
      faceStylizer =
          FaceStylizer.createFromOptions(ApplicationProvider.getApplicationContext(), options);
      faceStylizer.stylizeWithResultListener(getImageFromAsset(testImage));
    }

    @Test
    public void stylizer_successWithVideoMode() throws Exception {
      MPImage inputImage = getImageFromAsset(testImage);
      int inputWidth = inputImage.getWidth();
      int inputHeight = inputImage.getHeight();
      float inputAspectRatio = (float) inputWidth / inputHeight;

      FaceStylizerOptions options =
          FaceStylizerOptions.builder()
              .setBaseOptions(BaseOptions.builder().setModelAssetPath(modelFile).build())
              .setRunningMode(RunningMode.VIDEO)
              .build();
      faceStylizer =
          FaceStylizer.createFromOptions(ApplicationProvider.getApplicationContext(), options);
      for (int i = 0; i < 3; i++) {
        FaceStylizerResult actualResult =
            faceStylizer.stylizeForVideo(getImageFromAsset(testImage), /* timestampsMs= */ i);

        MPImage stylizedImage = actualResult.stylizedImage();
        assertThat(stylizedImage).isNotNull();
        assertThat(stylizedImage.getWidth()).isEqualTo((int) (modelImageSize * inputAspectRatio));
        assertThat(stylizedImage.getHeight()).isEqualTo(modelImageSize);
      }
    }

    @Test
    public void stylizer_successWithVideoModeWithResultListener() throws Exception {
      MPImage inputImage = getImageFromAsset(testImage);
      int inputWidth = inputImage.getWidth();
      int inputHeight = inputImage.getHeight();
      float inputAspectRatio = (float) inputWidth / inputHeight;

      FaceStylizerOptions options =
          FaceStylizerOptions.builder()
              .setBaseOptions(BaseOptions.builder().setModelAssetPath(modelFile).build())
              .setRunningMode(RunningMode.VIDEO)
              .setResultListener(
                  (result, originalImage) -> {
                    assertThat(originalImage).isEqualTo(inputImage);

                    MPImage stylizedImage = result.stylizedImage();
                    assertThat(stylizedImage).isNotNull();
                    assertThat(stylizedImage.getWidth())
                        .isEqualTo((int) (modelImageSize * inputAspectRatio));
                    assertThat(stylizedImage.getHeight()).isEqualTo(modelImageSize);
                  })
              .build();
      faceStylizer =
          FaceStylizer.createFromOptions(ApplicationProvider.getApplicationContext(), options);
      for (int i = 0; i < 3; i++) {
        faceStylizer.stylizeForVideoWithResultListener(inputImage, /* timestampsMs= */ i);
      }
    }

    @Test
    public void stylizer_successWithLiveStreamMode() throws Exception {
      MPImage inputImage = getImageFromAsset(testImage);
      int inputWidth = inputImage.getWidth();
      int inputHeight = inputImage.getHeight();
      float inputAspectRatio = (float) inputWidth / inputHeight;

      FaceStylizerOptions options =
          FaceStylizerOptions.builder()
              .setBaseOptions(BaseOptions.builder().setModelAssetPath(modelFile).build())
              .setRunningMode(RunningMode.LIVE_STREAM)
              .setResultListener(
                  (result, originalImage) -> {
                    MPImage stylizedImage = result.stylizedImage();
                    assertThat(stylizedImage).isNotNull();
                    assertThat(stylizedImage.getWidth())
                        .isEqualTo((int) (modelImageSize * inputAspectRatio));
                    assertThat(stylizedImage.getHeight()).isEqualTo(modelImageSize);
                  })
              .build();

      faceStylizer =
          FaceStylizer.createFromOptions(ApplicationProvider.getApplicationContext(), options);
      for (int i = 0; i < 3; i++) {
        faceStylizer.stylizeAsync(inputImage, /* timestampsMs= */ i);
      }
    }

    @Test
    public void stylizer_failsWithOutOfOrderInputTimestamps() throws Exception {
      MPImage image = getImageFromAsset(testImage);
      FaceStylizerOptions options =
          FaceStylizerOptions.builder()
              .setBaseOptions(BaseOptions.builder().setModelAssetPath(modelFile).build())
              .setRunningMode(RunningMode.LIVE_STREAM)
              .setResultListener((result, inputImage) -> {})
              .build();
      faceStylizer =
          FaceStylizer.createFromOptions(ApplicationProvider.getApplicationContext(), options);
      faceStylizer.stylizeAsync(image, /* timestampsMs= */ 1);
      MediaPipeException exception =
          assertThrows(
              MediaPipeException.class,
              () -> faceStylizer.stylizeAsync(image, /* timestampsMs= */ 0));
      assertThat(exception)
          .hasMessageThat()
          .contains("having a smaller timestamp than the processed timestamp");
    }
  }

  private static MPImage getImageFromAsset(String filePath) throws Exception {
    AssetManager assetManager = ApplicationProvider.getApplicationContext().getAssets();
    InputStream istr = assetManager.open(filePath);
    return new BitmapImageBuilder(BitmapFactory.decodeStream(istr)).build();
  }
}
