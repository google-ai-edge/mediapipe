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

package com.google.mediapipe.tasks.vision.facestylizer;

import static com.google.common.truth.Truth.assertThat;
import static org.junit.Assert.assertThrows;

import android.content.res.AssetManager;
import android.graphics.BitmapFactory;
import android.graphics.RectF;
import androidx.test.core.app.ApplicationProvider;
import androidx.test.ext.junit.runners.AndroidJUnit4;
import com.google.mediapipe.framework.image.BitmapImageBuilder;
import com.google.mediapipe.framework.image.MPImage;
import com.google.mediapipe.tasks.core.BaseOptions;
import com.google.mediapipe.tasks.vision.core.ImageProcessingOptions;
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
  private static final String modelFile = "face_stylizer.task";
  private static final String largeFaceTestImage = "portrait.jpg";
  private static final String smallFaceTestImage = "portrait_small.jpg";
  private static final int modelImageSize = 256;

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
    public void stylizer_succeedsWithImageMode() throws Exception {
      FaceStylizerOptions options =
          FaceStylizerOptions.builder()
              .setBaseOptions(BaseOptions.builder().setModelAssetPath(modelFile).build())
              .build();

      faceStylizer =
          FaceStylizer.createFromOptions(ApplicationProvider.getApplicationContext(), options);

      MPImage inputImage = getImageFromAsset(largeFaceTestImage);
      FaceStylizerResult actualResult = faceStylizer.stylize(inputImage);
      MPImage stylizedImage = actualResult.stylizedImage().get();
      assertThat(stylizedImage).isNotNull();
      assertThat(stylizedImage.getWidth()).isEqualTo(modelImageSize);
      assertThat(stylizedImage.getHeight()).isEqualTo(modelImageSize);
    }

    @Test
    public void stylizer_succeedsWithSmallImage() throws Exception {
      FaceStylizerOptions options =
          FaceStylizerOptions.builder()
              .setBaseOptions(BaseOptions.builder().setModelAssetPath(modelFile).build())
              .build();

      faceStylizer =
          FaceStylizer.createFromOptions(ApplicationProvider.getApplicationContext(), options);

      MPImage inputImage = getImageFromAsset(smallFaceTestImage);
      FaceStylizerResult actualResult = faceStylizer.stylize(inputImage);
      MPImage stylizedImage = actualResult.stylizedImage().get();
      assertThat(stylizedImage).isNotNull();
      assertThat(stylizedImage.getWidth()).isEqualTo(modelImageSize);
      assertThat(stylizedImage.getHeight()).isEqualTo(modelImageSize);
    }

    @Test
    public void stylizer_succeedsWithRegionOfInterest() throws Exception {
      FaceStylizerOptions options =
          FaceStylizerOptions.builder()
              .setBaseOptions(BaseOptions.builder().setModelAssetPath(modelFile).build())
              .build();
      faceStylizer =
          FaceStylizer.createFromOptions(ApplicationProvider.getApplicationContext(), options);

      MPImage inputImage = getImageFromAsset(largeFaceTestImage);

      // Region-of-interest around the face.
      RectF roi =
          new RectF(/* left= */ 0.32f, /* top= */ 0.02f, /* right= */ 0.67f, /* bottom= */ 0.32f);
      ImageProcessingOptions imageProcessingOptions =
          ImageProcessingOptions.builder().setRegionOfInterest(roi).build();

      FaceStylizerResult actualResult = faceStylizer.stylize(inputImage, imageProcessingOptions);
      MPImage stylizedImage = actualResult.stylizedImage().get();
      assertThat(stylizedImage).isNotNull();
      assertThat(stylizedImage.getWidth()).isEqualTo(modelImageSize);
      assertThat(stylizedImage.getHeight()).isEqualTo(modelImageSize);
    }

    @Test
    public void stylizer_succeedsWithNoFaceDetected() throws Exception {
      FaceStylizerOptions options =
          FaceStylizerOptions.builder()
              .setBaseOptions(BaseOptions.builder().setModelAssetPath(modelFile).build())
              .build();
      faceStylizer =
          FaceStylizer.createFromOptions(ApplicationProvider.getApplicationContext(), options);

      MPImage inputImage = getImageFromAsset(largeFaceTestImage);

      // Region-of-interest that doesn't contain a human face.
      RectF roi =
          new RectF(/* left= */ 0.1f, /* top= */ 0.1f, /* right= */ 0.2f, /* bottom= */ 0.2f);
      ImageProcessingOptions imageProcessingOptions =
          ImageProcessingOptions.builder().setRegionOfInterest(roi).build();
      FaceStylizerResult actualResult = faceStylizer.stylize(inputImage, imageProcessingOptions);
      assertThat(actualResult.stylizedImage()).isNotNull();
      assertThat(actualResult.stylizedImage().isPresent()).isFalse();
    }

    @Test
    public void stylizer_successWithImageModeWithResultListener() throws Exception {
      MPImage inputImage = getImageFromAsset(largeFaceTestImage);
      FaceStylizerOptions options =
          FaceStylizerOptions.builder()
              .setBaseOptions(BaseOptions.builder().setModelAssetPath(modelFile).build())
              .setResultListener(
                  (result, originalImage) -> {
                    MPImage stylizedImage = result.stylizedImage().get();
                    assertThat(stylizedImage).isNotNull();
                    assertThat(stylizedImage.getWidth()).isEqualTo(modelImageSize);
                    assertThat(stylizedImage.getHeight()).isEqualTo(modelImageSize);
                  })
              .build();
      faceStylizer =
          FaceStylizer.createFromOptions(ApplicationProvider.getApplicationContext(), options);
      faceStylizer.stylizeWithResultListener(getImageFromAsset(largeFaceTestImage));
    }
  }

  private static MPImage getImageFromAsset(String filePath) throws Exception {
    AssetManager assetManager = ApplicationProvider.getApplicationContext().getAssets();
    InputStream istr = assetManager.open(filePath);
    return new BitmapImageBuilder(BitmapFactory.decodeStream(istr)).build();
  }
}
