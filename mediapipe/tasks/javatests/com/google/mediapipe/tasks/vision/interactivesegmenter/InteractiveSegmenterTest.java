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

package com.google.mediapipe.tasks.vision.interactivesegmenter;

import static com.google.common.truth.Truth.assertThat;

import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Color;
import androidx.test.core.app.ApplicationProvider;
import androidx.test.ext.junit.runners.AndroidJUnit4;
import com.google.mediapipe.framework.image.BitmapExtractor;
import com.google.mediapipe.framework.image.BitmapImageBuilder;
import com.google.mediapipe.framework.image.ByteBufferExtractor;
import com.google.mediapipe.framework.image.MPImage;
import com.google.mediapipe.tasks.components.containers.NormalizedKeypoint;
import com.google.mediapipe.tasks.core.BaseOptions;
import com.google.mediapipe.tasks.vision.imagesegmenter.ImageSegmenterResult;
import com.google.mediapipe.tasks.vision.interactivesegmenter.InteractiveSegmenter.InteractiveSegmenterOptions;
import java.io.InputStream;
import java.nio.ByteBuffer;
import java.util.ArrayList;
import java.util.List;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Suite;
import org.junit.runners.Suite.SuiteClasses;

/** Test for {@link InteractiveSegmenter}. */
@RunWith(Suite.class)
@SuiteClasses({
  InteractiveSegmenterTest.KeypointRoi.class,
  InteractiveSegmenterTest.ScribbleRoi.class,
})
public class InteractiveSegmenterTest {
  private static final String DEEPLAB_MODEL_FILE = "ptm_512_hdt_ptm_woid.tflite";
  private static final String MAGIC_TOUCH_MODEL_FILE = "magic_touch.tflite";
  private static final String CATS_AND_DOGS_IMAGE = "cats_and_dogs.jpg";
  private static final String CAT_IMAGE = "cat_large.jpg";
  private static final float GOLDEN_MASK_SIMILARITY = 0.94f;

  @RunWith(AndroidJUnit4.class)
  public static final class KeypointRoi extends InteractiveSegmenterTest {
    @Test
    public void segment_successWithCategoryMask() throws Exception {
      final String inputImageName = CATS_AND_DOGS_IMAGE;
      final InteractiveSegmenter.RegionOfInterest roi =
          InteractiveSegmenter.RegionOfInterest.create(NormalizedKeypoint.create(0.25f, 0.9f));
      InteractiveSegmenterOptions options =
          InteractiveSegmenterOptions.builder()
              .setBaseOptions(BaseOptions.builder().setModelAssetPath(DEEPLAB_MODEL_FILE).build())
              .setOutputConfidenceMasks(false)
              .setOutputCategoryMask(true)
              .build();
      InteractiveSegmenter interactiveSegmenter =
          InteractiveSegmenter.createFromOptions(
              ApplicationProvider.getApplicationContext(), options);
      MPImage image = getImageFromAsset(inputImageName);
      ImageSegmenterResult actualResult = interactiveSegmenter.segment(image, roi);
      assertThat(actualResult.categoryMask().isPresent()).isTrue();
    }

    @Test
    public void segment_successWithConfidenceMask() throws Exception {
      final String inputImageName = CATS_AND_DOGS_IMAGE;
      final InteractiveSegmenter.RegionOfInterest roi =
          InteractiveSegmenter.RegionOfInterest.create(NormalizedKeypoint.create(0.3f, 0.4f));
      InteractiveSegmenterOptions options =
          InteractiveSegmenterOptions.builder()
              .setBaseOptions(BaseOptions.builder().setModelAssetPath(DEEPLAB_MODEL_FILE).build())
              .setOutputConfidenceMasks(true)
              .setOutputCategoryMask(false)
              .build();
      InteractiveSegmenter interactiveSegmenter =
          InteractiveSegmenter.createFromOptions(
              ApplicationProvider.getApplicationContext(), options);
      ImageSegmenterResult actualResult =
          interactiveSegmenter.segment(getImageFromAsset(inputImageName), roi);
      assertThat(actualResult.confidenceMasks().isPresent()).isTrue();
      List<MPImage> confidenceMasks = actualResult.confidenceMasks().get();
      assertThat(confidenceMasks.size()).isEqualTo(2);
    }

    @Test
    public void segment_successWithMagicTouch() throws Exception {
      final String inputImageName = CAT_IMAGE;
      final InteractiveSegmenter.RegionOfInterest roi =
          InteractiveSegmenter.RegionOfInterest.create(NormalizedKeypoint.create(0.30f, 0.4f));
      InteractiveSegmenterOptions options =
          InteractiveSegmenterOptions.builder()
              .setBaseOptions(
                  BaseOptions.builder().setModelAssetPath(MAGIC_TOUCH_MODEL_FILE).build())
              .setOutputConfidenceMasks(false)
              .setOutputCategoryMask(true)
              .build();
      InteractiveSegmenter interactiveSegmenter =
          InteractiveSegmenter.createFromOptions(
              ApplicationProvider.getApplicationContext(), options);
      ImageSegmenterResult actualResult =
          interactiveSegmenter.segment(getImageFromAsset(inputImageName), roi);
      assertThat(actualResult.categoryMask()).isPresent();
      MPImage actualMaskBuffer = actualResult.categoryMask().get();
      MPImage expectedMaskBuffer = getImageFromAsset("cat_large_mask.png");
      verifyCategoryMask(actualMaskBuffer, expectedMaskBuffer, GOLDEN_MASK_SIMILARITY);
    }
  }

  @RunWith(AndroidJUnit4.class)
  public static final class ScribbleRoi extends InteractiveSegmenterTest {
    @Test
    public void segment_successWithCategoryMask() throws Exception {
      final String inputImageName = CATS_AND_DOGS_IMAGE;
      ArrayList<NormalizedKeypoint> scribble = new ArrayList<>();
      scribble.add(NormalizedKeypoint.create(0.25f, 0.9f));
      scribble.add(NormalizedKeypoint.create(0.25f, 0.91f));
      scribble.add(NormalizedKeypoint.create(0.25f, 0.92f));
      final InteractiveSegmenter.RegionOfInterest roi =
          InteractiveSegmenter.RegionOfInterest.create(scribble);
      InteractiveSegmenterOptions options =
          InteractiveSegmenterOptions.builder()
              .setBaseOptions(BaseOptions.builder().setModelAssetPath(DEEPLAB_MODEL_FILE).build())
              .setOutputConfidenceMasks(false)
              .setOutputCategoryMask(true)
              .build();
      InteractiveSegmenter interactiveSegmenter =
          InteractiveSegmenter.createFromOptions(
              ApplicationProvider.getApplicationContext(), options);
      MPImage image = getImageFromAsset(inputImageName);
      ImageSegmenterResult actualResult = interactiveSegmenter.segment(image, roi);
      assertThat(actualResult.categoryMask().isPresent()).isTrue();
    }

    @Test
    public void segment_successWithConfidenceMask() throws Exception {
      final String inputImageName = CATS_AND_DOGS_IMAGE;
      ArrayList<NormalizedKeypoint> scribble = new ArrayList<>();
      scribble.add(NormalizedKeypoint.create(0.25f, 0.9f));
      scribble.add(NormalizedKeypoint.create(0.25f, 0.91f));
      scribble.add(NormalizedKeypoint.create(0.25f, 0.92f));
      final InteractiveSegmenter.RegionOfInterest roi =
          InteractiveSegmenter.RegionOfInterest.create(scribble);
      InteractiveSegmenterOptions options =
          InteractiveSegmenterOptions.builder()
              .setBaseOptions(BaseOptions.builder().setModelAssetPath(DEEPLAB_MODEL_FILE).build())
              .setOutputConfidenceMasks(true)
              .setOutputCategoryMask(false)
              .build();
      InteractiveSegmenter interactiveSegmenter =
          InteractiveSegmenter.createFromOptions(
              ApplicationProvider.getApplicationContext(), options);
      ImageSegmenterResult actualResult =
          interactiveSegmenter.segment(getImageFromAsset(inputImageName), roi);
      assertThat(actualResult.confidenceMasks().isPresent()).isTrue();
      List<MPImage> confidenceMasks = actualResult.confidenceMasks().get();
      assertThat(confidenceMasks.size()).isEqualTo(2);
    }
  }

  private static void verifyCategoryMask(
      MPImage actualMask, MPImage goldenMask, float similarityThreshold) {
    assertThat(actualMask.getWidth()).isEqualTo(goldenMask.getWidth());
    assertThat(actualMask.getHeight()).isEqualTo(goldenMask.getHeight());
    ByteBuffer actualMaskBuffer = ByteBufferExtractor.extract(actualMask);
    Bitmap goldenMaskBitmap = BitmapExtractor.extract(goldenMask);
    int consistentPixels = 0;
    final int numPixels = actualMask.getWidth() * actualMask.getHeight();
    actualMaskBuffer.rewind();
    for (int y = 0; y < actualMask.getHeight(); y++) {
      for (int x = 0; x < actualMask.getWidth(); x++) {
        boolean actualForground = actualMaskBuffer.get() != 0;
        boolean goldenForeground = goldenMaskBitmap.getPixel(x, y) != Color.BLACK;
        if (actualForground == goldenForeground) {
          ++consistentPixels;
        }
      }
    }
    assertThat((float) consistentPixels / numPixels).isGreaterThan(similarityThreshold);
  }

  private static MPImage getImageFromAsset(String filePath) throws Exception {
    AssetManager assetManager = ApplicationProvider.getApplicationContext().getAssets();
    InputStream istr = assetManager.open(filePath);
    return new BitmapImageBuilder(BitmapFactory.decodeStream(istr)).build();
  }
}
