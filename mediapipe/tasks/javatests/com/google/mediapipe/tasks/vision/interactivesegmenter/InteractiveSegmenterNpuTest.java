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
import com.google.mediapipe.tasks.core.BaseOptions.DelegateOptions;
import com.google.mediapipe.tasks.core.Delegate;
import com.google.mediapipe.tasks.vision.imagesegmenter.ImageSegmenterResult;
import com.google.mediapipe.tasks.vision.interactivesegmenter.InteractiveSegmenter.InteractiveSegmenterOptions;
import java.io.InputStream;
import java.nio.ByteBuffer;
import org.junit.Test;
import org.junit.runner.RunWith;

/** Test for {@link InteractiveSegmenter}. */
@RunWith(AndroidJUnit4.class)
public class InteractiveSegmenterNpuTest {
  private static final String MAGIC_TOUCH_MODEL_FILE = "magic_touch_npu_s24.tflite";

  private static final float GOLDEN_MASK_SIMILARITY = 0.94f;

  @Test
  public void segment_successWithCategoryMask() throws Exception {
    final String inputImageName = "cat_large.jpg";
    final String goldenImageName = "cat_large_mask.png";
    final InteractiveSegmenter.RegionOfInterest roi =
        InteractiveSegmenter.RegionOfInterest.create(NormalizedKeypoint.create(0.30f, 0.4f));
    InteractiveSegmenterOptions options =
        InteractiveSegmenterOptions.builder()
            .setBaseOptions(
                BaseOptions.builder()
                    .setModelAssetPath(MAGIC_TOUCH_MODEL_FILE)
                    .setDelegate(Delegate.NPU)
                    .setDelegateOptions(
                        DelegateOptions.NpuOptions.builder()
                            .setDispatchLibraryDirectory(
                                ApplicationProvider.getApplicationContext()
                                    .getApplicationInfo()
                                    .nativeLibraryDir)
                            .build())
                    .build())
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
    MPImage expectedMaskBuffer = getImageFromAsset(goldenImageName);
    verifyCategoryMask(actualMaskBuffer, expectedMaskBuffer, GOLDEN_MASK_SIMILARITY);
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
