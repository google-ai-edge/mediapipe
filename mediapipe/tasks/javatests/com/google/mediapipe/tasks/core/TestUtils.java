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

package com.google.mediapipe.tasks.core;

import android.content.Context;
import android.content.res.AssetManager;
import com.google.common.io.ByteStreams;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;

/** Helper class for the Java test in MediaPipe Tasks. */
public final class TestUtils {

  /**
   * Loads the file and create a {@link File} object by reading a file from the asset directory.
   * Simulates downloading or reading a file that's not precompiled with the app.
   *
   * @return a {@link File} object for the model.
   */
  public static File loadFile(Context context, String fileName) {
    File target = new File(context.getFilesDir(), fileName);
    try (InputStream is = context.getAssets().open(fileName);
        FileOutputStream os = new FileOutputStream(target)) {
      ByteStreams.copy(is, os);
    } catch (IOException e) {
      throw new AssertionError("Failed to load model file at " + fileName, e);
    }
    return target;
  }

  /**
   * Reads a file into a direct {@link ByteBuffer} object from the asset directory.
   *
   * @return a {@link ByteBuffer} object for the file.
   */
  public static ByteBuffer loadToDirectByteBuffer(Context context, String fileName)
      throws IOException {
    AssetManager assetManager = context.getAssets();
    InputStream inputStream = assetManager.open(fileName);
    byte[] bytes = ByteStreams.toByteArray(inputStream);

    ByteBuffer buffer = ByteBuffer.allocateDirect(bytes.length).order(ByteOrder.nativeOrder());
    buffer.put(bytes);
    return buffer;
  }

  /**
   * Reads a file into a non-direct {@link ByteBuffer} object from the asset directory.
   *
   * @return a {@link ByteBuffer} object for the file.
   */
  public static ByteBuffer loadToNonDirectByteBuffer(Context context, String fileName)
      throws IOException {
    AssetManager assetManager = context.getAssets();
    InputStream inputStream = assetManager.open(fileName);
    byte[] bytes = ByteStreams.toByteArray(inputStream);
    return ByteBuffer.wrap(bytes);
  }

  public enum ByteBufferType {
    DIRECT,
    BACK_UP_ARRAY,
    OTHER // Non-direct ByteBuffer without a back-up array.
  }

  private TestUtils() {}
}
