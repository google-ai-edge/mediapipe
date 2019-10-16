// Copyright 2019 The MediaPipe Authors.
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

package com.google.mediapipe.framework;

import android.content.Context;
import android.content.res.AssetManager;
import com.google.common.io.ByteStreams;
import java.io.IOException;
import java.io.InputStream;

/**
 * Helper methods for handling Android assets.
 */
public final class AndroidAssetUtil {
  /**
   * Returns an asset's contents as a byte array. This is meant to be used in combination with
   * {@link Graph#loadBinaryGraph}.
   *
   * @param assetName The name of an asset, same as in {@link AssetManager#open(String)}.
   */
  public static byte[] getAssetBytes(AssetManager assets, String assetName) {
    byte[] assetData;
    try {
      InputStream stream = assets.open(assetName);
      assetData = ByteStreams.toByteArray(stream);
      stream.close();
    } catch (IOException e) {
      throw new RuntimeException(e);
    }
    return assetData;
  }

  /**
   * Initializes the native asset manager, which is used by native code to access assets directly.
   *
   * <p>Note: When possible, using {@link AssetCache} is preferred for portability, since it does
   * not require any special handling for Android assets on the native code side.
   */
  public static boolean initializeNativeAssetManager(Context androidContext) {
    return nativeInitializeAssetManager(
        androidContext, androidContext.getCacheDir().getAbsolutePath());
  }

  private static native boolean nativeInitializeAssetManager(
      Context androidContext, String cacheDirPath);

  private AndroidAssetUtil() {}
}
