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
import android.content.pm.PackageManager.NameNotFoundException;
import android.content.res.AssetManager;
import android.text.TextUtils;
import androidx.annotation.VisibleForTesting;
import com.google.common.base.Preconditions;
import com.google.common.flogger.FluentLogger;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import javax.annotation.Nullable;

/**
 * A singleton class to help accessing assets as normal files in native code.
 *
 * <p>This class extracts Android assets as files in a cache directory so that they can be accessed
 * by code that expects a regular file path.
 *
 * <p>The cache is automatically purged when the versionCode in the app's manifest changes, to avoid
 * using stale assets. If a versionCode is not specified, the cache is disabled.
 */
public class AssetCache {

  private static final FluentLogger logger = FluentLogger.forEnclosingClass();
  @VisibleForTesting static final String MEDIAPIPE_ASSET_CACHE_DIR = "mediapipe_asset_cache";
  private static AssetCache assetCache;
  private int appVersionCode;
  private AssetCacheDbHelper versionDatabase;
  private Context context;

  /**
   * Create {@link AssetCache} with an Android context.
   *
   * <p>Asset manager needs context to access the asset files. {@link Create} can be called in the
   * main activity.
   */
  public static synchronized AssetCache create(Context context) {
    Preconditions.checkNotNull(context);
    if (assetCache == null) {
      assetCache = new AssetCache(context);
    }
    return assetCache;
  }

  /**
   * Purge the cached assets.
   *
   * <p>This should only be needed in local dev builds that do not update the versionCode in the
   * app's manifest.
   */
  public static synchronized void purgeCache(Context context) {
    AssetCacheDbHelper dbHelper = new AssetCacheDbHelper(context);
    dbHelper.invalidateCache(-1);
    dbHelper.close();
  }

  /**
   * Get {@link AssetCache} without context.
   *
   * <p>If not created, {@code null} is returned.
   */
  @Nullable
  public static synchronized AssetCache getAssetCache() {
    return assetCache;
  }

  /**
   * Loads all the assets in a given assets path.
   * @param assetsPath the assets path from which to load.
   */
  public synchronized void loadAllAssets(String assetsPath) {
    Preconditions.checkNotNull(assetsPath);

    AssetManager assetManager = context.getAssets();
    String[] assetFiles = null;
    try {
      assetFiles = assetManager.list(assetsPath);
    } catch (IOException e) {
      logger.atSevere().withCause(e).log("Unable to get files in assets path: %s", assetsPath);
    }
    if (assetFiles == null || assetFiles.length == 0) {
      logger.atWarning().log("No files to load");
      return;
    }

    for (String file : assetFiles) {
      // If a path was specified, prepend it to the filename with "/", otherwise, just
      // use the file name.
      String path = TextUtils.isEmpty(assetsPath) ? file : assetsPath + "/" + file;
      getAbsolutePathFromAsset(path);
    }
  }

  /**
   * Get the absolute path for an asset file.
   *
   * <p>The asset will be unpacked to the application's files directory if not already done.
   *
   * @param assetPath path to a file under asset.
   * @return the absolute file system path to the unpacked asset file.
   */
  public synchronized String getAbsolutePathFromAsset(String assetPath) {
    AssetManager assetManager = context.getAssets();
    File destinationDir = getDefaultMediaPipeCacheDir();
    destinationDir.mkdir();
    File assetFile = new File(assetPath);
    String assetName = assetFile.getName();
    File destinationFile = new File(destinationDir.getPath(), assetName);
    // If app version code is not defined, we don't use cache.
    if (destinationFile.exists() && appVersionCode != 0
        && versionDatabase.checkVersion(assetPath, appVersionCode)) {
      return destinationFile.getAbsolutePath();
    }
    InputStream inStream = null;
    try {
      inStream = assetManager.open(assetPath);
      writeStreamToFile(inStream, destinationFile);
    } catch (IOException ioe) {
      logger.atSevere().withCause(ioe).log("Unable to unpack: %s", assetPath);
      try {
        if (inStream != null) {
          inStream.close();
        }
      } catch (IOException ioe2) {
        return null;
      }
      return null;
    }
    // If app version code is not defined, we don't use cache.
    if (appVersionCode != 0) {
      versionDatabase.insertAsset(assetPath, destinationFile.getAbsolutePath(), appVersionCode);
    }
    return destinationFile.getAbsolutePath();
  }

  /**
   * Return all the file names of the assets that were saved to cache from the application's
   *     resources.
   */
  public synchronized String[] getAvailableAssets() {
    File assetsDir = getDefaultMediaPipeCacheDir();
    if (assetsDir.exists()) {
      return assetsDir.list();
    }
    return new String[0];
  }

  /**
   * Returns the default cache directory used by the AssetCache to store the assets.
   */
  public File getDefaultMediaPipeCacheDir() {
    return new File(context.getCacheDir(), MEDIAPIPE_ASSET_CACHE_DIR);
  }

  private AssetCache(Context context) {
    this.context = context;
    versionDatabase = new AssetCacheDbHelper(context);
    try {
      appVersionCode = context.getPackageManager()
          .getPackageInfo(context.getPackageName(), 0).versionCode;
      logger.atInfo().log("Current app version code: %d", appVersionCode);
    } catch (NameNotFoundException e) {
      throw new RuntimeException("Can't get app version code.", e);
    }
    // Remove all the cached items that don't agree with the current app version.
    versionDatabase.invalidateCache(appVersionCode);
  }

  private static void writeStreamToFile(InputStream inStream, File destinationFile)
      throws IOException {
    final int bufferSize = 1000;
    FileOutputStream outStream = null;
    try {
      outStream = new FileOutputStream(destinationFile);
      byte[] buffer = new byte[bufferSize];
      while (true) {
        int n = inStream.read(buffer);
        if (n == -1) {
          break;
        }
        outStream.write(buffer, 0, n);
      }
    } finally {
      if (outStream != null) {
        outStream.close();
      }
    }
  }
}
