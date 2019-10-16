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

/** {@link MediaPipeRunner} is an abstract class for running MediaPipe graph in Android. */
public abstract class MediaPipeRunner extends Graph {
  protected Context context;

  public MediaPipeRunner(Context context) {
    // Creates a singleton AssetCache.
    AssetCache.create(context);
    this.context = context;
  }

  public void loadBinaryGraphFromAsset(String assetPath) {
    try {
      this.loadBinaryGraph(AssetCache.getAssetCache().getAbsolutePathFromAsset(assetPath));
    } catch (MediaPipeException e) {
      // TODO: Report this error from MediaPipe.
    }
  }

  /**
   * Starts running the graph.
   */
  public abstract void start();
  /**
   * Pauses a running graph.
   */
  public abstract void pause();
  /**
   * Resumes a paused graph.
   */
  public abstract void resume();
  /**
   * Stops the running graph and releases the resource. Call this in Activity onDestroy callback.
   */
  public abstract void release();
}
