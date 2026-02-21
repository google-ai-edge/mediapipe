// Copyright 2026 The MediaPipe Authors.
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

package com.google.mediapipe.tasks.vision.provider

/**
 * Listener for receiving model and NPU dispatch library download status updates.
 *
 * @param <T> The type of the state object associated with the download.
 */
interface DownloadListener<T> {
  /**
   * Called when download progress is updated.
   *
   * @param progress The download progress, a float between 0.0 and 1.0.
   * @param state The current state of the download.
   */
  fun onProgress(progress: Float, state: T)

  /** Called when the download is completed. */
  fun onCompleted()

  /**
   * Called when the download fails.
   *
   * @param e The exception that caused the download to fail.
   * @param state The current state of the download.
   */
  fun onFailed(e: Exception, state: T)

  /**
   * Called when the download is paused because it is waiting for Wi-Fi.
   *
   * @param state The current state of the download.
   */
  fun onWaitingForWifi(state: T)

  /**
   * Called when user confirmation is required to proceed with the download (e.g., over cellular
   * data).
   *
   * @param state The current state of the download.
   */
  fun onShowConfirmationDialog(state: T)
}
