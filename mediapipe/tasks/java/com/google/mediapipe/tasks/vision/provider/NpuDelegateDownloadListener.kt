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

import com.google.android.play.core.splitinstall.SplitInstallSessionState
import com.google.android.play.core.splitinstall.model.SplitInstallSessionStatus
import java.lang.RuntimeException
import java.util.concurrent.atomic.AtomicBoolean

/** Listener for receiving NPU delegate download status updates. */
abstract class NpuDelegateDownloadListener : DownloadListener<SplitInstallSessionState> {
  private val isDone = AtomicBoolean(false)

  internal fun updateState(state: SplitInstallSessionState, e: Exception? = null) {
    if (isDone.get()) return

    when (state.status()) {
      SplitInstallSessionStatus.PENDING -> onProgress(0.0f, state)
      SplitInstallSessionStatus.DOWNLOADING -> {
        val progress =
          if (state.totalBytesToDownload() == 0L) {
            0.0F
          } else {
            state.bytesDownloaded() * 1.0F / state.totalBytesToDownload()
          }
        onProgress(progress, state)
      }
      SplitInstallSessionStatus.DOWNLOADED,
      SplitInstallSessionStatus.INSTALLING -> {
        onProgress(1.0F, state)
      }
      SplitInstallSessionStatus.INSTALLED -> {
        isDone.set(true)
        onCompleted()
      }
      SplitInstallSessionStatus.FAILED,
      SplitInstallSessionStatus.CANCELED -> {
        isDone.set(true)
        onFailed(RuntimeException("Download failed with error code: ${state.errorCode()}"), state)
      }
      SplitInstallSessionStatus.CANCELING -> {
        // No action needed for canceling intermediate state
      }
      SplitInstallSessionStatus.REQUIRES_USER_CONFIRMATION -> onShowConfirmationDialog(state)
      SplitInstallSessionStatus.UNKNOWN -> {
        // Idle state, no action needed.
      }
    }
  }
}
