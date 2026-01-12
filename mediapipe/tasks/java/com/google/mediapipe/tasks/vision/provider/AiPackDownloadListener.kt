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

import com.google.android.play.core.aipacks.AiPackState
import com.google.android.play.core.aipacks.model.AiPackStatus
import java.lang.RuntimeException
import java.util.concurrent.atomic.AtomicBoolean

/** Listener for receiving AI Pack download status updates. */
abstract class AiPackDownloadListener : DownloadListener<AiPackState> {
  private val isDone = AtomicBoolean(false)

  internal fun updateState(state: AiPackState, e: Exception? = null) {
    if (isDone.get()) return

    when (state.status()) {
      AiPackStatus.PENDING -> onProgress(0.0f, state)
      AiPackStatus.DOWNLOADING -> {
        val progress = state.bytesDownloaded() * 1.0F / state.totalBytesToDownload()
        onProgress(progress, state)
      }
      AiPackStatus.COMPLETED -> {
        isDone.set(true)
        onCompleted()
      }
      AiPackStatus.FAILED,
      AiPackStatus.CANCELED -> {
        isDone.set(true)
        onFailed(RuntimeException("Download failed with error code: ${state.errorCode()}"), state)
      }
      AiPackStatus.WAITING_FOR_WIFI -> onWaitingForWifi(state)
      AiPackStatus.REQUIRES_USER_CONFIRMATION -> onShowConfirmationDialog(state)
      AiPackStatus.NOT_INSTALLED,
      AiPackStatus.UNKNOWN -> {
        // Idle state, no action needed.
      }
    }
  }
}
