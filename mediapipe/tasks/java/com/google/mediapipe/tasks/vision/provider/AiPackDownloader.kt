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

import android.content.Context
import com.google.android.play.core.aipacks.AiPackManager
import com.google.android.play.core.aipacks.AiPackState
import com.google.android.play.core.aipacks.AiPackStateUpdateListener
import com.google.android.play.core.aipacks.model.AiPackStatus
import java.util.concurrent.Executor

/**
 * Handles downloading of AI Packs using Play for On-Device AI.
 *
 * @property context The application [Context].
 * @property packName The name of the AI Pack to download.
 * @property listener The [DownloadListener] to receive updates on AI Pack download status.
 */
class AiPackDownloader(
  context: Context,
  private val executor: Executor,
  private val aiPackManager: AiPackManager,
  private val packName: String,
  private val listener: DownloadListener<AiPackState>,
) {

  private val internalListener = AiPackStateUpdateListener { state ->
    // Only notify listeners for the current pack.
    if (state.name() == this.packName) {
      updateDownloadStatus(state)
    }
  }

  init {
    aiPackManager.registerListener(internalListener)
  }

  /** Removes the currently set [DownloadListener]. */
  fun removeListener() {
    aiPackManager.unregisterListener(internalListener)
  }

  /** Initiates the download of an AI Pack. */
  fun downloadPack() {
    val unused = aiPackManager.fetch(listOf(this.packName))
  }

  private fun updateDownloadStatus(state: AiPackState) {
    when (state.status()) {
      AiPackStatus.PENDING -> listener.onProgress(0.0F, state)
      AiPackStatus.DOWNLOADING -> {
        val progress = state.bytesDownloaded() * 1.0F / state.totalBytesToDownload()
        listener.onProgress(progress, state)
      }
      AiPackStatus.COMPLETED -> {
        listener.onCompleted()
      }
      AiPackStatus.FAILED,
      AiPackStatus.CANCELED -> {
        listener.onFailed(
          RuntimeException("Download failed with error code: ${state.errorCode()}"),
          state,
        )
      }
      AiPackStatus.WAITING_FOR_WIFI -> {
        // Notify the listener that the download is waiting for Wi-Fi. The user can then
        // decide to initiate a download over cellular data via requestCellularDataDownload.
        listener.onWaitingForWifi(state)
      }
      AiPackStatus.REQUIRES_USER_CONFIRMATION -> {
        listener.onShowConfirmationDialog(state)
      }
      AiPackStatus.NOT_INSTALLED,
      AiPackStatus.UNKNOWN -> {
        // Idle state, no action needed.
      }
    }
  }
}
