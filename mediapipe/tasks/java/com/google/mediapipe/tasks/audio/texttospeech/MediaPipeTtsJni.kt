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

@file:Suppress("PackageName", "KotlinArray")

package com.google.mediapipe.tasks.audio.texttospeech

/** Internal JNI bindings and native library loader for MediaPipe TextToSpeech. */
internal object MediaPipeTtsJni {
  private const val JNI_LIB_NAME = "mediapipe_tasks_text_to_speech_jni"

  init {
    System.loadLibrary(JNI_LIB_NAME)
  }

  external fun nativeCreate(
    modelAssetPath: String,
    delegate: Int,
    speakingRate: Float,
    targetBucket: Int,
    defaultVoiceId: String?,
    defaultLocale: String?,
    cacheDirectory: String,
  ): Long

  external fun nativeListVoices(handle: Long): Array<VoiceInfo>

  external fun nativeSynthesize(
    handle: Long,
    text: String,
    voiceId: String?,
    locale: String?,
  ): AudioResult

  external fun nativeStartStream(
    handle: Long,
    initialText: String?,
    voiceId: String?,
    locale: String?,
    finishInput: Boolean,
  ): Long

  external fun nativeStreamReadNext(streamHandle: Long): AudioChunk?

  external fun nativeStreamPushText(streamHandle: Long, textDelta: String)

  external fun nativeStreamFinishInput(streamHandle: Long)

  external fun nativeStreamCancel(streamHandle: Long)

  external fun nativeStreamDestroy(streamHandle: Long)

  external fun nativeCancel(handle: Long)

  external fun nativeDestroy(handle: Long)
}
