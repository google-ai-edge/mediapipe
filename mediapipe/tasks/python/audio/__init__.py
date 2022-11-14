# Copyright 2022 The MediaPipe Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""MediaPipe Tasks Audio API."""

import mediapipe.tasks.python.audio.core
import mediapipe.tasks.python.audio.audio_classifier
import mediapipe.tasks.python.audio.audio_embedder

AudioClassifier = audio_classifier.AudioClassifier
AudioClassifierOptions = audio_classifier.AudioClassifierOptions
AudioClassifierResult = audio_classifier.AudioClassifierResult
AudioEmbedder = audio_embedder.AudioEmbedder
AudioEmbedderOptions = audio_embedder.AudioEmbedderOptions
AudioEmbedderResult = audio_embedder.AudioEmbedderResult
RunningMode = core.audio_task_running_mode.AudioTaskRunningMode

# Remove unnecessary modules to avoid duplication in API docs.
del audio_classifier
del audio_embedder
del core
del mediapipe
