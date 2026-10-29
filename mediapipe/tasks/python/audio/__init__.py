# Copyright 2022 The MediaPipe Authors.
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

import mediapipe.tasks.python.audio.audio_classifier
import mediapipe.tasks.python.audio.audio_transcriber
import mediapipe.tasks.python.audio.core
import mediapipe.tasks.python.audio.text_to_speech

AudioClassifier = audio_classifier.AudioClassifier
AudioClassifierOptions = audio_classifier.AudioClassifierOptions
AudioClassifierResult = audio_classifier.AudioClassifierResult
AudioTranscriber = audio_transcriber.AudioTranscriber
AudioTranscriberOptions = audio_transcriber.AudioTranscriberOptions
AudioTranscriberResult = audio_transcriber.AudioTranscriberResult
ModelArchitecture = audio_transcriber.ModelArchitecture
RunningMode = core.audio_task_running_mode.AudioTaskRunningMode
TextToSpeech = text_to_speech.TextToSpeech
TextToSpeechOptions = text_to_speech.TextToSpeechOptions
TranscriptionSegment = audio_transcriber.TranscriptionSegment
TtsAudioResult = text_to_speech.TtsAudioResult
TtsStream = text_to_speech.TtsStream
VoiceInfo = text_to_speech.VoiceInfo
WordTimestamp = text_to_speech.WordTimestamp
compute_word_timestamps = text_to_speech.compute_word_timestamps

# Remove unnecessary modules to avoid duplication in API docs.
del audio_classifier
del audio_transcriber
del core
del mediapipe
del text_to_speech
