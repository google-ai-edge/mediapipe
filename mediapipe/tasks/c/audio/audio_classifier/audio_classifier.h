/* Copyright 2025 The MediaPipe Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may not obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#ifndef MEDIAPIPE_TASKS_C_AUDIO_AUDIO_CLASSIFIER_AUDIO_CLASSIFIER_H_
#define MEDIAPIPE_TASKS_C_AUDIO_AUDIO_CLASSIFIER_AUDIO_CLASSIFIER_H_

#include "mediapipe/tasks/c/audio/core/common.h"
#include "mediapipe/tasks/c/components/containers/classification_result.h"
#include "mediapipe/tasks/c/components/processors/classifier_options.h"
#include "mediapipe/tasks/c/core/base_options.h"
#include "mediapipe/tasks/c/core/mp_status.h"

#ifndef MP_EXPORT
#define MP_EXPORT __attribute__((visibility("default")))
#endif  // MP_EXPORT

#ifdef __cplusplus
extern "C" {
#endif

typedef struct MpAudioClassifierInternal* MpAudioClassifierPtr;

// The C representation of a list of audio classification results.
// The caller must call `MpAudioClassifierCloseResult` to free the memory.
typedef struct {
  struct ClassificationResult* results;
  int results_count;
} MpAudioClassifierResult;

// The options for configuring a MediaPipe AudioClassifier task.
struct MpAudioClassifierOptions {
  // Base options for configuring MediaPipe Tasks, such as specifying the model
  // file with metadata, accelerator options, op resolver, etc.
  struct BaseOptions base_options;

  // Options for configuring the classifier behavior, such as score threshold,
  // number of results, etc.
  struct ClassifierOptions classifier_options;

  // The running mode of the audio classifier. Default to the audio clips mode.
  // Audio classifier has two running modes:
  // 1) The audio clips mode for running classification on independent audio
  //    clips.
  // 2) The audio stream mode for running classification on the audio stream,
  //    such as from microphone. In this mode, the "result_callback" below must
  //    be specified to receive the classification results asynchronously.
  MpAudioRunningMode running_mode = kMpAudioRunningModeAudioClips;

  // The user-defined result callback for processing audio stream data.
  // The result callback should only be specified when the running mode is set
  // to kMpAudioRunningModeAudioStream.
  typedef void (*result_callback_fn)(MpStatus status,
                                     MpAudioClassifierResult* result);
  result_callback_fn result_callback;
};

// Creates an AudioClassifier from the provided `options`.
// The caller is responsible for calling `MpAudioClassifierClose` to release the
// classifier.
//
// @param options The options for configuring the audio classifier.
// @param classifier_out A pointer to receive the created audio classifier.
// @return An `MpStatus` indicating success or failure.
MP_EXPORT MpStatus
MpAudioClassifierCreate(struct MpAudioClassifierOptions* options,
                        MpAudioClassifierPtr* classifier_out);

// Performs audio classification on the provided audio clip. Only use this
// method when the AudioClassifier is created with the audio clips running mode.
//
// @param classifier The audio classifier instance.
// @param audio_data The audio data to be classified.
// @param result_out A pointer to receive the classification result.
// @return An `MpStatus` indicating success or failure.
MP_EXPORT MpStatus MpAudioClassifierClassify(
    MpAudioClassifierPtr classifier, const MpAudioData* audio_data,
    MpAudioClassifierResult* result_out);

// Sends audio data (a block in a continuous audio stream) to perform audio
// classification. Only use this method when the AudioClassifier is created
// with the audio stream running mode.
//
// @param classifier The audio classifier instance.
// @param audio_data The audio data to be classified.
// @return An `MpStatus` indicating whether the audio data was successfully sent
//     for classification.
MP_EXPORT MpStatus MpAudioClassifierClassifyAsync(
    MpAudioClassifierPtr classifier, const MpAudioData* audio_data,
    int64_t timestamp_ms);

// Frees the memory allocated inside a MpAudioClassifierResult result. Does not
// free the result pointer itself.
//
// @param result A pointer to the classification result.
// @return An `MpStatus` indicating success or failure.
MP_EXPORT MpStatus
MpAudioClassifierCloseResult(MpAudioClassifierResult* result);

// Shuts down the AudioClassifier when all the work is done. Frees all memory.
//
// @param classifier The audio classifier instance to be closed.
// @return An `MpStatus` indicating success or failure.
MP_EXPORT MpStatus MpAudioClassifierClose(MpAudioClassifierPtr classifier);

#ifdef __cplusplus
}  // extern C
#endif

#endif  // MEDIAPIPE_TASKS_C_AUDIO_AUDIO_CLASSIFIER_AUDIO_CLASSIFIER_H_
