/* Copyright 2023 The MediaPipe Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#ifndef MEDIAPIPE_TASKS_C_TEXT_TEXT_CLASSIFIER_TEXT_CLASSIFIER_H_
#define MEDIAPIPE_TASKS_C_TEXT_TEXT_CLASSIFIER_TEXT_CLASSIFIER_H_

#include "mediapipe/tasks/c/components/containers/classification_result.h"
#include "mediapipe/tasks/c/components/processors/classifier_options.h"
#include "mediapipe/tasks/c/core/base_options.h"

typedef ClassificationResult TextClassifierResult;

// The options for configuring a MediaPipe text classifier task.
struct TextClassifierOptions {
  // Base options for configuring MediaPipe Tasks, such as specifying the model
  // file with metadata, accelerator options, op resolver, etc.
  struct BaseOptions base_options;

  // Options for configuring the classifier behavior, such as score threshold,
  // number of results, etc.
  struct ClassifierOptions classifier_options;
};

// Creates a TextClassifier from the provided `options`.
void* text_classifier_create(struct TextClassifierOptions options);

// Performs classification on the input `text`.
TextClassifierResult text_classifier_classify(void* classifier,
                                              char* utf8_text);

// Shuts down the TextClassifier when all the work is done. Frees all memory.
void text_classifier_close(void* classifier);

#endif  // MEDIAPIPE_TASKS_C_TEXT_TEXT_CLASSIFIER_TEXT_CLASSIFIER_H_
