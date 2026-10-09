/* Copyright 2024 The MediaPipe Authors.

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
#ifndef MEDIAPIPE_TASKS_C_CORE_MP_LOGGING_H_
#define MEDIAPIPE_TASKS_C_CORE_MP_LOGGING_H_

#include "mediapipe/tasks/c/core/common.h"

#ifdef __cplusplus
extern "C" {
#endif

// A callback function for handling log messages.
// `severity` corresponds to absl::LogSeverity enum values
// (0: INFO, 1: WARNING, 2: ERROR, 3: FATAL)
typedef void (*MpLogCallback)(int severity, const char* message,
                              void* user_data);

// Registers a global logging sink to intercept ABSL logs.
// Only one log callback can be registered at a time. Calling this again
// replaces the previously registered callback.
MP_EXPORT void MpSetLoggingCallback(MpLogCallback callback, void* user_data);

// Removes the global logging sink if one was registered.
MP_EXPORT void MpRemoveLoggingCallback(void);

#ifdef __cplusplus
}
#endif

#endif  // MEDIAPIPE_TASKS_C_CORE_MP_LOGGING_H_
