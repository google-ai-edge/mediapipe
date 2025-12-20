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

#ifndef MEDIAPIPE_TASKS_C_CORE_COMMON_H_
#define MEDIAPIPE_TASKS_C_CORE_COMMON_H_

#include <cstdint>

#ifndef MP_EXPORT
#if defined(_MSC_VER)
#define MP_EXPORT __declspec(dllexport)
#else
#define MP_EXPORT __attribute__((visibility("default")))
#endif  // _WIN32
#endif  // MP_EXPORT

#ifdef __cplusplus
extern "C" {
#endif

// Represents a list of strings.
typedef struct {
  // The array of strings.
  char** strings;
  // The number of strings in the array.
  int num_strings;
} MpStringList;

// Frees an MpStringList.
MP_EXPORT void MpStringListFree(MpStringList* string_list);

// Frees an error message.
MP_EXPORT void MpErrorFree(char* error_message);

#ifdef __cplusplus
}  // extern C
#endif

#endif  // MEDIAPIPE_TASKS_C_CORE_COMMON_H_
