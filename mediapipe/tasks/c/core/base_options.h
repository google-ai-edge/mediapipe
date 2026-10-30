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

#ifndef MEDIAPIPE_TASKS_C_CORE_BASE_OPTIONS_H_
#define MEDIAPIPE_TASKS_C_CORE_BASE_OPTIONS_H_

#ifndef MP_EXPORT
#if defined(_MSC_VER)
#define MP_EXPORT __declspec(dllexport)
#else
#define MP_EXPORT __attribute__((visibility("default")))
#endif  // _MSC_VER
#endif  // MP_EXPORT

#ifdef __cplusplus
extern "C" {
#endif

// The delegate to run MediaPipe.
enum MpDelegate {
  MP_DELEGATE_DEFAULT = 0,
  MP_DELEGATE_CPU = 1,
  MP_DELEGATE_GPU = 2,
  MP_DELEGATE_EDGETPU_NNAPI = 3,
  MP_DELEGATE_NPU = 5,
};

// The environment that MediaPipe runs in.
enum MpHostEnvironment {
  MP_HOST_ENVIRONMENT_UNKNOWN = 0,
  MP_HOST_ENVIRONMENT_ANDROID = 1,
  MP_HOST_ENVIRONMENT_IOS = 2,
  MP_HOST_ENVIRONMENT_PYTHON = 3,
  MP_HOST_ENVIRONMENT_WEB = 4,
  MP_HOST_ENVIRONMENT_FLUTTER = 5,
  MP_HOST_ENVIRONMENT_FLUTTER_WEB = 6,
};

// Host OS on which MediaPipe tasks are running.
enum MpHostSystem {
  MP_HOST_SYSTEM_UNKNOWN = 0,
  MP_HOST_SYSTEM_LINUX = 1,
  MP_HOST_SYSTEM_MAC = 2,
  MP_HOST_SYSTEM_WINDOWS = 3,
  MP_HOST_SYSTEM_IOS = 4,
  MP_HOST_SYSTEM_ANDROID = 5,
};

// Base options for MediaPipe C Tasks.
struct MpBaseOptions {
  // The model asset file contents as bytes.
  const char* model_asset_buffer;

  // The size of the model assets buffer (or `0` if not set).
  unsigned int model_asset_buffer_count;

  // The path to the model asset to open and mmap in memory.
  const char* model_asset_path;

  // The file descriptor to a file opened with open(2).
  int file_descriptor;

  // The delegate to use for the MediaPipe graph.
  enum MpDelegate delegate;

  // The environment on which the task is running.
  enum MpHostEnvironment host_environment;

  // The OS on which the task is running.
  enum MpHostSystem host_system;

  // The host version, e.g., python version.
  const char* host_version;

  // The path to CA bundle to use for usage logging.
  const char* ca_bundle_path;

  // The app id of the host environment, e.g., Android package name.
  const char* app_id;

  // The app version of the host environment, e.g., Android version code.
  const char* app_version;
};

// Frees heap-allocated buffer and string fields (`model_asset_buffer`,
// `model_asset_path`, `host_version`, `ca_bundle_path`, `app_id`, and
// `app_version`) in `options` and resets all fields to their default values.
// Only call this if all non-null pointer fields in `options` were allocated on
// the heap (e.g., via `malloc` or `strdup`).
MP_EXPORT void MpBaseOptionsFree(struct MpBaseOptions* options);

#ifdef __cplusplus
}  // extern C
#endif

#endif  // MEDIAPIPE_TASKS_C_CORE_BASE_OPTIONS_H_
