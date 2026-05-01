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

#ifdef __cplusplus
extern "C" {
#endif

// The delegate to run MediaPipe.
enum Delegate {
  CPU = 0,
  GPU = 1,
  EDGETPU_NNAPI = 2,
};

// The environment that MediaPipe runs in.
enum HostEnvironment {
  HOST_ENVIRONMENT_UNKNOWN = 0,
  HOST_ENVIRONMENT_ANDROID = 1,
  HOST_ENVIRONMENT_IOS = 2,
  HOST_ENVIRONMENT_PYTHON = 3,
  HOST_ENVIRONMENT_WEB = 4,
};

// Host OS on which MediaPipe tasks are running.
enum HostSystem {
  HOST_SYSTEM_UNKNOWN = 0,
  HOST_SYSTEM_LINUX = 1,
  HOST_SYSTEM_MAC = 2,
  HOST_SYSTEM_WINDOWS = 3,
  HOST_SYSTEM_IOS = 4,
  HOST_SYSTEM_ANDROID = 5,
};

// Base options for MediaPipe C Tasks.
struct BaseOptions {
  // The model asset file contents as bytes.
  const char* model_asset_buffer;

  // The size of the model assets buffer (or `0` if not set).
  unsigned int model_asset_buffer_count;

  // The path to the model asset to open and mmap in memory.
  const char* model_asset_path;

  // The delegate to use for the MediaPipe graph.
  enum Delegate delegate;

  // The environment on which the task is running.
  enum HostEnvironment host_environment;

  // The OS on which the task is running.
  enum HostSystem host_system;

  // The host version, e.g., python version.
  const char* host_version;

  // The path to CA bundle to use for usage logging.
  const char* ca_bundle_path;
};

#ifdef __cplusplus
}  // extern C
#endif

#endif  // MEDIAPIPE_TASKS_C_CORE_BASE_OPTIONS_H_
