/* Copyright 2026 The MediaPipe Authors.

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

#include "mediapipe/tasks/c/core/base_options.h"

#include <cstdlib>

extern "C" {

void MpBaseOptionsFree(struct MpBaseOptions* options) {
  if (options == nullptr) {
    return;
  }
  free(const_cast<char*>(options->model_asset_buffer));
  options->model_asset_buffer = nullptr;
  options->model_asset_buffer_count = 0;

  free(const_cast<char*>(options->model_asset_path));
  options->model_asset_path = nullptr;

  options->file_descriptor = -1;
  options->delegate = MP_DELEGATE_DEFAULT;
  options->host_environment = MP_HOST_ENVIRONMENT_UNKNOWN;
  options->host_system = MP_HOST_SYSTEM_UNKNOWN;

  free(const_cast<char*>(options->host_version));
  options->host_version = nullptr;

  free(const_cast<char*>(options->ca_bundle_path));
  options->ca_bundle_path = nullptr;

  free(const_cast<char*>(options->app_id));
  options->app_id = nullptr;

  free(const_cast<char*>(options->app_version));
  options->app_version = nullptr;
}

}  // extern "C"
