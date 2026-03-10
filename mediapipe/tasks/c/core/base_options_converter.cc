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

#include "mediapipe/tasks/c/core/base_options_converter.h"

#include <memory>
#include <string>

#include "mediapipe/tasks/c/core/base_options.h"
#include "mediapipe/tasks/cc/core/base_options.h"
#include "mediapipe/tasks/cc/core/host_environment.h"

namespace mediapipe::tasks::c::core {
namespace {

mediapipe::tasks::core::BaseOptions::Delegate CppConvertToDelegate(
    const Delegate& in) {
  switch (in) {
    case CPU:
      return mediapipe::tasks::core::BaseOptions::Delegate::CPU;
    case GPU:
      return mediapipe::tasks::core::BaseOptions::Delegate::GPU;
    case EDGETPU_NNAPI:
      return mediapipe::tasks::core::BaseOptions::Delegate::EDGETPU_NNAPI;
  }
}

mediapipe::tasks::core::HostEnvironment CppConvertToHostEnvironment(
    const HostEnvironment& in) {
  switch (in) {
    case HOST_ENVIRONMENT_ANDROID:
      return mediapipe::tasks::core::HostEnvironment::HOST_ENVIRONMENT_ANDROID;
    case HOST_ENVIRONMENT_IOS:
      return mediapipe::tasks::core::HostEnvironment::HOST_ENVIRONMENT_IOS;
    case HOST_ENVIRONMENT_PYTHON:
      return mediapipe::tasks::core::HostEnvironment::HOST_ENVIRONMENT_PYTHON;
    default:
      return mediapipe::tasks::core::HostEnvironment::HOST_ENVIRONMENT_UNKNOWN;
  }
}

mediapipe::tasks::core::HostSystem CppConvertToHostSystem(
    const HostSystem& in) {
  switch (in) {
    case HOST_SYSTEM_LINUX:
      return mediapipe::tasks::core::HostSystem::HOST_SYSTEM_LINUX;
    case HOST_SYSTEM_MAC:
      return mediapipe::tasks::core::HostSystem::HOST_SYSTEM_MAC;
    case HOST_SYSTEM_WINDOWS:
      return mediapipe::tasks::core::HostSystem::HOST_SYSTEM_WINDOWS;
    default:
      return mediapipe::tasks::core::HostSystem::HOST_SYSTEM_UNKNOWN;
  }
}

}  // namespace

void CppConvertToBaseOptions(const BaseOptions& in,
                             mediapipe::tasks::core::BaseOptions* out) {
  out->model_asset_buffer =
      in.model_asset_buffer
          ? std::make_unique<std::string>(
                in.model_asset_buffer,
                in.model_asset_buffer + in.model_asset_buffer_count)
          : nullptr;
  out->model_asset_path =
      in.model_asset_path ? std::string(in.model_asset_path) : "";
  out->delegate = CppConvertToDelegate(in.delegate);
  out->host_environment = CppConvertToHostEnvironment(in.host_environment);
  out->host_system = CppConvertToHostSystem(in.host_system);
  if (in.host_version) {
    out->host_version = in.host_version;
  }
}

}  // namespace mediapipe::tasks::c::core
