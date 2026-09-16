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

#include "absl/log/absl_log.h"
#include "mediapipe/tasks/c/core/base_options.h"
#include "mediapipe/tasks/cc/core/base_options.h"
#include "mediapipe/tasks/cc/core/host_environment.h"

namespace mediapipe::tasks::c::core {
namespace {

mediapipe::tasks::core::BaseOptions::Delegate CppConvertToDelegate(
    const MpDelegate& in) {
  switch (in) {
    case MP_DELEGATE_CPU:
      return mediapipe::tasks::core::BaseOptions::Delegate::CPU;
    case MP_DELEGATE_GPU:
      return mediapipe::tasks::core::BaseOptions::Delegate::GPU;
    case MP_DELEGATE_EDGETPU_NNAPI:
      return mediapipe::tasks::core::BaseOptions::Delegate::EDGETPU_NNAPI;
    case MP_DELEGATE_LITERT:
      return mediapipe::tasks::core::BaseOptions::Delegate::LITERT;
  }
}

mediapipe::tasks::core::HostEnvironment CppConvertToHostEnvironment(
    const MpHostEnvironment& in) {
  switch (in) {
    case MP_HOST_ENVIRONMENT_ANDROID:
      return mediapipe::tasks::core::HostEnvironment::HOST_ENVIRONMENT_ANDROID;
    case MP_HOST_ENVIRONMENT_IOS:
      return mediapipe::tasks::core::HostEnvironment::HOST_ENVIRONMENT_IOS;
    case MP_HOST_ENVIRONMENT_PYTHON:
      return mediapipe::tasks::core::HostEnvironment::HOST_ENVIRONMENT_PYTHON;
    case MP_HOST_ENVIRONMENT_WEB:
      return mediapipe::tasks::core::HostEnvironment::HOST_ENVIRONMENT_WEB;
    default:
      return mediapipe::tasks::core::HostEnvironment::HOST_ENVIRONMENT_UNKNOWN;
  }
}

mediapipe::tasks::core::HostSystem CppConvertToHostSystem(
    const MpHostSystem& in) {
  switch (in) {
    case MP_HOST_SYSTEM_LINUX:
      return mediapipe::tasks::core::HostSystem::HOST_SYSTEM_LINUX;
    case MP_HOST_SYSTEM_MAC:
      return mediapipe::tasks::core::HostSystem::HOST_SYSTEM_MAC;
    case MP_HOST_SYSTEM_WINDOWS:
      return mediapipe::tasks::core::HostSystem::HOST_SYSTEM_WINDOWS;
    case MP_HOST_SYSTEM_IOS:
      return mediapipe::tasks::core::HostSystem::HOST_SYSTEM_IOS;
    case MP_HOST_SYSTEM_ANDROID:
      return mediapipe::tasks::core::HostSystem::HOST_SYSTEM_ANDROID;
    default:
      return mediapipe::tasks::core::HostSystem::HOST_SYSTEM_UNKNOWN;
  }
}

}  // namespace

mediapipe::tasks::core::BaseOptions::LiteRtOptions::CpuOptions
CppConvertToLiteRtCpuOptions(const MpLiteRtCpuOptions& in) {
  mediapipe::tasks::core::BaseOptions::LiteRtOptions::CpuOptions out;
  return out;
}

mediapipe::tasks::core::BaseOptions::LiteRtOptions::GpuOptions
CppConvertToLiteRtGpuOptions(const MpLiteRtGpuOptions& in) {
  mediapipe::tasks::core::BaseOptions::LiteRtOptions::GpuOptions out;
  return out;
}

mediapipe::tasks::core::BaseOptions::LiteRtOptions::NpuOptions
CppConvertToLiteRtNpuOptions(const MpLiteRtNpuOptions& in) {
  mediapipe::tasks::core::BaseOptions::LiteRtOptions::NpuOptions out;
  if (in.dispatch_library_directory) {
    out.dispatch_library_directory = in.dispatch_library_directory;
  }
  return out;
}

mediapipe::tasks::core::BaseOptions::LiteRtOptions CppConvertToLiteRtOptions(
    const MpLiteRtOptions& in) {
  mediapipe::tasks::core::BaseOptions::LiteRtOptions out;
  switch (in.hardware_accelerator) {
    case MP_LITERT_HARDWARE_ACCELERATOR_NPU:
      out.hardware_accelerator = mediapipe::tasks::core::BaseOptions::
          LiteRtOptions::HardwareAccelerator::NPU;
      out.accelerator_options =
          CppConvertToLiteRtNpuOptions(in.accelerator_options.npu_options);
      break;
    case MP_LITERT_HARDWARE_ACCELERATOR_GPU:
      out.hardware_accelerator = mediapipe::tasks::core::BaseOptions::
          LiteRtOptions::HardwareAccelerator::GPU;
      out.accelerator_options =
          CppConvertToLiteRtGpuOptions(in.accelerator_options.gpu_options);
      break;
    case MP_LITERT_HARDWARE_ACCELERATOR_CPU:
    default:
      out.hardware_accelerator = mediapipe::tasks::core::BaseOptions::
          LiteRtOptions::HardwareAccelerator::CPU;
      out.accelerator_options =
          CppConvertToLiteRtCpuOptions(in.accelerator_options.cpu_options);
      break;
  }
  return out;
}

void CppConvertToBaseOptions(const MpBaseOptions& in,
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
  out->num_threads = in.num_threads;
  out->host_environment = CppConvertToHostEnvironment(in.host_environment);
  out->host_system = CppConvertToHostSystem(in.host_system);
  out->host_version = in.host_version ? std::string(in.host_version) : "";
  out->ca_bundle_path = in.ca_bundle_path ? std::string(in.ca_bundle_path) : "";
  out->app_id = in.app_id ? std::string(in.app_id) : "";
  out->app_version = in.app_version ? std::string(in.app_version) : "";
  out->model_asset_descriptor_meta.fd =
      in.file_descriptor != 0 ? in.file_descriptor : -1;

  if (in.delegate == MP_DELEGATE_LITERT && in.litert_options != nullptr) {
    out->delegate_options = CppConvertToLiteRtOptions(*in.litert_options);
  }
  // Don't load GPU service unless delegate is GPU or LiteRT GPU.
  const bool is_gpu =
      (in.delegate == MP_DELEGATE_GPU) ||
      (in.delegate == MP_DELEGATE_LITERT && in.litert_options != nullptr &&
       in.litert_options->hardware_accelerator ==
           MP_LITERT_HARDWARE_ACCELERATOR_GPU);
  out->disable_default_service = !is_gpu;
}

MpHostEnvironment ToMpHostEnvironment(int host_environment) {
  switch (host_environment) {
    case MP_HOST_ENVIRONMENT_UNKNOWN:
      return MP_HOST_ENVIRONMENT_UNKNOWN;
    case MP_HOST_ENVIRONMENT_ANDROID:
      return MP_HOST_ENVIRONMENT_ANDROID;
    case MP_HOST_ENVIRONMENT_IOS:
      return MP_HOST_ENVIRONMENT_IOS;
    case MP_HOST_ENVIRONMENT_PYTHON:
      return MP_HOST_ENVIRONMENT_PYTHON;
    case MP_HOST_ENVIRONMENT_WEB:
      return MP_HOST_ENVIRONMENT_WEB;
    default:
      ABSL_LOG(DFATAL) << "Unknown MpHostEnvironment int value: "
                       << host_environment;
      return MP_HOST_ENVIRONMENT_UNKNOWN;
  }
}

MpHostSystem ToMpHostSystem(int host_system) {
  switch (host_system) {
    case MP_HOST_SYSTEM_UNKNOWN:
      return MP_HOST_SYSTEM_UNKNOWN;
    case MP_HOST_SYSTEM_LINUX:
      return MP_HOST_SYSTEM_LINUX;
    case MP_HOST_SYSTEM_MAC:
      return MP_HOST_SYSTEM_MAC;
    case MP_HOST_SYSTEM_WINDOWS:
      return MP_HOST_SYSTEM_WINDOWS;
    case MP_HOST_SYSTEM_IOS:
      return MP_HOST_SYSTEM_IOS;
    case MP_HOST_SYSTEM_ANDROID:
      return MP_HOST_SYSTEM_ANDROID;
    default:
      ABSL_LOG(DFATAL) << "Unknown MpHostSystem int value: " << host_system;
      return MP_HOST_SYSTEM_UNKNOWN;
  }
}

}  // namespace mediapipe::tasks::c::core
