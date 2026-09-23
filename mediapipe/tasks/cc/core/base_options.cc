/* Copyright 2022 The MediaPipe Authors.

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

#include "mediapipe/tasks/cc/core/base_options.h"

#include <cstdint>
#include <fstream>
#include <ios>
#include <memory>

#ifdef _WIN32
#include <io.h>
#include <windows.h>
#else
#include <unistd.h>
#endif
#include <string>
#include <string_view>
#include <utility>
#include <variant>

#include "absl/log/absl_log.h"
#include "mediapipe/calculators/tensor/inference_calculator.pb.h"
#include "mediapipe/tasks/cc/core/proto/acceleration.pb.h"
#include "mediapipe/tasks/cc/core/proto/base_options.pb.h"
#include "mediapipe/tasks/cc/core/proto/external_file.pb.h"

namespace mediapipe {
namespace tasks {
namespace core {
namespace {

using ProtoGpu = mediapipe::InferenceCalculatorOptions::Delegate::LiteRt::Gpu;

ProtoGpu::Precision ConvertGpuPrecisionToProto(
    BaseOptions::GpuOptions::Precision precision) {
  switch (precision) {
    case BaseOptions::GpuOptions::Precision::DEFAULT:
      return ProtoGpu::DEFAULT;
    case BaseOptions::GpuOptions::Precision::FP16:
      return ProtoGpu::FP16;
    case BaseOptions::GpuOptions::Precision::FP32:
      return ProtoGpu::FP32;
  }
  return ProtoGpu::DEFAULT;
}

BaseOptions::GpuOptions::Precision ConvertProtoToGpuPrecision(
    ProtoGpu::Precision precision) {
  switch (precision) {
    case ProtoGpu::DEFAULT:
      return BaseOptions::GpuOptions::Precision::DEFAULT;
    case ProtoGpu::FP16:
      return BaseOptions::GpuOptions::Precision::FP16;
    case ProtoGpu::FP32:
      return BaseOptions::GpuOptions::Precision::FP32;
  }
  return BaseOptions::GpuOptions::Precision::DEFAULT;
}

ProtoGpu::Backend ConvertGpuBackendToProto(
    BaseOptions::GpuOptions::Backend backend) {
  switch (backend) {
    case BaseOptions::GpuOptions::Backend::AUTOMATIC:
      return ProtoGpu::AUTOMATIC;
    case BaseOptions::GpuOptions::Backend::OPENGL:
      return ProtoGpu::OPENGL;
    case BaseOptions::GpuOptions::Backend::OPENCL:
      return ProtoGpu::OPENCL;
    case BaseOptions::GpuOptions::Backend::WEBGPU:
      return ProtoGpu::WEBGPU;
  }
  return ProtoGpu::AUTOMATIC;
}

BaseOptions::GpuOptions::Backend ConvertProtoToGpuBackend(
    ProtoGpu::Backend backend) {
  switch (backend) {
    case ProtoGpu::AUTOMATIC:
      return BaseOptions::GpuOptions::Backend::AUTOMATIC;
    case ProtoGpu::OPENGL:
      return BaseOptions::GpuOptions::Backend::OPENGL;
    case ProtoGpu::OPENCL:
      return BaseOptions::GpuOptions::Backend::OPENCL;
    case ProtoGpu::WEBGPU:
      return BaseOptions::GpuOptions::Backend::WEBGPU;
  }
  return BaseOptions::GpuOptions::Backend::AUTOMATIC;
}

}  // namespace

proto::Acceleration ConvertDelegateOptionsToAccelerationProto(
    const BaseOptions::CpuOptions& options) {
  proto::Acceleration acceleration_proto = proto::Acceleration();
  acceleration_proto.mutable_tflite();
  return acceleration_proto;
}

proto::Acceleration ConvertDelegateOptionsToAccelerationProto(
    const BaseOptions::GpuOptions& options) {
  proto::Acceleration acceleration_proto = proto::Acceleration();
  auto* gpu = acceleration_proto.mutable_gpu();
  gpu->set_use_advanced_gpu_api(true);
  if (!options.cached_kernel_path.empty()) {
    gpu->set_cached_kernel_path(options.cached_kernel_path);
  }
  if (!options.serialized_model_dir.empty()) {
    gpu->set_serialized_model_dir(options.serialized_model_dir);
  }
  if (!options.model_token.empty()) {
    gpu->set_model_token(options.model_token);
  }
  return acceleration_proto;
}

proto::Acceleration ConvertLiteRtDelegateOptionsToAccelerationProto(
    const BaseOptions::CpuOptions& options) {
  proto::Acceleration acceleration_proto;
  auto* cpu = acceleration_proto.mutable_litert()->mutable_cpu();
  return acceleration_proto;
}

proto::Acceleration ConvertLiteRtDelegateOptionsToAccelerationProto(
    const BaseOptions::GpuOptions& options) {
  proto::Acceleration acceleration_proto;
  auto* gpu = acceleration_proto.mutable_litert()->mutable_gpu();
  gpu->set_precision(ConvertGpuPrecisionToProto(options.precision));
  gpu->set_backend(ConvertGpuBackendToProto(options.backend));
  if (!options.cached_kernel_path.empty()) {
    if (options.serialized_model_dir.empty()) {
      ABSL_LOG(WARNING)
          << "BaseOptions::GpuOptions::cached_kernel_path ('"
          << options.cached_kernel_path
          << "') is ignored by the LiteRT GPU accelerator, which has no "
             "separate kernel binary cache. Caching is disabled; set "
             "serialized_model_dir instead.";
    } else {
      ABSL_LOG(WARNING) << "BaseOptions::GpuOptions::cached_kernel_path ('"
                        << options.cached_kernel_path
                        << "') is ignored by the LiteRT GPU accelerator; using "
                           "serialized_model_dir ('"
                        << options.serialized_model_dir
                        << "') as the single serialization location.";
    }
  }
  if (!options.serialized_model_dir.empty()) {
    gpu->mutable_cache_options()->set_serialization_dir(
        options.serialized_model_dir);
  }
  if (!options.model_token.empty()) {
    gpu->mutable_cache_options()->set_model_cache_key(options.model_token);
  }
  if (gpu->has_cache_options() &&
      gpu->cache_options().has_serialization_dir() &&
      gpu->cache_options().has_model_cache_key()) {
    gpu->mutable_cache_options()->set_serialize_program_cache(true);
  }
  return acceleration_proto;
}

proto::Acceleration ConvertLiteRtDelegateOptionsToAccelerationProto(
    const BaseOptions::NpuOptions& options) {
  proto::Acceleration acceleration_proto;
  auto* npu = acceleration_proto.mutable_litert()->mutable_npu();
  if (!options.dispatch_library_directory.empty()) {
    npu->set_dispatch_library_path(options.dispatch_library_directory);
  }
  return acceleration_proto;
}

template <typename T>
void SetDelegateOptionsOrDie(const BaseOptions* base_options,
                             proto::BaseOptions& base_options_proto) {
  if (base_options->delegate_options.has_value()) {
    if (!std::holds_alternative<T>(*base_options->delegate_options)) {
      ABSL_LOG(FATAL) << "Specified Delegate type does not match the provided "
                         "delegate options.";
    } else {
      proto::Acceleration acceleration_proto =
          ConvertDelegateOptionsToAccelerationProto(
              std::get<T>(*base_options->delegate_options));
      base_options_proto.mutable_acceleration()->Swap(&acceleration_proto);
    }
  }
}

template <typename T>
void SetLiteRtDelegateOptionsOrDie(const BaseOptions* base_options,
                                   proto::BaseOptions& base_options_proto) {
  if (base_options->delegate_options.has_value()) {
    if (!std::holds_alternative<T>(*base_options->delegate_options)) {
      ABSL_LOG(FATAL) << "Specified Delegate type does not match the provided "
                         "delegate options.";
    } else {
      // Merge rather than swap, so that the accelerator selection the caller
      // already made on `acceleration` is preserved when the delegate options
      // are applied.
      base_options_proto.mutable_acceleration()->MergeFrom(
          ConvertLiteRtDelegateOptionsToAccelerationProto(
              std::get<T>(*base_options->delegate_options)));
    }
  }
}

proto::BaseOptions ConvertBaseOptionsToProto(BaseOptions* base_options,
                                             bool use_litert) {
  proto::BaseOptions base_options_proto;
  if (!base_options->model_asset_path.empty()) {
    base_options_proto.mutable_model_asset()->set_file_name(
        base_options->model_asset_path);
  }
  if (base_options->model_asset_buffer) {
    base_options_proto.mutable_model_asset()->set_file_content(
        std::move(*base_options->model_asset_buffer));
  }
  if (base_options->model_asset_descriptor_meta.fd > 0) {
    auto* file_descriptor_meta_proto = base_options_proto.mutable_model_asset()
                                           ->mutable_file_descriptor_meta();
    file_descriptor_meta_proto->set_fd(
        base_options->model_asset_descriptor_meta.fd);
    if (base_options->model_asset_descriptor_meta.length > 0) {
      file_descriptor_meta_proto->set_length(
          base_options->model_asset_descriptor_meta.length);
    }
    if (base_options->model_asset_descriptor_meta.offset > 0) {
      file_descriptor_meta_proto->set_offset(
          base_options->model_asset_descriptor_meta.offset);
    }
  }
  switch (base_options->delegate) {
    case BaseOptions::Delegate::CPU:
      if (use_litert) {
        base_options_proto.mutable_acceleration()
            ->mutable_litert()
            ->mutable_cpu();
        SetLiteRtDelegateOptionsOrDie<BaseOptions::CpuOptions>(
            base_options, base_options_proto);
      } else {
        base_options_proto.mutable_acceleration()->mutable_tflite();
        SetDelegateOptionsOrDie<BaseOptions::CpuOptions>(base_options,
                                                         base_options_proto);
      }
      break;
    case BaseOptions::Delegate::GPU:
      if (use_litert) {
        base_options_proto.mutable_acceleration()
            ->mutable_litert()
            ->mutable_gpu();
        // Deliberately GPU-only: CPU is left out of the accelerator set so a
        // GPU request actually runs on the GPU. LiteRT fails compilation when
        // any op is left undelegated, which surfaces models the GPU backend
        // does not fully support. Note this is stricter than the legacy TFLite
        // GPU delegate, which partial-delegates silently.
        SetLiteRtDelegateOptionsOrDie<BaseOptions::GpuOptions>(
            base_options, base_options_proto);
      } else {
        base_options_proto.mutable_acceleration()
            ->mutable_gpu()
            ->set_use_advanced_gpu_api(true);
        SetDelegateOptionsOrDie<BaseOptions::GpuOptions>(base_options,
                                                         base_options_proto);
      }
      break;
    case BaseOptions::Delegate::NPU:
      base_options_proto.mutable_acceleration()
          ->mutable_litert()
          ->mutable_npu();
      SetLiteRtDelegateOptionsOrDie<BaseOptions::NpuOptions>(
          base_options, base_options_proto);
      break;
    case BaseOptions::Delegate::EDGETPU_NNAPI:
      base_options_proto.mutable_acceleration()
          ->mutable_nnapi()
          ->set_accelerator_name("google-edgetpu");
      break;
  }
  return base_options_proto;
}

BaseOptions ConvertProtoToBaseOptions(proto::BaseOptions&& base_options_proto) {
  BaseOptions base_options;
  if (base_options_proto.has_model_asset()) {
    auto* model_asset = base_options_proto.mutable_model_asset();
    if (model_asset->has_file_name()) {
      base_options.model_asset_path = model_asset->file_name();
    }
    if (model_asset->has_file_content()) {
      base_options.model_asset_buffer = std::make_unique<std::string>(
          std::move(*model_asset->mutable_file_content()));
    }
    if (model_asset->has_file_descriptor_meta()) {
      base_options.model_asset_descriptor_meta.fd =
          model_asset->file_descriptor_meta().fd();
      base_options.model_asset_descriptor_meta.length =
          model_asset->file_descriptor_meta().length();
      base_options.model_asset_descriptor_meta.offset =
          model_asset->file_descriptor_meta().offset();
    }
  }
  if (base_options_proto.has_acceleration()) {
    const auto& acceleration = base_options_proto.acceleration();
    if (acceleration.has_gpu()) {
      base_options.delegate = BaseOptions::Delegate::GPU;
      BaseOptions::GpuOptions gpu_options;
      if (acceleration.gpu().has_cached_kernel_path()) {
        gpu_options.cached_kernel_path =
            acceleration.gpu().cached_kernel_path();
      }
      if (acceleration.gpu().has_serialized_model_dir()) {
        gpu_options.serialized_model_dir =
            acceleration.gpu().serialized_model_dir();
      }
      if (acceleration.gpu().has_model_token()) {
        gpu_options.model_token = acceleration.gpu().model_token();
      }
      base_options.delegate_options = std::move(gpu_options);
    } else if (acceleration.has_xnnpack() || acceleration.has_tflite()) {
      base_options.delegate = BaseOptions::Delegate::CPU;
    } else if (acceleration.has_nnapi()) {
      base_options.delegate = BaseOptions::Delegate::EDGETPU_NNAPI;
    } else if (acceleration.has_litert()) {
      // Map the LiteRT accelerator back onto the equivalent `Delegate`.
      // Without this, a proto carrying `litert` falls through to the default
      // `CPU` delegate and silently loses the caller's accelerator choice on a
      // proto -> BaseOptions -> proto round trip.
      //
      // `BaseOptions` has no LiteRT bit, so the engine selection itself is not
      // preserved here; the task re-supplies it via the `use_litert` argument
      // to ConvertBaseOptionsToProto.
      const auto& litert = acceleration.litert();
      if (litert.has_npu()) {
        base_options.delegate = BaseOptions::Delegate::NPU;
        if (litert.npu().has_dispatch_library_path()) {
          BaseOptions::NpuOptions npu_options;
          npu_options.dispatch_library_directory =
              litert.npu().dispatch_library_path();
          base_options.delegate_options = std::move(npu_options);
        }
      } else if (litert.has_gpu()) {
        base_options.delegate = BaseOptions::Delegate::GPU;
        BaseOptions::GpuOptions gpu_options;
        gpu_options.precision =
            ConvertProtoToGpuPrecision(litert.gpu().precision());
        gpu_options.backend = ConvertProtoToGpuBackend(litert.gpu().backend());
        if (litert.gpu().has_cache_options()) {
          const auto& cache_options = litert.gpu().cache_options();
          // `cached_kernel_path` is ignored on LiteRT, so only
          // `serialization_dir` and `model_cache_key` are restored.
          if (cache_options.has_serialization_dir()) {
            gpu_options.serialized_model_dir =
                cache_options.serialization_dir();
          }
          if (cache_options.has_model_cache_key()) {
            gpu_options.model_token = cache_options.model_cache_key();
          }
        }
        base_options.delegate_options = std::move(gpu_options);
      } else {
        base_options.delegate = BaseOptions::Delegate::CPU;
      }
    }
  }
  return base_options;
}

// Portable helper to read from a file descriptor at a given offset without
// changing the file descriptor's current position.
int ReadFdAtOffset(int fd, char* buffer, int length, int64_t offset) {
#ifdef _WIN32
  HANDLE handle = reinterpret_cast<HANDLE>(_get_osfhandle(fd));
  if (handle == INVALID_HANDLE_VALUE) {
    return -1;
  }
  OVERLAPPED overlapped;
  std::memset(&overlapped, 0, sizeof(overlapped));
  overlapped.Offset = static_cast<DWORD>(offset);
  overlapped.OffsetHigh = static_cast<DWORD>(offset >> 32);
  DWORD bytes_read = 0;
  if (!::ReadFile(handle, buffer, static_cast<DWORD>(length), &bytes_read,
                  &overlapped)) {
    return -1;
  }
  return static_cast<int>(bytes_read);
#else
  return static_cast<int>(pread(fd, buffer, length, offset));
#endif
}

bool IsLiteRtLmModel(const BaseOptions& base_options) {
  if (!base_options.model_asset_path.empty()) {
    std::ifstream file(base_options.model_asset_path, std::ios::binary);
    if (file) {
      char header[8];
      file.read(header, 8);
      if (file.gcount() == 8 && std::string_view(header, 8) == "LITERTLM") {
        return true;
      }
    }
  }
  if (base_options.model_asset_buffer != nullptr) {
    if (base_options.model_asset_buffer->size() >= 8 &&
        base_options.model_asset_buffer->substr(0, 8) == "LITERTLM") {
      return true;
    }
  }
  if (base_options.model_asset_descriptor_meta.fd != -1) {
    int fd = base_options.model_asset_descriptor_meta.fd;
    int offset = base_options.model_asset_descriptor_meta.offset;
    if (offset < 0) {
      offset = 0;
    }
    char header[8];
    int bytes_read = ReadFdAtOffset(fd, header, 8, offset);
    if (bytes_read == 8 && std::string_view(header, 8) == "LITERTLM") {
      return true;
    }
  }
  return false;
}

}  // namespace core
}  // namespace tasks
}  // namespace mediapipe
