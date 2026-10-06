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

#include <memory>
#include <string>
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

template <typename T>
void SetDelegateOptionsOrDie(const BaseOptions* base_options,
                             proto::BaseOptions& base_options_proto) {
  if (base_options->delegate_options.has_value()) {
    if (!std::holds_alternative<T>(*base_options->delegate_options)) {
      ABSL_LOG(FATAL) << "Specified Delegate type does not match the provided "
                         "delegate options.";
    } else {
      std::visit(
          [&base_options_proto](const auto& delegate_options) {
            proto::Acceleration acceleration_proto =
                ConvertDelegateOptionsToAccelerationProto(delegate_options);
            base_options_proto.mutable_acceleration()->Swap(
                &acceleration_proto);
          },
          *base_options->delegate_options);
    }
  }
}

proto::BaseOptions ConvertBaseOptionsToProto(BaseOptions* base_options) {
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
      base_options_proto.mutable_acceleration()->mutable_tflite();
      SetDelegateOptionsOrDie<BaseOptions::CpuOptions>(base_options,
                                                       base_options_proto);
      break;
    case BaseOptions::Delegate::GPU:
      base_options_proto.mutable_acceleration()
          ->mutable_gpu()
          ->set_use_advanced_gpu_api(true);
      SetDelegateOptionsOrDie<BaseOptions::GpuOptions>(base_options,
                                                       base_options_proto);
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
    }
  }
  return base_options;
}
}  // namespace core
}  // namespace tasks
}  // namespace mediapipe
