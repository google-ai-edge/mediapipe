// Copyright 2024 The MediaPipe Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "mediapipe/gpu/webgpu/webgpu_device_registration.h"

#include <memory>
#include <utility>

#include "mediapipe/framework/deps/no_destructor.h"
#include "third_party/dawn/include/webgpu/webgpu_cpp.h"

namespace mediapipe {

// static
WebGpuDeviceRegistration& WebGpuDeviceRegistration::GetInstance() {
  static NoDestructor<WebGpuDeviceRegistration> instance;
  return *instance;
}

void WebGpuDeviceRegistration::RegisterWebGpuDevice(wgpu::Device device) {
  device_ = std::move(device);
}

void WebGpuDeviceRegistration::UnRegisterWebGpuDevice() { device_ = nullptr; }

WebGpuDeviceRegistration::WebGpuDeviceRegistration() = default;

WebGpuDeviceRegistration::~WebGpuDeviceRegistration() = default;

}  // namespace mediapipe
