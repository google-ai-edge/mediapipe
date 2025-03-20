// Copyright 2021 The MediaPipe Authors.
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

#include "mediapipe/gpu/webgpu/webgpu_service.h"

#ifdef __EMSCRIPTEN__
#include <cstring>
#endif  // __EMSCRIPTEN__

#include "mediapipe/framework/graph_service.h"

#ifdef __EMSCRIPTEN__
#include <emscripten.h>
#include <emscripten/em_js.h>
#include <webgpu/webgpu_cpp.h>

EM_JS_DEPS(webgpu_service_deps, "$stringToNewUTF8")
#else
#include "mediapipe/gpu/webgpu/webgpu_device_registration.h"
#endif  // __EMSCRIPTEN__

namespace mediapipe {

#ifdef __EMSCRIPTEN__
namespace {

EM_JS(char*, GetAdapterArchitecture, (), {
  const device = Module["preinitializedWebGPUDevice"];
  const architecture =
      device.adapterInfo ? device.adapterInfo.architecture : "Unknown";
  return stringToNewUTF8(architecture);
});

EM_JS(char*, GetAdapterDescription, (), {
  const device = Module["preinitializedWebGPUDevice"];
  const description =
      device.adapterInfo ? device.adapterInfo.description : "Unknown";
  return stringToNewUTF8(description);
});

EM_JS(char*, GetAdapterDeviceName, (), {
  const device = Module["preinitializedWebGPUDevice"];
  const deviceName = device.adapterInfo ? device.adapterInfo.device : "Unknown";
  return stringToNewUTF8(deviceName);
});

EM_JS(char*, GetAdapterVendor, (), {
  const device = Module["preinitializedWebGPUDevice"];
  const vendor = device.adapterInfo ? device.adapterInfo.vendor : "Unknown";
  return stringToNewUTF8(vendor);
});

}  // namespace
#endif  // __EMSCRIPTEN__

#ifdef __EMSCRIPTEN__
WebGpuService::WebGpuService()
    : canvas_selector_("canvas_webgpu"),
      device_(wgpu::Device::Acquire(emscripten_webgpu_get_device())) {
  adapter_info_.architecture.data = GetAdapterArchitecture();
  adapter_info_.architecture.length = strlen(adapter_info_.architecture.data);
  adapter_info_.description.data = GetAdapterDescription();
  adapter_info_.description.length = strlen(adapter_info_.description.data);
  adapter_info_.device.data = GetAdapterDeviceName();
  adapter_info_.device.length = strlen(adapter_info_.device.data);
  adapter_info_.vendor.data = GetAdapterVendor();
  adapter_info_.vendor.length = strlen(adapter_info_.vendor.data);
}
#else
WebGpuService::WebGpuService()
    : canvas_selector_(""),
      device_(WebGpuDeviceRegistration::GetInstance().GetWebGpuDevice()) {}
#endif  // __EMSCRIPTEN__

ABSL_CONST_INIT const GraphService<WebGpuService> kWebGpuService(
    "kWebGpuService", GraphServiceBase::kAllowDefaultInitialization);

}  // namespace mediapipe
