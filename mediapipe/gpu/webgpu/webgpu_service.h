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

#ifndef MEDIAPIPE_GPU_WEBGPU_WEBGPU_SERVICE_H_
#define MEDIAPIPE_GPU_WEBGPU_WEBGPU_SERVICE_H_

#include <memory>
#include <utility>

#include "absl/base/attributes.h"
#include "absl/base/thread_annotations.h"
#include "absl/container/flat_hash_map.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/synchronization/mutex.h"
#include "mediapipe/framework/deps/no_destructor.h"
#include "mediapipe/framework/graph_service.h"
#include "mediapipe/gpu/attachments.h"
#include "mediapipe/gpu/webgpu/webgpu_check.h"
#include "mediapipe/gpu/webgpu/webgpu_headers.h"
#include "mediapipe/web/jspi_check.h"

#ifndef __EMSCRIPTEN__
#include "mediapipe/gpu/webgpu/webgpu_device_registration.h"
#endif  // __EMSCRIPTEN__

namespace mediapipe {

struct WebGpuContext {
  wgpu::Instance instance;
  wgpu::Device device;
};

// Attachments can be used to cache common resources that are associated with
// a device, similarly to what we have for GlContext.
template <class T>
using WebGpuContextAttachment = internal::Attachment<WebGpuContext, T>;

namespace internal {

class WebGpuContextAttachmentManager {
 public:
  explicit WebGpuContextAttachmentManager(wgpu::Instance instance,
                                          wgpu::Device device)
      : context_({std::move(instance), std::move(device)}) {}

  // TODO: const result?
  template <class T>
  T& GetCachedAttachment(const WebGpuContextAttachment<T>& attachment) {
    absl::MutexLock lock(attachments_mutex_);
    internal::AttachmentPtr<void>& entry = attachments_[&attachment];
    if (entry == nullptr) {
      entry = attachment.factory()(context_);
    }
    return *static_cast<T*>(entry.get());
  }

  const wgpu::Device& device() const { return context_.device; }
  const wgpu::Instance& instance() const { return context_.instance; }

 private:
  WebGpuContext context_;
  absl::Mutex attachments_mutex_;
  absl::flat_hash_map<const AttachmentBase<WebGpuContext>*, AttachmentPtr<void>>
      attachments_ ABSL_GUARDED_BY(attachments_mutex_);
};

#ifdef __EMSCRIPTEN__
inline wgpu::Instance CreateEmscriptenInstance() {
  // On the web GPUInstance API doesn't exist and Emscripten just calls
  // into a global event manager. We can just create a new instance with
  // the required features.
  if (IsJspiAvailable()) {
    wgpu::InstanceDescriptor instance_desc = {};
    static const auto kTimedWaitAny = wgpu::InstanceFeatureName::TimedWaitAny;
    instance_desc.requiredFeatureCount = 1;
    instance_desc.requiredFeatures = &kTimedWaitAny;
    auto instance = wgpu::CreateInstance(&instance_desc);
    if (instance.Get() != nullptr) {
      return instance;
    }
    LOG(ERROR) << "Failed to create WebGPU instance with TimedWaitAny feature, "
                  "falling back to instance without it.";
  }
  return wgpu::CreateInstance(nullptr);
}

static WebGpuContextAttachmentManager& GetEmscriptenContextAttachmentManager() {
  static mediapipe::NoDestructor<WebGpuContextAttachmentManager> manager(
      CreateEmscriptenInstance(),
      wgpu::Device::Acquire(emscripten_webgpu_get_device()));
  return *manager;
}
#else
static WebGpuContextAttachmentManager& GetNativeContextAttachmentManager() {
  static mediapipe::NoDestructor<WebGpuContextAttachmentManager> manager(
      wgpu::Instance(
          WebGpuDeviceRegistration::GetInstance().GetWebGpuInstance()),
      wgpu::Device(WebGpuDeviceRegistration::GetInstance().GetWebGpuDevice()));
  return *manager;
}
#endif  // __EMSCRIPTEN__

}  // namespace internal

template <class T>
ABSL_DEPRECATED("Use WebGpuService::GetAttachment instead.")
T& GetWebGpuDeviceCachedAttachment(
    const wgpu::Device& device, const WebGpuContextAttachment<T>& attachment);

class WebGpuService {
 public:
  static absl::StatusOr<std::shared_ptr<WebGpuService>> Create() {
    if (IsWebGpuAvailable()) {
      // Using bare new to invoke private constructor.
      return std::shared_ptr<WebGpuService>(new WebGpuService());
    } else {
      return absl::UnavailableError("WebGpu is not available");
    }
  }

  static absl::StatusOr<std::shared_ptr<WebGpuService>> Create(
      wgpu::Device device, wgpu::Instance instance = nullptr) {
    return std::shared_ptr<WebGpuService>(
        new WebGpuService(std::move(device), std::move(instance)));
  }

  // Note: some clients set DISABLE_DEPRECATED_FIND_EVENT_TARGET_BEHAVIOR=0, so
  // we have to use ids rather than selectors for now.
  // However, note that if we transition to selectors, we will need to change
  // our WebGL canvas handling logic accordingly, and in particular we want to
  // preserve our ability to use canvases not parented to the DOM.
  const char* canvas_selector() const { return canvas_selector_; }

  wgpu::Device device() const { return device_; }
  wgpu::Instance instance() const;

#ifdef __EMSCRIPTEN__
  const wgpu::AdapterInfo& adapter_info() const {
    return *reinterpret_cast<const wgpu::AdapterInfo*>(&adapter_info_);
  }
#endif  // __EMSCRIPTEN__

  template <class T>
  T& GetAttachment(const WebGpuContextAttachment<T>& attachment) {
    return attachment_manager_.GetCachedAttachment<T>(attachment);
  }

 private:
  WebGpuService();
  WebGpuService(wgpu::Device device, wgpu::Instance instance);

  const char* canvas_selector_;
  wgpu::Instance instance_;
  wgpu::Device device_;
#ifdef __EMSCRIPTEN__
  // Adapter is not yet piped through Emscripten (i.e. `device.GetAdapter()`).
  // Instead we pass GPUAdapterInfo obtained in TypeScript via
  // `GPUAdapter.requestAdapterInfo()` as a part of
  // `preinitializedWebGPUDevice`. Ideally we would want to pass it as a
  // separate object (or more precisely pointer to object in JsValStore), but
  // MediaPipe services don't support parameterized constructors.
  //
  WGPUAdapterInfo adapter_info_;
#endif  // __EMSCRIPTEN__

  internal::WebGpuContextAttachmentManager attachment_manager_;
};

ABSL_CONST_INIT extern const GraphService<WebGpuService> kWebGpuService;

#ifdef __EMSCRIPTEN__
template <class T>
T& GetWebGpuDeviceCachedAttachment(
    const wgpu::Device& device, const WebGpuContextAttachment<T>& attachment) {
  // Currently we only handle the single device given to Emscripten.
  auto& attachments = internal::GetEmscriptenContextAttachmentManager();
  // Note: emscripten_webgpu_get_device, in spite of its name, creates a new
  // wrapper with a new handle each time it's called, even though they all
  // refer to the same device. TODO: fix it in upstream. For now we
  // just rely the assumption that there is one device.
  // ABSL_CHECK_EQ(device.Get(), attachments.device().Get());
  //
  // Note: GetWebGpuDeviceCachedAttachment free function is deprecated in favor
  // of WebGpuService::GetAttachment and will be removed eventually.
  return attachments.GetCachedAttachment<T>(attachment);
}
#else
template <class T>
T& GetWebGpuDeviceCachedAttachment(
    const wgpu::Device& device, const WebGpuContextAttachment<T>& attachment) {
  // Note: GetWebGpuDeviceCachedAttachment free function is deprecated in favor
  // of WebGpuService::GetAttachment and will be removed eventually.
  auto& attachments = internal::GetNativeContextAttachmentManager();
  return attachments.GetCachedAttachment<T>(attachment);
}
#endif  // __EMSCRIPTEN__

}  // namespace mediapipe

#endif  // MEDIAPIPE_GPU_WEBGPU_WEBGPU_SERVICE_H_
