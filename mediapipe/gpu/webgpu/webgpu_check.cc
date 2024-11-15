#include "mediapipe/gpu/webgpu/webgpu_check.h"

#if __EMSCRIPTEN__
#include <emscripten/emscripten.h>
#else
#if MEDIAPIPE_USE_WEBGPU
#include "mediapipe/gpu/webgpu/webgpu_device_registration.h"
#endif  // MEDIAPIPE_USE_WEBGPU
#endif  // __EMSCRIPTEN__

namespace mediapipe {

bool IsWebGpuAvailable() {
#ifdef __EMSCRIPTEN__
  bool webgpu_device_available = [] {
    return EM_ASM_INT({ return !!Module['preinitializedWebGPUDevice']; });
  }();
  return webgpu_device_available;
#else
#if MEDIAPIPE_USE_WEBGPU
  return WebGpuDeviceRegistration::GetInstance().GetWebGpuDevice() != nullptr;
#else
  return false;
#endif  // MEDIAPIPE_USE_WEBGPU
#endif  // __EMSCRIPTEN__
}

}  // namespace mediapipe
