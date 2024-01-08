#ifndef MEDIAPIPE_FRAMEWORK_FORMATS_AHWB_VIEW_H_
#define MEDIAPIPE_FRAMEWORK_FORMATS_AHWB_VIEW_H_

#include "mediapipe/framework/port.h"
#ifdef MEDIAPIPE_GPU_BUFFER_USE_AHWB
#include "mediapipe/framework/formats/hardware_buffer.h"
#include "mediapipe/gpu/gpu_buffer_storage.h"

namespace mediapipe {

// Wrapper to facilitate short lived access to Android Hardware Buffer objects.
// Intended use cases:
// - Extracting an AHWB for processing in another library after it's produced by
// MediaPipe.
// - Sending AHWBs to compute devices that are able to map the memory for their
// own usage.
// The AHWB abstractions in GpuBuffer and Tensor are likely more suitable for
// other CPU/GPU uses of AHWBs.
class AhwbView {
 public:
  explicit AhwbView(HardwareBuffer* ahwb) : ahwb_(ahwb) {}
  // Non-copyable
  AhwbView(const AhwbView&) = delete;
  AhwbView& operator=(const AhwbView&) = delete;
  // Non-movable
  AhwbView(AhwbView&&) = delete;

  // Only supports synchronous usage. All users of GetHandle must finish
  // accessing the buffer before this view object is destroyed to avoid race
  // conditions.
  // TODO: Support asynchronous usage.
  const AHardwareBuffer* GetHandle() const {
    return ahwb_->GetAHardwareBuffer();
  }

 private:
  const HardwareBuffer* ahwb_;
};

namespace internal {
// Makes this class available as a GpuBuffer view.
template <>
class ViewProvider<AhwbView> {
 public:
  virtual ~ViewProvider() = default;
  virtual const AhwbView GetReadView(types<AhwbView>) const = 0;
  virtual AhwbView GetWriteView(types<AhwbView>) = 0;
};

}  // namespace internal

}  // namespace mediapipe

#endif  // MEDIAPIPE_GPU_BUFFER_USE_AHWB
#endif  // MEDIAPIPE_FRAMEWORK_FORMATS_AHWB_VIEW_H_
