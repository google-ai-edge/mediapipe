#ifndef MEDIAPIPE_FRAMEWORK_FORMATS_TENSOR_HARDWARE_BUFFER_H_
#define MEDIAPIPE_FRAMEWORK_FORMATS_TENSOR_HARDWARE_BUFFER_H_

#if !defined(MEDIAPIPE_NO_JNI) && \
    (__ANDROID_API__ >= 26 ||     \
     defined(__ANDROID_UNAVAILABLE_SYMBOLS_ARE_WEAK__))

#include <android/hardware_buffer.h>

#include <cstdint>

#include "mediapipe/framework/formats/tensor_buffer.h"
#include "mediapipe/framework/formats/tensor_internal.h"
#include "mediapipe/framework/formats/tensor_v2.h"

namespace mediapipe {

// Supports:
// - float 16 and 32 bits
// - signed / unsigned integers 8,16,32 bits
class TensorHardwareBufferView;
struct TensorHardwareBufferViewDescriptor : public Tensor::ViewDescriptor {
  using ViewT = TensorHardwareBufferView;
  TensorBufferDescriptor buffer;
};

class TensorHardwareBufferView : public Tensor::View {
 public:
  TENSOR_UNIQUE_VIEW_TYPE_ID();
  ~TensorHardwareBufferView() = default;

  const TensorHardwareBufferViewDescriptor& descriptor() const override {
    return descriptor_;
  }
  AHardwareBuffer* handle() const { return ahwb_handle_; }

 protected:
  TensorHardwareBufferView(int access_capability, Tensor::View::Access access,
                           Tensor::View::State state,
                           const TensorHardwareBufferViewDescriptor& desc,
                           AHardwareBuffer* ahwb_handle)
      : Tensor::View(kId, access_capability, access, state),
        descriptor_(desc),
        ahwb_handle_(ahwb_handle) {}

 private:
  bool MatchDescriptor(
      uint64_t view_type_id,
      const Tensor::ViewDescriptor& base_descriptor) const override {
    if (!Tensor::View::MatchDescriptor(view_type_id, base_descriptor))
      return false;
    auto descriptor =
        static_cast<const TensorHardwareBufferViewDescriptor&>(base_descriptor);
    return descriptor.buffer.format == descriptor_.buffer.format &&
           descriptor.buffer.size_alignment <=
               descriptor_.buffer.size_alignment &&
           descriptor_.buffer.size_alignment %
                   descriptor.buffer.size_alignment ==
               0;
  }
  const TensorHardwareBufferViewDescriptor& descriptor_;
  AHardwareBuffer* ahwb_handle_ = nullptr;
};

}  // namespace mediapipe

#endif  // !defined(MEDIAPIPE_NO_JNI) && \
    (__ANDROID_API__ >= 26 ||     \
     defined(__ANDROID_UNAVAILABLE_SYMBOLS_ARE_WEAK__))

#endif  // MEDIAPIPE_FRAMEWORK_FORMATS_TENSOR_HARDWARE_BUFFER_H_
