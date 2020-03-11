#include "mediapipe/util/tflite/tensor_buffer.h"

namespace mediapipe {

TensorBuffer::TensorBuffer() {}

TensorBuffer::~TensorBuffer() { uses_gpu_ = false; }

TensorBuffer::TensorBuffer(TfLiteTensor& tensor) {
  cpu_ = tensor;
  uses_gpu_ = false;
}

#if !defined(MEDIAPIPE_DISABLE_GL_COMPUTE)
TensorBuffer::TensorBuffer(std::shared_ptr<tflite::gpu::gl::GlBuffer> tensor) {
  gpu_ = std::move(tensor);
  uses_gpu_ = true;
}
// static
std::shared_ptr<tflite::gpu::gl::GlBuffer> TensorBuffer::CreateGlBuffer(
    std::shared_ptr<mediapipe::GlContext> context) {
  std::shared_ptr<tflite::gpu::gl::GlBuffer> ptr(
      new tflite::gpu::gl::GlBuffer, [context](tflite::gpu::gl::GlBuffer* ref) {
        if (context) {
          context->Run([ref]() {
            if (ref) delete ref;
          });
        } else {
          if (ref) delete ref;  // No context provided.
        }
      });
  return ptr;
}
#endif  // MEDIAPIPE_DISABLE_GL_COMPUTE

#if defined(MEDIAPIPE_IOS)
TensorBuffer::TensorBuffer(id<MTLBuffer> tensor) {
  gpu_ = tensor;
  uses_gpu_ = true;
}
#endif  // MEDIAPIPE_IOS

}  // namespace mediapipe
