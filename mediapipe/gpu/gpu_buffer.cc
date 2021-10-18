#include "mediapipe/gpu/gpu_buffer.h"

#include "mediapipe/gpu/gl_context.h"

#if MEDIAPIPE_GPU_BUFFER_USE_CV_PIXEL_BUFFER
#include "mediapipe/objc/util.h"
#endif  // MEDIAPIPE_GPU_BUFFER_USE_CV_PIXEL_BUFFER

namespace mediapipe {

#if MEDIAPIPE_GPU_BUFFER_USE_CV_PIXEL_BUFFER

GpuBuffer GpuBuffer::CopyingImageFrame(const ImageFrame& image_frame) {
  auto maybe_buffer = CreateCVPixelBufferCopyingImageFrame(image_frame);
  // Converts absl::StatusOr to absl::Status since CHECK_OK() currently only
  // deals with absl::Status in MediaPipe OSS.
  CHECK_OK(maybe_buffer.status());
  return GpuBuffer(std::move(maybe_buffer).value());
}
#endif  // MEDIAPIPE_GPU_BUFFER_USE_CV_PIXEL_BUFFER

#if !MEDIAPIPE_GPU_BUFFER_USE_CV_PIXEL_BUFFER
GpuBuffer GpuBuffer::CopyingImageFrame(const ImageFrame& image_frame) {
  return GpuBuffer(GlTextureBuffer::Create(image_frame));
}

#endif  // !MEDIAPIPE_GPU_BUFFER_USE_CV_PIXEL_BUFFER

}  // namespace mediapipe
