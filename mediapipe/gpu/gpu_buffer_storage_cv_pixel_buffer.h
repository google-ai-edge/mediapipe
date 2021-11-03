#ifndef MEDIAPIPE_GPU_GPU_BUFFER_STORAGE_CV_PIXEL_BUFFER_H_
#define MEDIAPIPE_GPU_GPU_BUFFER_STORAGE_CV_PIXEL_BUFFER_H_

#include <CoreVideo/CoreVideo.h>

#include "mediapipe/gpu/gl_texture_view.h"
#include "mediapipe/gpu/gpu_buffer_storage.h"
#include "mediapipe/objc/CFHolder.h"

namespace mediapipe {

class GlContext;

class GpuBufferStorageCvPixelBuffer
    : public mediapipe::internal::GpuBufferStorageImpl<
          GpuBufferStorageCvPixelBuffer,
          mediapipe::internal::ViewProvider<GlTextureView>>,
      public CFHolder<CVPixelBufferRef> {
 public:
  using CFHolder<CVPixelBufferRef>::CFHolder;
  GpuBufferStorageCvPixelBuffer(const CFHolder<CVPixelBufferRef>& other)
      : CFHolder(other) {}
  GpuBufferStorageCvPixelBuffer(CFHolder<CVPixelBufferRef>&& other)
      : CFHolder(std::move(other)) {}
  int width() const { return static_cast<int>(CVPixelBufferGetWidth(**this)); }
  int height() const {
    return static_cast<int>(CVPixelBufferGetHeight(**this));
  }
  virtual GpuBufferFormat format() const {
    return GpuBufferFormatForCVPixelFormat(
        CVPixelBufferGetPixelFormatType(**this));
  }
  GlTextureView GetReadView(mediapipe::internal::types<GlTextureView>,
                            std::shared_ptr<GpuBuffer> gpu_buffer,
                            int plane) const override;
  GlTextureView GetWriteView(mediapipe::internal::types<GlTextureView>,
                             std::shared_ptr<GpuBuffer> gpu_buffer,
                             int plane) override;
  std::unique_ptr<ImageFrame> AsImageFrame() const override;

 private:
  void ViewDoneWriting(const GlTextureView& view);
};

}  // namespace mediapipe

#endif  // MEDIAPIPE_GPU_GPU_BUFFER_STORAGE_CV_PIXEL_BUFFER_H_
