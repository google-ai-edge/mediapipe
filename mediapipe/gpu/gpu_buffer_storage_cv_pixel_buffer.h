#ifndef MEDIAPIPE_GPU_GPU_BUFFER_STORAGE_CV_PIXEL_BUFFER_H_
#define MEDIAPIPE_GPU_GPU_BUFFER_STORAGE_CV_PIXEL_BUFFER_H_

#include <CoreVideo/CoreVideo.h>

#include "mediapipe/gpu/gl_texture_view.h"
#include "mediapipe/gpu/gpu_buffer_storage.h"
#include "mediapipe/gpu/image_frame_view.h"
#include "mediapipe/objc/CFHolder.h"

namespace mediapipe {

class GlContext;

class GpuBufferStorageCvPixelBuffer
    : public internal::GpuBufferStorageImpl<
          GpuBufferStorageCvPixelBuffer, internal::ViewProvider<GlTextureView>,
          internal::ViewProvider<ImageFrame>>,
      public CFHolder<CVPixelBufferRef> {
 public:
  using CFHolder<CVPixelBufferRef>::CFHolder;
  GpuBufferStorageCvPixelBuffer(int width, int height, GpuBufferFormat format);
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
  GlTextureView GetReadView(internal::types<GlTextureView>,
                            std::shared_ptr<GpuBuffer> gpu_buffer,
                            int plane) const override;
  GlTextureView GetWriteView(internal::types<GlTextureView>,
                             std::shared_ptr<GpuBuffer> gpu_buffer,
                             int plane) override;
  std::shared_ptr<const ImageFrame> GetReadView(
      internal::types<ImageFrame>,
      std::shared_ptr<GpuBuffer> gpu_buffer) const override;
  std::shared_ptr<ImageFrame> GetWriteView(
      internal::types<ImageFrame>,
      std::shared_ptr<GpuBuffer> gpu_buffer) override;

 private:
  GlTextureView GetTexture(std::shared_ptr<GpuBuffer> gpu_buffer, int plane,
                           GlTextureView::DoneWritingFn done_writing) const;
  void ViewDoneWriting(const GlTextureView& view);
};

namespace internal {
// These functions enable backward-compatible construction of a GpuBuffer from
// CVPixelBufferRef without having to expose that type in the main GpuBuffer
// header.
std::shared_ptr<internal::GpuBufferStorage> AsGpuBufferStorage(
    CFHolder<CVPixelBufferRef> pixel_buffer);
std::shared_ptr<internal::GpuBufferStorage> AsGpuBufferStorage(
    CVPixelBufferRef pixel_buffer);
}  // namespace internal

}  // namespace mediapipe

#endif  // MEDIAPIPE_GPU_GPU_BUFFER_STORAGE_CV_PIXEL_BUFFER_H_
