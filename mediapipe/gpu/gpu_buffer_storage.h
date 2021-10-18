#ifndef MEDIAPIPE_GPU_GPU_BUFFER_STORAGE_H_
#define MEDIAPIPE_GPU_GPU_BUFFER_STORAGE_H_

#include "mediapipe/framework/formats/image_frame.h"
#include "mediapipe/gpu/gpu_buffer_format.h"

namespace mediapipe {
class GlTextureView;
class GpuBuffer;
}  // namespace mediapipe

namespace mediapipe {
namespace internal {

using mediapipe::GlTextureView;
using mediapipe::GpuBuffer;
using mediapipe::GpuBufferFormat;

class GlTextureViewManager {
 public:
  virtual ~GlTextureViewManager() = default;
  virtual GlTextureView GetGlTextureReadView(
      std::shared_ptr<GpuBuffer> gpu_buffer, int plane) const = 0;
  virtual GlTextureView GetGlTextureWriteView(
      std::shared_ptr<GpuBuffer> gpu_buffer, int plane) = 0;
  virtual void ViewDoneWriting(const GlTextureView& view) = 0;
};

class GpuBufferStorage : public GlTextureViewManager {
 public:
  virtual ~GpuBufferStorage() = default;
  virtual int width() const = 0;
  virtual int height() const = 0;
  virtual GpuBufferFormat format() const = 0;
  virtual std::unique_ptr<ImageFrame> AsImageFrame() const = 0;
};

}  // namespace internal
}  // namespace mediapipe

#endif  // MEDIAPIPE_GPU_GPU_BUFFER_STORAGE_H_
