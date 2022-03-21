#ifndef MEDIAPIPE_GPU_GPU_BUFFER_STORAGE_IMAGE_FRAME_H_
#define MEDIAPIPE_GPU_GPU_BUFFER_STORAGE_IMAGE_FRAME_H_

#include <memory>

#include "mediapipe/framework/formats/image_frame.h"
#include "mediapipe/gpu/gpu_buffer_format.h"
#include "mediapipe/gpu/gpu_buffer_storage.h"
#include "mediapipe/gpu/image_frame_view.h"

namespace mediapipe {

// Implements support for ImageFrame as a backing storage of GpuBuffer.
class GpuBufferStorageImageFrame
    : public internal::GpuBufferStorageImpl<
          GpuBufferStorageImageFrame, internal::ViewProvider<ImageFrame>> {
 public:
  explicit GpuBufferStorageImageFrame(std::shared_ptr<ImageFrame> image_frame)
      : image_frame_(image_frame) {}
  GpuBufferStorageImageFrame(int width, int height, GpuBufferFormat format) {
    image_frame_ = std::make_shared<ImageFrame>(
        ImageFormatForGpuBufferFormat(format), width, height);
  }
  int width() const override { return image_frame_->Width(); }
  int height() const override { return image_frame_->Height(); }
  GpuBufferFormat format() const override {
    return GpuBufferFormatForImageFormat(image_frame_->Format());
  }
  std::shared_ptr<const ImageFrame> image_frame() const { return image_frame_; }
  std::shared_ptr<ImageFrame> image_frame() { return image_frame_; }
  std::shared_ptr<const ImageFrame> GetReadView(
      internal::types<ImageFrame>,
      std::shared_ptr<GpuBuffer> gpu_buffer) const override {
    return image_frame_;
  }
  std::shared_ptr<ImageFrame> GetWriteView(
      internal::types<ImageFrame>,
      std::shared_ptr<GpuBuffer> gpu_buffer) override {
    return image_frame_;
  }

 private:
  std::shared_ptr<ImageFrame> image_frame_;
};

}  // namespace mediapipe

#endif  // MEDIAPIPE_GPU_GPU_BUFFER_STORAGE_IMAGE_FRAME_H_
