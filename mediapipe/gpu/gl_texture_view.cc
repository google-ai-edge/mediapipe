#include "mediapipe/gpu/gl_texture_view.h"

namespace mediapipe {

void GlTextureView::Release() {
  if (detach_) detach_(*this);
  detach_ = nullptr;
  gl_context_ = nullptr;
  gpu_buffer_ = nullptr;
  plane_ = 0;
  name_ = 0;
  width_ = 0;
  height_ = 0;
}

}  // namespace mediapipe
