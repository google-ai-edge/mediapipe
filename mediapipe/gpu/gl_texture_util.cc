#include "mediapipe/gpu/gl_texture_util.h"

namespace mediapipe {

void CopyGlTexture(const GlTextureView& src, GlTextureView& dst) {
  glViewport(0, 0, src.width(), src.height());
  glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, src.target(),
                         src.name(), 0);

  glActiveTexture(GL_TEXTURE0);
  glBindTexture(dst.target(), dst.name());
  glCopyTexSubImage2D(dst.target(), 0, 0, 0, 0, 0, dst.width(), dst.height());

  glBindTexture(dst.target(), 0);
  glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, src.target(), 0,
                         0);
}

void FillGlTextureRgba(GlTextureView& view, float r, float g, float b,
                       float a) {
  glViewport(0, 0, view.width(), view.height());
  glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, view.target(),
                         view.name(), 0);
  glClearColor(r, g, b, a);
  glClear(GL_COLOR_BUFFER_BIT);
  glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, view.target(), 0,
                         0);
}

}  // namespace mediapipe
