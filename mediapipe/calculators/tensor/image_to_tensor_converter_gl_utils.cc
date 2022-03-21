#include "mediapipe/calculators/tensor/image_to_tensor_converter_gl_utils.h"

#include "mediapipe/framework/port.h"

#if MEDIAPIPE_OPENGL_ES_VERSION >= MEDIAPIPE_OPENGL_ES_30

#include <array>
#include <memory>
#include <vector>

#include "mediapipe/framework/port/status_macros.h"
#include "mediapipe/framework/port/statusor.h"
#include "mediapipe/gpu/gl_base.h"
#include "mediapipe/gpu/gl_context.h"

namespace mediapipe {

namespace {

class GlNoOpOverride : public GlOverride {};

class GlTexParameteriOverride : public GlOverride {
 public:
  GlTexParameteriOverride(GLenum name, GLint old_value)
      : name_(name), old_value_(old_value) {}

  ~GlTexParameteriOverride() override {
    glTexParameteri(GL_TEXTURE_2D, name_, old_value_);
  }

 private:
  GLenum name_;
  GLint old_value_;
};

template <int kNumValues>
class GlTexParameterfvOverride : public GlOverride {
 public:
  GlTexParameterfvOverride(GLenum name,
                           std::array<float, kNumValues> old_values)
      : name_(name), old_values_(std::move(old_values)) {}

  ~GlTexParameterfvOverride() {
    glTexParameterfv(GL_TEXTURE_2D, name_, &old_values_[0]);
  }

 private:
  GLenum name_;
  std::array<float, kNumValues> old_values_;
};

}  // namespace

std::unique_ptr<GlOverride> OverrideGlTexParametri(GLenum name, GLint value) {
  GLint old_value;
  glGetTexParameteriv(GL_TEXTURE_2D, name, &old_value);
  if (value != old_value) {
    glTexParameteri(GL_TEXTURE_2D, name, value);
    return {absl::make_unique<GlTexParameteriOverride>(name, old_value)};
  }
  return {absl::make_unique<GlNoOpOverride>()};
}

template <int kNumValues>
std::unique_ptr<GlOverride> OverrideGlTexParameterfv(
    GLenum name, std::array<GLfloat, kNumValues> values) {
  std::array<float, kNumValues> old_values;
  glGetTexParameterfv(GL_TEXTURE_2D, name, values.data());
  if (values != old_values) {
    glTexParameterfv(GL_TEXTURE_2D, name, values.data());
    return {absl::make_unique<GlTexParameterfvOverride<kNumValues>>(
        name, std::move(old_values))};
  }
  return {absl::make_unique<GlNoOpOverride>()};
}

template std::unique_ptr<GlOverride> OverrideGlTexParameterfv<4>(
    GLenum name, std::array<GLfloat, 4> values);

bool IsGlClampToBorderSupported(const mediapipe::GlContext& gl_context) {
  return gl_context.gl_major_version() > 3 ||
         (gl_context.gl_major_version() == 3 &&
          gl_context.gl_minor_version() >= 2);
}

}  // namespace mediapipe

#endif  // MEDIAPIPE_OPENGL_ES_VERSION >= MEDIAPIPE_OPENGL_ES_30
