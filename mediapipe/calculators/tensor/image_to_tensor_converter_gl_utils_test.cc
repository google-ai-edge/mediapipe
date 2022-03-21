#include "mediapipe/framework/port.h"

#if MEDIAPIPE_OPENGL_ES_VERSION >= MEDIAPIPE_OPENGL_ES_30

#include "mediapipe/calculators/tensor/image_to_tensor_converter_gl_utils.h"
#include "mediapipe/framework/port/gtest.h"
#include "mediapipe/framework/port/status_matchers.h"
#include "mediapipe/gpu/gl_base.h"
#include "mediapipe/gpu/gl_context.h"

namespace mediapipe {
namespace {

TEST(ImageToTensorConverterGlUtilsTest, GlTexParameteriOverrider) {
  auto status_or_context = mediapipe::GlContext::Create(nullptr, false);
  MP_ASSERT_OK(status_or_context);
  auto context = status_or_context.value();

  std::vector<GLint> min_filter_changes;
  context->Run([&min_filter_changes]() {
    GLuint texture = 0;
    glGenTextures(1, &texture);
    glBindTexture(GL_TEXTURE_2D, texture);

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    GLint value = 0;
    glGetTexParameteriv(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, &value);
    min_filter_changes.push_back(value);

    {
      auto min_filter_linear =
          OverrideGlTexParametri(GL_TEXTURE_MIN_FILTER, GL_LINEAR);
      glGetTexParameteriv(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, &value);
      min_filter_changes.push_back(value);

      // reverter is destroyed automatically reverting previously set value
    }
    glGetTexParameteriv(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, &value);
    min_filter_changes.push_back(value);
  });

  EXPECT_THAT(min_filter_changes,
              testing::ElementsAre(GL_NEAREST, GL_LINEAR, GL_NEAREST));
}

}  // namespace
}  // namespace mediapipe

#endif  // MEDIAPIPE_OPENGL_ES_VERSION >= MEDIAPIPE_OPENGL_ES_30
