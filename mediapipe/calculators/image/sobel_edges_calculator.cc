// Copyright 2019 The MediaPipe Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "mediapipe/framework/port/ret_check.h"
#include "mediapipe/framework/port/status.h"
#include "mediapipe/gpu/gl_simple_calculator.h"
#include "mediapipe/gpu/gl_simple_shaders.h"
#include "mediapipe/gpu/shader_util.h"

enum { ATTRIB_VERTEX, ATTRIB_TEXTURE_POSITION, NUM_ATTRIBUTES };

namespace mediapipe {

// Applies the Sobel filter to an image. Expects a grayscale image stored as
// RGB, like LuminanceCalculator outputs.
// See GlSimpleCalculatorBase for inputs, outputs and input side packets.
class SobelEdgesCalculator : public GlSimpleCalculator {
 public:
  ::mediapipe::Status GlSetup() override;
  ::mediapipe::Status GlRender(const GlTexture& src,
                               const GlTexture& dst) override;
  ::mediapipe::Status GlTeardown() override;

 private:
  GLuint program_ = 0;
  GLint frame_;
  GLint pixel_w_;
  GLint pixel_h_;
};
REGISTER_CALCULATOR(SobelEdgesCalculator);

::mediapipe::Status SobelEdgesCalculator::GlSetup() {
  // Load vertex and fragment shaders
  const GLint attr_location[NUM_ATTRIBUTES] = {
      ATTRIB_VERTEX,
      ATTRIB_TEXTURE_POSITION,
  };
  const GLchar* attr_name[NUM_ATTRIBUTES] = {
      "vertexPosition",
      "vertexTextureCoordinate",
  };

  const GLchar* vert_src = GLES_VERSION_COMPAT
      R"(
#if __VERSION__ < 130
  #define in attribute
  #define out varying
#endif  // __VERSION__ < 130

    in vec4 vertexPosition;
    in vec4 vertexTextureCoordinate;

    // width of a pixel in normalized texture coordinates (0..1)
    uniform highp float pixelW;

    // height of a pixel in normalized texture coordinates (0..1)
    uniform highp float pixelH;

    // Dependent texture reads (i.e. texture reads where texture coordinates
    // are computed in the fragment shader) are slow on pre-ES 3.0 hardware.
    // Avoid them by computing all texture coordinates in the vertex shader.

    // iOS OGLES performance guide: https://developer.apple.com/library/ios/documentation/3DDrawing/Conceptual/OpenGLES_ProgrammingGuide/BestPracticesforShaders/BestPracticesforShaders.html

    // Code for coordinates: u = up, d = down, l = left, r = right, c = center.
    // Horizontal coordinate first, then vertical.
    out vec2 luTexCoord;
    out vec2 lcTexCoord;
    out vec2 ldTexCoord;

    out vec2 cuTexCoord;
//  out vec2 ccTexCoord;
    out vec2 cdTexCoord;

    out vec2 ruTexCoord;
    out vec2 rcTexCoord;
    out vec2 rdTexCoord;

    void main() {
      gl_Position = vertexPosition;

      vec2 right = vec2(pixelW, 0.0);
      vec2 up = vec2(0.0, pixelH);

      lcTexCoord = vertexTextureCoordinate.xy - right;
      luTexCoord = lcTexCoord + up;
      ldTexCoord = lcTexCoord - up;

      vec2 ccTexCoord = vertexTextureCoordinate.xy;
      cuTexCoord = ccTexCoord + up;
      cdTexCoord = ccTexCoord - up;

      rcTexCoord = vertexTextureCoordinate.xy + right;
      ruTexCoord = rcTexCoord + up;
      rdTexCoord = rcTexCoord - up;
    }
  )";
  const GLchar* frag_src = GLES_VERSION_COMPAT
      R"(
#if __VERSION__ < 130
  #define in varying
#endif  // __VERSION__ < 130

#ifdef GL_ES
  #define fragColor gl_FragColor
  precision highp float;
#else
  #define lowp
  #define mediump
  #define highp
  #define texture2D texture
  out vec4 fragColor;
#endif  // defined(GL_ES)

    in vec2 luTexCoord;
    in vec2 lcTexCoord;
    in vec2 ldTexCoord;

    in vec2 cuTexCoord;
//  in vec2 ccTexCoord;
    in vec2 cdTexCoord;

    in vec2 ruTexCoord;
    in vec2 rcTexCoord;
    in vec2 rdTexCoord;

    uniform sampler2D inputImage;

    void main() {
      float luPx = texture2D(inputImage, luTexCoord).r;
      float lcPx = texture2D(inputImage, lcTexCoord).r;
      float ldPx = texture2D(inputImage, ldTexCoord).r;

      float cuPx = texture2D(inputImage, cuTexCoord).r;
//    float ccPx = texture2D(inputImage, ccTexCoord).r;
      float cdPx = texture2D(inputImage, cdTexCoord).r;

      float ruPx = texture2D(inputImage, ruTexCoord).r;
      float rcPx = texture2D(inputImage, rcTexCoord).r;
      float rdPx = texture2D(inputImage, rdTexCoord).r;

      float h = -luPx - 2.0 * lcPx - ldPx + ruPx + 2.0 * rcPx + rdPx;
      float v = -luPx - 2.0 * cuPx - ruPx + ldPx + 2.0 * cdPx + rdPx;

      float mag = length(vec2(h, v));

      fragColor = vec4(vec3(mag), 1.0);
    }
  )";

  // shader program
  GlhCreateProgram(vert_src, frag_src, NUM_ATTRIBUTES,
                   (const GLchar**)&attr_name[0], attr_location, &program_);
  RET_CHECK(program_) << "Problem initializing the program.";
  frame_ = glGetUniformLocation(program_, "inputImage");
  pixel_w_ = glGetUniformLocation(program_, "pixelW");
  pixel_h_ = glGetUniformLocation(program_, "pixelH");
  return ::mediapipe::OkStatus();
}

::mediapipe::Status SobelEdgesCalculator::GlRender(const GlTexture& src,
                                                   const GlTexture& dst) {
  static const GLfloat square_vertices[] = {
      -1.0f, -1.0f,  // bottom left
      1.0f,  -1.0f,  // bottom right
      -1.0f, 1.0f,   // top left
      1.0f,  1.0f,   // top right
  };
  static const float texture_vertices[] = {
      0.0f, 0.0f,  // bottom left
      1.0f, 0.0f,  // bottom right
      0.0f, 1.0f,  // top left
      1.0f, 1.0f,  // top right
  };

  // program
  glUseProgram(program_);
  glUniform1i(frame_, 1);

  // parameters
  glUniform1i(frame_, 1);
  glUniform1f(pixel_w_, 1.0 / src.width());
  glUniform1f(pixel_h_, 1.0 / src.height());

  // vertex storage
  GLuint vbo[2];
  glGenBuffers(2, vbo);
  GLuint vao;
  glGenVertexArrays(1, &vao);
  glBindVertexArray(vao);

  // vbo 0
  glBindBuffer(GL_ARRAY_BUFFER, vbo[0]);
  glBufferData(GL_ARRAY_BUFFER, 4 * 2 * sizeof(GLfloat), square_vertices,
               GL_STATIC_DRAW);
  glEnableVertexAttribArray(ATTRIB_VERTEX);
  glVertexAttribPointer(ATTRIB_VERTEX, 2, GL_FLOAT, 0, 0, nullptr);

  // vbo 1
  glBindBuffer(GL_ARRAY_BUFFER, vbo[1]);
  glBufferData(GL_ARRAY_BUFFER, 4 * 2 * sizeof(GLfloat), texture_vertices,
               GL_STATIC_DRAW);
  glEnableVertexAttribArray(ATTRIB_TEXTURE_POSITION);
  glVertexAttribPointer(ATTRIB_TEXTURE_POSITION, 2, GL_FLOAT, 0, 0, nullptr);

  // draw
  glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);

  // cleanup
  glDisableVertexAttribArray(ATTRIB_VERTEX);
  glDisableVertexAttribArray(ATTRIB_TEXTURE_POSITION);
  glBindBuffer(GL_ARRAY_BUFFER, 0);
  glBindVertexArray(0);
  glDeleteVertexArrays(1, &vao);
  glDeleteBuffers(2, vbo);

  return ::mediapipe::OkStatus();
}

::mediapipe::Status SobelEdgesCalculator::GlTeardown() {
  if (program_) {
    glDeleteProgram(program_);
    program_ = 0;
  }
  return ::mediapipe::OkStatus();
}

}  // namespace mediapipe
