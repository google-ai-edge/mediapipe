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

#include "mediapipe/gpu/gl_simple_shaders.h"

namespace mediapipe {

// This macro converts everything between its parentheses to a string.
// Using this instead of R"()" preserves C-like syntax coloring in most
// editors, which is desirable for shaders.
#if !defined(_STRINGIFY)
#define __STRINGIFY(_x) #_x
#define _STRINGIFY(_x) __STRINGIFY(_x)
#endif

// Our fragment shaders use DEFAULT_PRECISION to define the default precision
// for a type. The macro strips out the precision declaration on desktop GL,
// where it's not supported.
//
// Note: this does not use a raw string because some compilers don't handle raw
// strings inside macros correctly. It uses a macro because we want to be able
// to concatenate strings by juxtaposition. We want to concatenate strings by
// juxtaposition so we can export const char* static data containing the
// pre-expanded strings.
//
// TODO: this was written before we could rely on C++11 support.
// Consider replacing it with constexpr string concatenation, or replacing the
// static variables with functions.
#define PRECISION_COMPAT                              \
  GLES_VERSION_COMPAT                                 \
  "#ifdef GL_ES \n"                                   \
  "#define DEFAULT_PRECISION(p, t) precision p t; \n" \
  "#else \n"                                          \
  "#define DEFAULT_PRECISION(p, t) \n"                \
  "#define lowp \n"                                   \
  "#define mediump \n"                                \
  "#define highp \n"                                  \
  "#endif  // defined(GL_ES) \n"

#define VERTEX_PREAMBLE     \
  PRECISION_COMPAT          \
  "#if __VERSION__ < 130\n" \
  "#define in attribute\n"  \
  "#define out varying\n"   \
  "#endif  // __VERSION__ < 130\n"

// Note: on systems where highp precision for floats is not supported (look up
// GL_FRAGMENT_PRECISION_HIGH), we replace it with mediump.
// gl_FragColor is re-defined to 'frag_out' when using 3.20 Core (GLSL 330+).
#define FRAGMENT_PREAMBLE                                        \
  PRECISION_COMPAT                                               \
  "#if __VERSION__ < 130\n"                                      \
  "#define in varying\n"                                         \
  "#define texture texture2D\n"                                  \
  "#if defined(GL_ES) && !defined(GL_FRAGMENT_PRECISION_HIGH)\n" \
  "#define highp mediump\n"                                      \
  "#endif  // GL_ES && !GL_FRAGMENT_PRECISION_HIGH\n"            \
  "#elif __VERSION__ > 320 && !defined(GL_ES)\n"                 \
  "out vec4 frag_out; \n"                                        \
  "#define gl_FragColor frag_out\n"                              \
  "#define texture2D texture\n"                                  \
  "#endif  // __VERSION__ < 130\n"

const GLchar* const kMediaPipeVertexShaderPreamble = VERTEX_PREAMBLE;
const GLchar* const kMediaPipeFragmentShaderPreamble = FRAGMENT_PREAMBLE;

const GLchar* const kBasicVertexShader = VERTEX_PREAMBLE _STRINGIFY(
    // vertex position in clip space (-1..1)
    in vec4 position;

    // texture coordinate for each vertex in normalized texture space (0..1)
    in mediump vec4 texture_coordinate;

    // texture coordinate for fragment shader (will be interpolated)
    out mediump vec2 sample_coordinate;

    void main() {
      gl_Position = position;
      sample_coordinate = texture_coordinate.xy;
    });

const GLchar* const kScaledVertexShader = VERTEX_PREAMBLE _STRINGIFY(
    in vec4 position; in mediump vec4 texture_coordinate;
    out mediump vec2 sample_coordinate; uniform vec4 scale;

    void main() {
      gl_Position = position * scale;
      sample_coordinate = texture_coordinate.xy;
    });

const GLchar* const kTransformedVertexShader = VERTEX_PREAMBLE _STRINGIFY(
    in vec4 position; in mediump vec4 texture_coordinate;
    out mediump vec2 sample_coordinate; uniform mat3 transform;
    uniform vec2 viewport_size;

    void main() {
      // switch from clip to viewport aspect ratio in order to properly
      // apply transformation
      vec2 half_viewport_size = viewport_size * 0.5;
      vec3 pos = vec3(position.xy * half_viewport_size, 1);

      // apply transform
      pos = transform * pos;

      // switch back to clip space
      gl_Position = vec4(pos.xy / half_viewport_size, 0, 1);

      sample_coordinate = texture_coordinate.xy;
    });

const GLchar* const kBasicTexturedFragmentShader = FRAGMENT_PREAMBLE _STRINGIFY(
    DEFAULT_PRECISION(mediump, float)

        in mediump vec2 sample_coordinate;  // texture coordinate (0..1)
    uniform sampler2D video_frame;

    void main() { gl_FragColor = texture(video_frame, sample_coordinate); });

const GLchar* const kBasicTexturedFragmentShaderOES = FRAGMENT_PREAMBLE
    "#extension GL_OES_EGL_image_external : require\n" _STRINGIFY(
        DEFAULT_PRECISION(mediump, float)

            in mediump vec2 sample_coordinate;  // texture coordinate (0..1)
        uniform samplerExternalOES video_frame;

        void main() {
          gl_FragColor = texture(video_frame, sample_coordinate);
        });

const GLchar* const kFlatColorFragmentShader = FRAGMENT_PREAMBLE _STRINGIFY(
    DEFAULT_PRECISION(mediump, float)

        uniform vec3 color;  // r,g,b color components

    void main() { gl_FragColor = vec4(color.r, color.g, color.b, 1.0); });

const GLchar* const kRgbWeightFragmentShader = FRAGMENT_PREAMBLE _STRINGIFY(
    DEFAULT_PRECISION(mediump, float)

        in mediump vec2 sample_coordinate;  // texture coordinate (0..1)
    uniform sampler2D video_frame; uniform vec3 weights;  // r,g,b weights

    void main() {
      vec4 color = texture(video_frame, sample_coordinate);
      gl_FragColor.bgra = vec4(weights.z * color.b, weights.y * color.g,
                               weights.x * color.r, color.a);
    });

const GLchar* const kYUV2TexToRGBFragmentShader = FRAGMENT_PREAMBLE _STRINGIFY(
    DEFAULT_PRECISION(mediump, float)

        in highp vec2 sample_coordinate;
    uniform sampler2D video_frame_y; uniform sampler2D video_frame_uv;

    void main() {
      mediump vec3 yuv;
      lowp vec3 rgb;
      yuv.r = texture(video_frame_y, sample_coordinate).r;
      // Subtract (0.5, 0.5) because conversion is done assuming UV color
      // midpoint of (128, 128).
      yuv.gb = texture(video_frame_uv, sample_coordinate).rg - vec2(0.5, 0.5);
      // Using BT.709 which is the standard for HDTV.
      rgb = mat3(1, 1, 1, 0, -0.18732, 1.8556, 1.57481, -0.46813, 0) * yuv;
      gl_FragColor = vec4(rgb, 1);
    });

}  // namespace mediapipe
