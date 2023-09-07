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

#include "mediapipe/gpu/shader_util.h"

#include <stdlib.h>

#include <cmath>
#include <string>
#include <vector>

#include "absl/log/absl_check.h"
#include "absl/log/absl_log.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "absl/strings/str_split.h"
#include "mediapipe/framework/port/logging.h"

#if DEBUG
#define GL_DEBUG_LOG(type, object, action)                        \
  do {                                                            \
    GLint log_length = 0;                                         \
    glGet##type##iv(object, GL_INFO_LOG_LENGTH, &log_length);     \
    if (log_length > 0) {                                         \
      GLchar* log = static_cast<GLchar*>(malloc(log_length));     \
      glGet##type##InfoLog(object, log_length, &log_length, log); \
      ABSL_LOG(INFO) << #type " " action " log:\n" << log;        \
      free(log);                                                  \
    }                                                             \
  } while (0)
#else
#define GL_DEBUG_LOG(type, object, action)
#endif

#define GL_ERROR_LOG(type, object, action)                        \
  do {                                                            \
    GLint log_length = 0;                                         \
    glGet##type##iv(object, GL_INFO_LOG_LENGTH, &log_length);     \
    if (log_length > 0) {                                         \
      GLchar* log = static_cast<GLchar*>(malloc(log_length));     \
      glGet##type##InfoLog(object, log_length, &log_length, log); \
      ABSL_LOG(ERROR) << #type " " action " log:\n" << log;       \
      free(log);                                                  \
    }                                                             \
  } while (0)

namespace mediapipe {
namespace {

constexpr int kMaxShaderInfoLength = 1024;

std::string AddLineNumbers(const GLchar* source) {
  // Use format "%ni %s", with n=1 for 1..9 lines, n=2 for 10..99 lines etc.
  // Note that StrFormat needs either a constexpr format or a ParsedFormat.
  std::vector<std::string> lines = absl::StrSplit(source, '\n');
  std::string format = absl::StrFormat(
      "%%%ii %%s", static_cast<int>(ceilf(log10(1 + lines.size()))));
  auto parsed_format = absl::ParsedFormat<'i', 's'>::New(format);
  ABSL_CHECK(parsed_format);
  for (int n = 0; n < lines.size(); n++) {
    lines[n] = absl::StrFormat(*parsed_format, n + 1, lines[n]);
  }
  return absl::StrJoin(lines, "\n");
}

}  // namespace

GLint GlhCompileShader(GLenum target, const GLchar* source, GLuint* shader,
                       bool force_log_errors) {
  *shader = glCreateShader(target);
  if (*shader == 0) {
    return GL_FALSE;
  }
  glShaderSource(*shader, 1, &source, NULL);
  glCompileShader(*shader);

  GL_DEBUG_LOG(Shader, *shader, "compile");

#if UNSAFE_EMSCRIPTEN_SKIP_GL_ERROR_HANDLING
  if (!force_log_errors) {
    return GL_TRUE;
  }
#endif  // UNSAFE_EMSCRIPTEN_SKIP_GL_ERROR_HANDLING

  GLint status;

  glGetShaderiv(*shader, GL_COMPILE_STATUS, &status);
  ABSL_LOG_IF(ERROR, status == GL_FALSE) << "Failed to compile shader:\n"
                                         << AddLineNumbers(source);

  if (status == GL_FALSE) {
    int length = 0;
    GLchar cmessage[kMaxShaderInfoLength];
    glGetShaderInfoLog(*shader, kMaxShaderInfoLength, &length, cmessage);
    ABSL_LOG(ERROR) << "Error message: " << std::string(cmessage, length);
  }
  return status;
}

GLint GlhLinkProgram(GLuint program, bool force_log_errors) {
  glLinkProgram(program);

#if UNSAFE_EMSCRIPTEN_SKIP_GL_ERROR_HANDLING
  if (!force_log_errors) {
    return GL_TRUE;
  }
#endif  // UNSAFE_EMSCRIPTEN_SKIP_GL_ERROR_HANDLING

  GLint status;

  GL_DEBUG_LOG(Program, program, "link");

  glGetProgramiv(program, GL_LINK_STATUS, &status);
  ABSL_LOG_IF(ERROR, status == GL_FALSE)
      << "Failed to link program " << program;

  return status;
}

GLint GlhValidateProgram(GLuint program) {
  GLint status;

  glValidateProgram(program);

  GL_DEBUG_LOG(Program, program, "validate");

  glGetProgramiv(program, GL_VALIDATE_STATUS, &status);
  ABSL_LOG_IF(ERROR, status == GL_FALSE)
      << "Failed to validate program " << program;

  return status;
}

GLint GlhCreateProgram(const GLchar* vert_src, const GLchar* frag_src,
                       GLsizei attr_count, const GLchar* const* attr_names,
                       const GLint* attr_locations, GLuint* program,
                       bool force_log_errors) {
  GLuint vert_shader = 0;
  GLuint frag_shader = 0;
  GLint ok = GL_TRUE;

  *program = glCreateProgram();
  if (*program == 0) {
    return GL_FALSE;
  }

  ok = ok && GlhCompileShader(GL_VERTEX_SHADER, vert_src, &vert_shader,
                              force_log_errors);
  ok = ok && GlhCompileShader(GL_FRAGMENT_SHADER, frag_src, &frag_shader,
                              force_log_errors);

  if (ok) {
    glAttachShader(*program, vert_shader);
    glAttachShader(*program, frag_shader);

    // Attribute location binding must be set before linking.
    for (int i = 0; i < attr_count; i++) {
      glBindAttribLocation(*program, attr_locations[i], attr_names[i]);
    }

    ok = GlhLinkProgram(*program, force_log_errors);

    glDetachShader(*program, frag_shader);
    glDetachShader(*program, vert_shader);
  }

  if (vert_shader) glDeleteShader(vert_shader);
  if (frag_shader) glDeleteShader(frag_shader);

  if (!ok) {
    glDeleteProgram(*program);
    *program = 0;
  }

  return ok;
}

bool CompileShader(GLenum shader_type, const std::string& shader_source,
                   GLuint* shader) {
  *shader = glCreateShader(shader_type);
  if (*shader == 0) {
    VLOG(2) << "Unable to create shader of type: " << shader_type;
    return false;
  }
  const char* shader_source_cstr = shader_source.c_str();
  glShaderSource(*shader, 1, &shader_source_cstr, NULL);
  glCompileShader(*shader);

  GLint compiled;
  glGetShaderiv(*shader, GL_COMPILE_STATUS, &compiled);
  if (!compiled) {
    VLOG(2) << "Unable to compile shader:\n"
            << AddLineNumbers(shader_source_cstr);
    GL_ERROR_LOG(Shader, *shader, "compile");
    glDeleteShader(*shader);
    *shader = 0;
    return false;
  }
  return true;
}

bool CreateShaderProgram(
    GLuint vertex_shader, GLuint fragment_shader,
    const std::unordered_map<GLuint, std::string>& attributes,
    GLuint* shader_program) {
  *shader_program = glCreateProgram();
  if (*shader_program == 0) {
    VLOG(2) << "Unable to create shader program";
    return false;
  }
  glAttachShader(*shader_program, vertex_shader);
  glAttachShader(*shader_program, fragment_shader);

  for (const auto& it : attributes) {
    glBindAttribLocation(*shader_program, it.first, it.second.c_str());
  }
  glLinkProgram(*shader_program);

  GLint is_linked = 0;
  glGetProgramiv(*shader_program, GL_LINK_STATUS, &is_linked);
  if (!is_linked) {
    glDeleteProgram(*shader_program);
    *shader_program = 0;
    return false;
  }
  return true;
}

}  // namespace mediapipe
