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

package com.google.mediapipe.glutil;

import android.graphics.Bitmap;
import android.opengl.GLES20;
import android.opengl.GLUtils;
import com.google.common.flogger.FluentLogger;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;
import java.util.Map;
import javax.annotation.Nullable;

/**
 * Utility class for managing GLSL shaders.
 */
public class ShaderUtil {
  private static final FluentLogger logger = FluentLogger.forEnclosingClass();

  /**
   * Loads a shader from source.
   * @param shaderType a valid GL shader type, e.g. {@link GLES20#GL_VERTEX_SHADER} or
   *        {@link GLES20#GL_FRAGMENT_SHADER}.
   * @param source the shader's source in text form.
   * @return a handle to the created shader, or 0 in case of error.
   */
  public static int loadShader(int shaderType, String source) {
    int shader = GLES20.glCreateShader(shaderType);
    GLES20.glShaderSource(shader, source);
    GLES20.glCompileShader(shader);
    int[] compiled = new int[1];
    GLES20.glGetShaderiv(shader, GLES20.GL_COMPILE_STATUS, compiled, 0);
    if (compiled[0] == 0) {
      logger.atSevere().log("Could not compile shader %d: %s", shaderType,
          GLES20.glGetShaderInfoLog(shader));
      GLES20.glDeleteShader(shader);
      shader = 0;
    }
    return shader;
  }

  /**
   * Creates a shader program.
   *
   * @param vertexSource source of the vertex shader.
   * @param fragmentSource source of the fragment shader.
   * @param attributeLocations a map of desired locations for attributes. Can be null.
   * @return a handle to the created program, or 0 in case of error.
   */
  public static int createProgram(
      String vertexSource,
      String fragmentSource,
      @Nullable Map<String, Integer> attributeLocations) {
    int vertexShader = loadShader(GLES20.GL_VERTEX_SHADER, vertexSource);
    if (vertexShader == 0) {
      return 0;
    }
    int fragmentShader = loadShader(GLES20.GL_FRAGMENT_SHADER, fragmentSource);
    if (fragmentShader == 0) {
      return 0;
    }

    int program = GLES20.glCreateProgram();
    if (program == 0) {
      logger.atSevere().log("Could not create program");
    }
    GLES20.glAttachShader(program, vertexShader);
    GLES20.glAttachShader(program, fragmentShader);

    if (attributeLocations != null) {
      for (Map.Entry<String, Integer> entry : attributeLocations.entrySet()) {
        GLES20.glBindAttribLocation(program, entry.getValue(), entry.getKey());
      }
    }

    GLES20.glLinkProgram(program);
    int[] linkStatus = new int[1];
    GLES20.glGetProgramiv(program, GLES20.GL_LINK_STATUS, linkStatus, 0);
    if (linkStatus[0] != GLES20.GL_TRUE) {
      logger.atSevere().log("Could not link program: %s", GLES20.glGetProgramInfoLog(program));
      GLES20.glDeleteProgram(program);
      program = 0;
    }
    return program;
  }

  /**
   * Creates a texture. Binds it to texture unit 0 to perform setup.
   * @return the name of the new texture.
   */
  public static int createRgbaTexture(int width, int height) {
    final int[] textureName = new int[] {0};
    GLES20.glGenTextures(1, textureName, 0);

    GLES20.glActiveTexture(GLES20.GL_TEXTURE0);
    GLES20.glBindTexture(GLES20.GL_TEXTURE_2D, textureName[0]);
    GLES20.glTexImage2D(
        GLES20.GL_TEXTURE_2D,
        0,
        GLES20.GL_RGBA,
        width, height,
        0,
        GLES20.GL_RGBA,
        GLES20.GL_UNSIGNED_BYTE,
        null);
    ShaderUtil.checkGlError("glTexImage2D");
    GLES20.glTexParameteri(GLES20.GL_TEXTURE_2D, GLES20.GL_TEXTURE_MIN_FILTER, GLES20.GL_LINEAR);
    GLES20.glTexParameteri(GLES20.GL_TEXTURE_2D, GLES20.GL_TEXTURE_MAG_FILTER, GLES20.GL_LINEAR);
    GLES20.glTexParameteri(GLES20.GL_TEXTURE_2D, GLES20.GL_TEXTURE_WRAP_S, GLES20.GL_CLAMP_TO_EDGE);
    GLES20.glTexParameteri(GLES20.GL_TEXTURE_2D, GLES20.GL_TEXTURE_WRAP_T, GLES20.GL_CLAMP_TO_EDGE);
    ShaderUtil.checkGlError("texture setup");
    return textureName[0];
  }

  /**
   * Creates a texture from a Bitmap. Binds it to texture unit 0 to perform setup.
   *
   * @return the name of the new texture.
   */
  public static int createRgbaTexture(Bitmap bitmap) {
    final int[] textureName = new int[] {0};
    GLES20.glGenTextures(1, textureName, 0);

    GLES20.glActiveTexture(GLES20.GL_TEXTURE0);
    GLES20.glBindTexture(GLES20.GL_TEXTURE_2D, textureName[0]);
    GLUtils.texImage2D(GLES20.GL_TEXTURE_2D, 0, bitmap, 0);
    ShaderUtil.checkGlError("texImage2D");
    GLES20.glTexParameteri(GLES20.GL_TEXTURE_2D, GLES20.GL_TEXTURE_MIN_FILTER, GLES20.GL_LINEAR);
    GLES20.glTexParameteri(GLES20.GL_TEXTURE_2D, GLES20.GL_TEXTURE_MAG_FILTER, GLES20.GL_LINEAR);
    GLES20.glTexParameteri(GLES20.GL_TEXTURE_2D, GLES20.GL_TEXTURE_WRAP_S, GLES20.GL_CLAMP_TO_EDGE);
    GLES20.glTexParameteri(GLES20.GL_TEXTURE_2D, GLES20.GL_TEXTURE_WRAP_T, GLES20.GL_CLAMP_TO_EDGE);
    ShaderUtil.checkGlError("texture setup");
    return textureName[0];
  }

  /**
   * Creates a {@link FloatBuffer} with the given arguments as contents.
   * The buffer is created in native format for efficient use with OpenGL.
   */
  public static FloatBuffer floatBuffer(float... values) {
    ByteBuffer byteBuffer =
        ByteBuffer.allocateDirect(
            values.length * 4 /* sizeof(float) */);
    // use the device hardware's native byte order
    byteBuffer.order(ByteOrder.nativeOrder());

    // create a floating point buffer from the ByteBuffer
    FloatBuffer floatBuffer = byteBuffer.asFloatBuffer();
    // add the coordinates to the FloatBuffer
    floatBuffer.put(values);
    // set the buffer to read the first coordinate
    floatBuffer.position(0);
    return floatBuffer;
  }

  /**
   * Calls {@link GLES20#glGetError} and raises an exception if there was an error.
   */
  public static void checkGlError(String msg) {
    int error = GLES20.glGetError();
    if (error != GLES20.GL_NO_ERROR) {
      throw new RuntimeException(msg + ": GL error: 0x" + Integer.toHexString(error));
    }
  }
}
