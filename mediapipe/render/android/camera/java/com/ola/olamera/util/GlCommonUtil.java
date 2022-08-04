//
// Source code recreated from a .class file by IntelliJ IDEA
// (powered by Fernflower decompiler)
//

package com.ola.olamera.util;

import android.annotation.TargetApi;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.opengl.GLES20;
import android.opengl.GLUtils;


import java.nio.Buffer;

@TargetApi(18)
public class GlCommonUtil {
    public static final String TAG = "GlCommonUtil";
    public static final int NO_TEXTURE = -1;

    public GlCommonUtil() {
    }

    public static int createProgram(String vertexSource, String fragmentSource) {
        int vs = loadShader(35633, vertexSource);
        int fs = loadShader(35632, fragmentSource);
        int program = GLES20.glCreateProgram();
        GLES20.glAttachShader(program, vs);
        GLES20.glAttachShader(program, fs);
        GLES20.glLinkProgram(program);
        int[] linkStatus = new int[1];
        GLES20.glGetProgramiv(program, 35714, linkStatus, 0);
        if (linkStatus[0] != 1) {
            CameraLogger.e("GlCommonUtil", "Could not link program:");
            CameraLogger.e("GlCommonUtil", GLES20.glGetProgramInfoLog(program));
            GLES20.glDeleteProgram(program);
            program = 0;
        }

        return program;
    }

    public static int loadShader(int shaderType, String source) {
        int shader = GLES20.glCreateShader(shaderType);
        GLES20.glShaderSource(shader, source);
        GLES20.glCompileShader(shader);
        int[] compiled = new int[1];
        GLES20.glGetShaderiv(shader, 35713, compiled, 0);
        if (compiled[0] == 0) {
            CameraLogger.e("GlCommonUtil", "Could not compile shader(TYPE=" + shaderType + "):");
            CameraLogger.e("GlCommonUtil", GLES20.glGetShaderInfoLog(shader));
            GLES20.glDeleteShader(shader);
            shader = 0;
        }

        return shader;
    }

    public static void checkGlError(String op) {
        int glCode = GLES20.glGetError();
        if (glCode != 0) {
            CameraShould.fail(String.format("GLUtils::CheckGLError GL Operation %s() glError (0x%x)\n ", op, glCode));
        }
    }

    public static void checkEglError(String op) {
    }

    public static void deleteGLTexture(int iTextureID) {
        GLES20.glActiveTexture(33984);
        int[] aTextures = new int[]{iTextureID};
        GLES20.glDeleteTextures(1, aTextures, 0);
    }

    public static void deleteGLTexture(int[] textures) {
        GLES20.glDeleteTextures(textures.length, textures, 0);
    }

    public static void createFramebuffers(int[] framebuffers, int[] framebufferTextures, int[] renderBuffers, int width, int height) {
        if (null != framebuffers && null != framebufferTextures && width > 0 && height > 0 && framebuffers.length == framebufferTextures.length) {
            for (int i = 0; i < framebuffers.length; ++i) {
                GLES20.glGenFramebuffers(1, framebuffers, i);
                GLES20.glGenTextures(1, framebufferTextures, i);
                GLES20.glGenRenderbuffers(1, renderBuffers, i);
                GLES20.glBindTexture(3553, framebufferTextures[i]);
                GLES20.glTexImage2D(3553, 0, 6408, width, height, 0, 6408, 5121, (Buffer) null);
                GLES20.glTexParameterf(3553, 10240, 9729.0F);
                GLES20.glTexParameterf(3553, 10241, 9729.0F);
                GLES20.glTexParameterf(3553, 10242, 33071.0F);
                GLES20.glTexParameterf(3553, 10243, 33071.0F);
                GLES20.glBindFramebuffer(36160, framebuffers[i]);
                GLES20.glFramebufferTexture2D(36160, 36064, 3553, framebufferTextures[i], 0);
                GLES20.glBindRenderbuffer(36161, renderBuffers[i]);
                GLES20.glRenderbufferStorage(36161, 33189, width, height);
                GLES20.glFramebufferRenderbuffer(36160, 36096, 36161, renderBuffers[i]);
                GLES20.glBindTexture(3553, 0);
                GLES20.glBindFramebuffer(36160, 0);
                GLES20.glBindRenderbuffer(36161, 0);
            }

        } else {
            CameraLogger.e("GlCommonUtil", "createFramebuffers param is error!");
        }
    }

    public static void destroyFramebuffers(int[] framebuffers, int[] framebufferTextures, int[] renderBuffers) {
        if (framebuffers != null) {
            GLES20.glDeleteFramebuffers(framebuffers.length, framebuffers, 0);
            Object framebuffers1 = null;
        }

        if (framebufferTextures != null) {
            GLES20.glDeleteTextures(framebufferTextures.length, framebufferTextures, 0);
            Object framebufferTextures1 = null;
        }

        if (renderBuffers != null) {
            GLES20.glDeleteRenderbuffers(renderBuffers.length, renderBuffers, 0);
            Object renderBuffers1 = null;
        }

    }

    public static int createTexture(int width, int height) {
        if (width <= 0 | height <= 0) {
            CameraLogger.e("GlCommonUtil", "createTexture param is error!");
            return -1;
        } else {
            int[] textures = new int[1];
            GLES20.glGenTextures(1, textures, 0);
            GLES20.glBindTexture(3553, textures[0]);
            GLES20.glTexParameterf(3553, 10240, 9729.0F);
            GLES20.glTexParameterf(3553, 10241, 9729.0F);
            GLES20.glTexParameterf(3553, 10242, 33071.0F);
            GLES20.glTexParameterf(3553, 10243, 33071.0F);
            GLES20.glTexImage2D(3553, 0, 6408, width, height, 0, 6408, 5121, (Buffer) null);
            return textures[0];
        }
    }

    public static void createTextures(int[] textures, int width, int height) {
        if (null != textures && textures.length >= 1 && width >= 0 && height >= 0) {
            for (int i = 0; i < textures.length; ++i) {
                textures[i] = createTexture(width, height);
            }

        } else {
            CameraLogger.e("GlCommonUtil", "createTextures param is error!");
        }
    }

    public static int loadTexture(Bitmap img, int usedTexId, boolean recycle) {
        int[] textures = new int[1];
        if (usedTexId == -1) {
            GLES20.glGenTextures(1, textures, 0);
            GLES20.glBindTexture(3553, textures[0]);
            GLES20.glTexParameterf(3553, 10240, 9729.0F);
            GLES20.glTexParameterf(3553, 10241, 9729.0F);
            GLES20.glTexParameterf(3553, 10242, 33071.0F);
            GLES20.glTexParameterf(3553, 10243, 33071.0F);
            GLUtils.texImage2D(3553, 0, img, 0);
        } else {
            GLES20.glBindTexture(3553, usedTexId);
            GLUtils.texSubImage2D(3553, 0, 0, 0, img);
            textures[0] = usedTexId;
        }

        if (recycle) {
            img.recycle();
        }

        return textures[0];
    }

    public static int loadTexture(String imgPath, int usedTexId) {
        Bitmap img = BitmapFactory.decodeFile(imgPath);
        int[] textures = new int[1];
        if (usedTexId == -1) {
            GLES20.glGenTextures(1, textures, 0);
            GLES20.glBindTexture(3553, textures[0]);
            GLES20.glTexParameterf(3553, 10240, 9729.0F);
            GLES20.glTexParameterf(3553, 10241, 9729.0F);
            GLES20.glTexParameterf(3553, 10242, 33071.0F);
            GLES20.glTexParameterf(3553, 10243, 33071.0F);
            GLUtils.texImage2D(3553, 0, img, 0);
        } else {
            GLES20.glBindTexture(3553, usedTexId);
            GLUtils.texSubImage2D(3553, 0, 0, 0, img);
            textures[0] = usedTexId;
        }

        img.recycle();
        return textures[0];
    }
}
