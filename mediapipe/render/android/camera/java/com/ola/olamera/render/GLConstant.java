package com.ola.olamera.render;


public class GLConstant {
    public static final String VERSION = "1.0.0";

    public static final String TAG = "CameraVideoView";

    public static final String SHADER_DEFAULT_VERTEX = "" +
            "attribute vec4 position;\n" +
            "attribute vec4 inputTextureCoordinate;\n" +
            "\n" +
            "uniform   mat4 uPosMtx;\n" +
            "uniform   mat4 uTexMtx;\n" +
            "varying   vec2 textureCoordinate;\n" +
            "void main() {\n" +
            "  gl_Position = uPosMtx * position;\n" +
            "  textureCoordinate   = (uTexMtx * inputTextureCoordinate).xy;\n" +
            "}";

    public static final String SHADER_DEFAULT_FRAGMENT_OES = "" +
//            " #version 300 es\n" +

//            "#extension GL_OES_EGL_image_external_essl3 : require\n" +
                        "#extension GL_OES_EGL_image_external : require\n" +

            "precision mediump float;\n" +
            "varying vec2 textureCoordinate;\n" +
            "uniform samplerExternalOES sTexture;\n" +
            "void main() {\n" +
            "    gl_FragColor = texture2D(sTexture, textureCoordinate);\n" +
            "}";

    public static final String SHADER_DEFAULT_FRAGMENT_NOT_OES = "" +
            "precision mediump float;\n" +
            "varying vec2 textureCoordinate;\n" +
            "uniform sampler2D sTexture;\n" +
            "void main() {\n" +
            "    gl_FragColor = texture2D(sTexture, textureCoordinate);\n" +
            "}";
}
