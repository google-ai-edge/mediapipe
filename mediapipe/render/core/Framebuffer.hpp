/*
 * GPUImage-x
 *
 * Copyright (C) 2017 Yijin Wang, Yiqian Wang
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef Framebuffer_hpp
#define Framebuffer_hpp

#include "GPUImageMacros.h"
#if defined(__APPLE__)
#import <OpenGLES/ES3/gl.h>
#import <OpenGLES/ES3/glext.h>
#import <CoreVideo/CoreVideo.h>
#elif defined(__ANDROID__) || defined(ANDROID)
#if defined(__ANDROID__) || defined(ANDROID)
// for EGL calls
#define EGL_EGLEXT_PROTOTYPES
#include "EGL/egl.h"
#include "EGL/eglext.h"
#include "GLES/gl.h"
#define GL_GLEXT_PROTOTYPES
#include "GLES/glext.h"
#include "android/hardware_buffer.h"

#endif




#include <GLES3/gl3.h>
#include <GLES3/gl3ext.h>
#endif
#include <vector>
#include "Ref.hpp"


NS_GI_BEGIN

typedef struct {
    GLenum minFilter;
    GLenum magFilter;
    GLenum wrapS;
    GLenum wrapT;
    GLenum internalFormat;
    GLenum format;
    GLenum type;
} TextureAttributes;

class Context;
class Framebuffer {
public:

    /// 这里将外部的纹理传入并创建FBO，无需_generateTexture
    /// @param width 纹理宽
    /// @param height 纹理高
    Framebuffer(Context *context, int width, int height,
                const TextureAttributes textureAttributes = defaultTextureAttribures,
                GLuint textureId = -1);
    Framebuffer(Context *context, int width, int height, bool onlyGenerateTexture = false,
                const TextureAttributes textureAttributes = defaultTextureAttribures);
    Framebuffer(Context *context, int width, int height, GLuint handle,
                const TextureAttributes textureAttributes = defaultTextureAttribures);
    
    Framebuffer();
    
    virtual ~Framebuffer();

    GLuint getTexture() const {
        return _texture;
    }

    void setTexture(GLuint textureId) {
        _texture = textureId;
    }

    GLuint getFramebuffer() const {
        return _framebuffer;
    }

    int getWidth() const { return _width; }
    int getHeight() const { return _height; }
    const TextureAttributes& getTextureAttributes() const { return _textureAttributes; };
    bool hasFramebuffer() { return _hasFB; };

    void active();
    void inactive();
    
    virtual void lockAddress() {};
    virtual void unlockAddress() {};
    
    virtual void lock(std::string lockKey = "Unknow");
    virtual void unlock(std::string lockKey = "Unknow");
    virtual void resetRetainCount();
    
    int framebufferRetainCount() {
        return _framebufferRetainCount;
    }
    
    virtual void* frameBufferGetBaseAddress();
    virtual int getBytesPerRow();
    
    
    virtual void _generateTexture();
    virtual void _generateFramebuffer(bool needGenerateTexture = true);

    Context *getContext();

    GLchar *renderTargetData;

    static TextureAttributes defaultTextureAttribures;

    int _width, _height;
    TextureAttributes _textureAttributes;
    bool _hasFB;
    bool useTextureCache = false;

    GLuint _texture;
    GLuint _framebuffer = -1;
    Context *_context;
    bool isDealloc = false;
    int _framebufferRetainCount = 0;
    std::string _lockKey = "Unknow";
    std::string _hashCode = "";
    std::string _typeCode = "";
    bool _useExternalTexture = false;

};


NS_GI_END

#endif /* Framebuffer_hpp */
