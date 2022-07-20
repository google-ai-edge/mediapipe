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

#ifndef Context_hpp
#define Context_hpp
#include "GPUImageMacros.h"
#include "FramebufferCache.hpp"
#include "Target.hpp"
#include "GLProgram.hpp"
#include <mutex>
#include <pthread.h>


#if defined(__APPLE__)
#import <OpenGLES/EAGL.h>
#import <OpenGLES/ES3/gl.h>
#import <CoreVideo/CoreVideo.h>
#else
#include <EGL/egl.h>
#endif


NS_GI_BEGIN
class Filter;
class Context {
public:

    Context();
    ~Context();

    static void init();
    static void destroy();

    static Context* getInstance();

    FramebufferCache* getFramebufferCache() const;
    void setActiveShaderProgram(GLProgram* shaderProgram);
    void purge();
    void cleanupFramebuffers();
    
    const GLfloat *textureCoordinatesForRotation(Opipe::RotationMode rotationMode);
    
    enum ContextType
    {
        GPUImageContext = 0,
        OfflineRenderContext,
        IOContext,
    };
   
    void useAsCurrent(ContextType type = GPUImageContext, bool force = false);

#if defined(__APPLE__)
    EAGLContext* getEglContext() const { return _eglContext; };
    EAGLContext* getEglUpipeContext() const { return _eglUpipeContext; };
    void renewOfflineRenderContext();
    void presentBufferForDisplay();
#else
    EGLContext getEglContext() const { return _eglContext->context(); };
    void renewOfflineRenderContext();
    void initEGLContext(EGLContext shareContext);

    void reset();
#endif

    // 不要用这个截帧 性能极低
    bool isCapturingFrame;
    Filter* captureUpToFilter;
    unsigned char* capturedFrameData;
    int captureWidth;
    int captureHeight;
    static std::string activatedContextKey;
    
    //Filter
    GLuint vertexArray;
    GLuint elementArray[8];
    
    void releaseVBOBuffers();
    
    //Framebuffer
    std::vector<Framebuffer*> _framebuffers;
    
    //GLProgram
    std::vector<GLProgram*> _programs;
    
#if defined(__APPLE__)
    CVOpenGLESTextureCacheRef iOSGLTextureCache;
    EAGLSharegroup *shareGroup;
#endif
    
private:
    static Context* _instance;
    static std::map<std::string, Context*> _ContextCache;
    
    static std::mutex _mutex;
    FramebufferCache* _framebufferCache;
    GLProgram* _curShaderProgram;

#if defined(__ANDROID__) || defined(ANDROID)
        class EAGLContext
        {
        public:
            void useAsCurrent();
            EGLContext sharedContext;
            EGLContext context(){return _context;};
            EAGLContext(EGLContext sharedContext);
            ~EAGLContext();

        private:
            EGLContext _context;
            EGLSurface _pbuffer;
            EGLDisplay _eglDisplay;
        };
#endif

    EAGLContext* _eglContext;
    EAGLContext* _eglOfflinerenderContext;
    EAGLContext* _eglContextIO;
    EAGLContext* _eglUpipeContext;
};

NS_GI_END

#endif /* Context_hpp */
