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



#if defined(__APPLE__)
#import <OpenGLES/EAGLDrawable.h>
#import <OpenGLES/ES3/glext.h>

#endif
#include "Filter.hpp"
#include "Context.hpp"
#include "GPUImageUtil.h"

NS_GI_BEGIN

Context* Context::_instance = 0;
std::mutex Context::_mutex;
std::string Context::activatedContextKey = "";
std::map<std::string, Context*> Context::_ContextCache;

Context::Context(EAGLContext *context)
:_curShaderProgram(0)
,isCapturingFrame(false)
,captureUpToFilter(0)
,capturedFrameData(0)
,_eglContext(0)
,_eglOfflinerenderContext(0)
,_eglContextIO(0)
,vertexArray(-1) {
    _framebufferCache = new FramebufferCache(this);

#if defined(__APPLE__)

    _eglContext = context;
    shareGroup = [_eglContext sharegroup];
    _eglContextIO = [[EAGLContext alloc] initWithAPI:kEAGLRenderingAPIOpenGLES3 sharegroup:shareGroup];
    _eglOfflinerenderContext = [[EAGLContext alloc] initWithAPI:kEAGLRenderingAPIOpenGLES3 sharegroup:shareGroup];
    
#endif
}

Context::Context()
:_curShaderProgram(0)
,isCapturingFrame(false)
,captureUpToFilter(0)
,capturedFrameData(0)
,_eglContext(0)
,_eglOfflinerenderContext(0)
,_eglContextIO(0)
,vertexArray(-1)
{
    _framebufferCache = new FramebufferCache(this);

#if defined(__APPLE__)
    _eglContextIO = [[EAGLContext alloc] initWithAPI:kEAGLRenderingAPIOpenGLES3];
    shareGroup = [_eglContextIO sharegroup];
    _eglContext = [[EAGLContext alloc] initWithAPI:kEAGLRenderingAPIOpenGLES3 sharegroup:shareGroup];

    _eglOfflinerenderContext = [[EAGLContext alloc] initWithAPI:kEAGLRenderingAPIOpenGLES3 sharegroup:shareGroup];
    
    NSDictionary * cacheAttributes = @{ (NSString *)kCVOpenGLESTextureCacheMaximumTextureAgeKey: @(0.0) };

    __unused CVReturn cvret;
    cvret = CVOpenGLESTextureCacheCreate(kCFAllocatorDefault,
                                         (__bridge CFDictionaryRef _Nullable)(cacheAttributes),
                                         _eglContext,
                                         nil,
                                         &iOSGLTextureCache);
#endif

}

Context::~Context() {
    glFinish();
    delete _framebufferCache;
    
    #if defined(__ANDROID__) || defined(ANDROID)
    if (_eglContextIO)
    {
        delete _eglContextIO;
        _eglContextIO = 0;
    }

    if (_eglOfflinerenderContext)
    {
        delete _eglOfflinerenderContext;
        _eglOfflinerenderContext = 0;
    }

    if (_eglContext)
    {
        delete _eglContext;
        _eglContext = 0;
    }
#else
    _eglContextIO = NULL;
    _eglContext = NULL;
    _eglOfflinerenderContext = NULL;
    shareGroup = NULL;
    
#endif
    

    for (auto const &program : _programs) {
        if (program->getID() != -1) {
            glDeleteProgram(program->getID());
        }
    }


#if defined(__APPLE__)
    if (iOSGLTextureCache) {
        
        CVOpenGLESTextureCacheFlush(iOSGLTextureCache, 0);
        CFRelease(iOSGLTextureCache);
    }
#endif
    
}

Context* Context::getInstance() {
    std::unique_lock<std::mutex> lock(_mutex);
    if (!_instance)
    {
        _instance = new (std::nothrow) Context;
        
        if (!Context::activatedContextKey.empty()) {
            _ContextCache[Context::activatedContextKey] = _instance;
        }
    }
    if (_ContextCache.find(Context::activatedContextKey) != _ContextCache.end()) {
        return _ContextCache[activatedContextKey];
    } else {
        return _instance;
    }
};

void Context::init() {
    destroy();
    getInstance();
}

void Context::destroy() {
    std::unique_lock<std::mutex> lock(_mutex);
    if (!Context::activatedContextKey.empty() && _ContextCache.find(Context::activatedContextKey) != _ContextCache.end()) {
        Context *instance = _ContextCache[Context::activatedContextKey];
        _ContextCache.erase(activatedContextKey);
        delete instance;
        instance = 0;
    } else if (_instance) {
        delete _instance;
        _instance = 0;
    }
}

FramebufferCache* Context::getFramebufferCache() const {
    return _framebufferCache;
}

void Context::setActiveShaderProgram(GLProgram* shaderProgram) {
    if (_curShaderProgram != shaderProgram)
    {
        _curShaderProgram = shaderProgram;
        shaderProgram->use();
    } else if(shaderProgram){
        //double check gl current program id
        GLint cur_program_id ;
        CHECK_GL(glGetIntegerv(GL_CURRENT_PROGRAM, &cur_program_id));
        if( cur_program_id != shaderProgram->getID()){
            _curShaderProgram = shaderProgram;
            shaderProgram->use();
        }
    }

}

void Context::cleanupFramebuffers() {
    _framebufferCache->purge();
    _framebuffers.clear();
}

void Context::purge() {
//    _framebufferCache->purge();
//    _framebuffers.clear();
//
#if defined(__APPLE__)
    if (iOSGLTextureCache) {
        
        CVOpenGLESTextureCacheFlush(iOSGLTextureCache, 0);
    }
#endif
}

#if defined(__APPLE__)
void Context::useAsCurrent(ContextType type/* = GPUImageContext*/, bool force/* = false*/)
{
    if (type == IOContext)
    {
        if ([EAGLContext currentContext] != _eglContextIO || force)
        {
            [EAGLContext setCurrentContext:_eglContextIO];
        }
    }
    else if (type == OfflineRenderContext)
    {
        if ([EAGLContext currentContext] != _eglOfflinerenderContext || force)
        {
            [EAGLContext setCurrentContext:_eglOfflinerenderContext];
        }
    }
    else
    {
        if ([EAGLContext currentContext] != _eglContext || force)
        {
            [EAGLContext setCurrentContext:_eglContext];
        }
    }
}

void Context::renewOfflineRenderContext()
{
    EAGLSharegroup *group = [_eglContext sharegroup];
    _eglOfflinerenderContext = [[EAGLContext alloc] initWithAPI:kEAGLRenderingAPIOpenGLES3 sharegroup:group];
}

void Context::presentBufferForDisplay() {
    [_eglContext presentRenderbuffer:GL_RENDERBUFFER ];
}


#else
void Context::useAsCurrent(ContextType type/* = GPUImageContext*/, bool force /* = false*/)
{
    if (type == IOContext)
    {
        if (_eglContextIO != NULL && (eglGetCurrentContext() != _eglContextIO->context() || force)) {
            _eglContextIO->useAsCurrent();
        }
    }
    else if (type == OfflineRenderContext)
    {
        if (_eglOfflinerenderContext != NULL && (eglGetCurrentContext() != _eglOfflinerenderContext->context() || force)) {
            _eglOfflinerenderContext->useAsCurrent();
        }
    }
    else
    {
        //use java EGL instead
        //        _eglContext->useAsCurrent();
    }
}

void Context::renewOfflineRenderContext()
{
    if (_eglOfflinerenderContext)
    {
        EGLContext sharedContext = _eglOfflinerenderContext->sharedContext;
        delete _eglOfflinerenderContext;
        _eglOfflinerenderContext = 0;
        _eglOfflinerenderContext = new EAGLContext(sharedContext);
    }
}

void Context::reset() {
    _curShaderProgram = nullptr;
}

void Context::initEGLContext(EGLContext shareContext) {
    purge();
    if (_eglContextIO)
    {
        delete _eglContextIO;
        _eglContextIO = 0;
    }

    if (_eglOfflinerenderContext)
    {
        delete _eglOfflinerenderContext;
        _eglOfflinerenderContext = 0;
    }

    if (_eglContext)
    {
        delete _eglContext;
        _eglContext = 0;
    }
    _eglContextIO = new EAGLContext(shareContext);
    _eglOfflinerenderContext = new EAGLContext(shareContext);
//use java EGL instead
//    _eglContext = new EAGLContext(shareContext);
}

Context::EAGLContext::EAGLContext(EGLContext sharedContext) : sharedContext(sharedContext)
{
    EGLint eglConfigAttrs[] =
            {
                    EGL_DEPTH_SIZE,         24,
                    EGL_RED_SIZE,           8,
                    EGL_GREEN_SIZE,         8,
                    EGL_BLUE_SIZE,          8,
                    EGL_ALPHA_SIZE,         8,
                    EGL_STENCIL_SIZE,       8,
                    EGL_SURFACE_TYPE,       EGL_PBUFFER_BIT,
                    EGL_RENDERABLE_TYPE,    EGL_OPENGL_ES2_BIT,
                    EGL_NONE
            };
    EGLint eglContextAttrs[] =
            {
                    EGL_CONTEXT_CLIENT_VERSION,    3,
                    EGL_NONE
            };
    EGLint pbufferAttribList[] =
            {
                    EGL_WIDTH,      512,
                    EGL_HEIGHT,     512,
                    EGL_LARGEST_PBUFFER,    EGL_TRUE,
                    EGL_NONE
            };
    // egl display
    _eglDisplay = eglGetDisplay(EGL_DEFAULT_DISPLAY);

    EGLint majorVersion;
    EGLint minorVersion;
    eglInitialize(_eglDisplay,&majorVersion,&minorVersion);

    // egl config
    EGLConfig config;
    EGLint numConfigs = 0;
    eglChooseConfig(_eglDisplay,eglConfigAttrs,&config,1,&numConfigs);

    _pbuffer = eglCreatePbufferSurface(_eglDisplay,config,pbufferAttribList);

    _context = eglCreateContext(_eglDisplay,config,sharedContext,eglContextAttrs);

}

Context::EAGLContext::~EAGLContext()
{
    eglDestroySurface(_eglDisplay,_pbuffer);
    eglDestroyContext(_eglDisplay,_context);
}

void Context::EAGLContext::useAsCurrent()
{
    eglMakeCurrent(_eglDisplay, _pbuffer, _pbuffer, _context);
}

#endif

const GLfloat * Context::textureCoordinatesForRotation(Opipe::RotationMode rotationMode)
{
    static const GLfloat noRotationTextureCoordinates[] = {
            0.0f, 0.0f,
            1.0f, 0.0f,
            0.0f, 1.0f,
            1.0f, 1.0f,
        };
        
        static const GLfloat rotateLeftTextureCoordinates[] = {
            1.0f, 0.0f,
            1.0f, 1.0f,
            0.0f, 0.0f,
            0.0f, 1.0f,
        };
        
        static const GLfloat rotateRightTextureCoordinates[] = {
            0.0f, 1.0f,
            0.0f, 0.0f,
            1.0f, 1.0f,
            1.0f, 0.0f,
        };
        
        static const GLfloat verticalFlipTextureCoordinates[] = {
            0.0f, 1.0f,
            1.0f, 1.0f,
            0.0f,  0.0f,
            1.0f,  0.0f,
        };
        
        static const GLfloat horizontalFlipTextureCoordinates[] = {
            1.0f, 0.0f,
            0.0f, 0.0f,
            1.0f,  1.0f,
            0.0f,  1.0f,
        };
        
        static const GLfloat rotateRightVerticalFlipTextureCoordinates[] = {
            0.0f, 0.0f,
            0.0f, 1.0f,
            1.0f, 0.0f,
            1.0f, 1.0f,
        };

        static const GLfloat rotateRightHorizontalFlipTextureCoordinates[] = {
            1.0f, 1.0f,
            1.0f, 0.0f,
            0.0f, 1.0f,
            0.0f, 0.0f,
        };

        static const GLfloat rotate180TextureCoordinates[] = {
            1.0f, 1.0f,
            0.0f, 1.0f,
            1.0f, 0.0f,
            0.0f, 0.0f,
        };

        switch(rotationMode)
        {
            case NoRotation: return noRotationTextureCoordinates;
            case RotateLeft: return rotateLeftTextureCoordinates;
            case RotateRight: return rotateRightTextureCoordinates;
            case FlipVertical: return verticalFlipTextureCoordinates;
            case FlipHorizontal: return horizontalFlipTextureCoordinates;
            case RotateRightFlipVertical: return rotateRightVerticalFlipTextureCoordinates;
            case RotateRightFlipHorizontal: return rotateRightHorizontalFlipTextureCoordinates;
            case Rotate180: return rotate180TextureCoordinates;
        }
}

void Context::releaseVBOBuffers()
{
    if (vertexArray != -1) {
        CHECK_GL(glDeleteBuffers(1, &vertexArray));
        vertexArray = -1;
        CHECK_GL(glDeleteBuffers(8, elementArray));
        for (int i = 0; i < 8; i++) {
            elementArray[i] = -1;
        }
    }
}

NS_GI_END
