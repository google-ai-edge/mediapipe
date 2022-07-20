//
//  CVFramebuffer.cpp
//  Quaramera
//
//  Created by wangrenzhu on 2021/4/30.
//
#include "CVFramebuffer.hpp"
#include <assert.h>
#include <algorithm>
#include "Context.hpp"
#include "GPUImageUtil.h"
#include "GPUImageMacros.h"
#include <Foundation/Foundation.h>
#include <OpenGLES/EAGLIOSurface.h>
#include <CoreVideo/CoreVideo.h>
#include <CoreFoundation/CoreFoundation.h>

NS_GI_BEGIN

CVFramebuffer::CVFramebuffer(Context *context,
                             int width,
                             int height,
                             const TextureAttributes textureAttributes,
                             GLuint textureId) : Opipe::Framebuffer()
{
    _context = context;
    useTextureCache = true;
    _width = width;
    _height = height;
    _textureAttributes = textureAttributes;
    _texture = textureId;
    _generateFramebuffer(false);
    _context->_framebuffers.push_back(this);
}

CVFramebuffer::CVFramebuffer(Context *context,
                             int width,
                             int height,
                             bool onlyGenerateTexture/* = false*/,
                             const TextureAttributes textureAttributes) : Opipe::Framebuffer()
{
    _context = context;
    useTextureCache = true;
    _width = width;
    _height = height;
    _textureAttributes = textureAttributes;
    _hasFB = !onlyGenerateTexture;
    if (_hasFB) {
        _generateFramebuffer();
    } else {
        _generateTexture();
    }
    
    _context->_framebuffers.push_back(this);
}

CVFramebuffer::CVFramebuffer(Context *context,
                             int width, int height,
                             GLuint handle, IOSurfaceID surfaceID,
                             const TextureAttributes textureAttributes) : Opipe::Framebuffer() {
    _context = context;
    useTextureCache = true;
    _width = width;
    _height = height;
    _texture = handle;
    _textureAttributes = textureAttributes;
    _ioSurfaceId = surfaceID;
    if (@available(iOS 11.0, *)) {
        renderIOSurface = IOSurfaceLookup(surfaceID); //可能为空
        if (renderIOSurface) {
            NSDictionary *cvBufferProperties = @{(id)kCVPixelBufferIOSurfacePropertiesKey : @{},
                                                 (id)kCVPixelBufferIOSurfaceOpenGLESTextureCompatibilityKey: @(YES),
                                                 (id)kCVPixelBufferOpenGLCompatibilityKey : @(YES),
            };
            
            CVPixelBufferCreateWithIOSurface(kCFAllocatorDefault, renderIOSurface,
                                             (__bridge CFDictionaryRef)cvBufferProperties, &renderTarget);
            GLenum internalFormat = textureAttributes.internalFormat;
            GLenum extformat = GL_BGRA_EXT;
            
            if (internalFormat == GL_LUMINANCE) {
                extformat = GL_R16F_EXT;
            }
            
    #if !TARGET_OS_SIMULATOR
            CHECK_GL(glBindTexture(GL_TEXTURE_2D, _texture));
            EAGLContext *currentContext = this->getContext()->getEglContext();
            [EAGLContext setCurrentContext:currentContext];
            BOOL rs = [currentContext texImageIOSurface:renderIOSurface target:GL_TEXTURE_2D
                                         internalFormat:internalFormat
                                                  width:_width height:_height
                                                 format:extformat
                                                   type:_textureAttributes.type plane:0];
            
            if (rs) {
                LogE("CVFramebuffer", "IOSurface binding 成功");
            }
    #endif
        }

    } else {
        
        assert(0);
    }
    
}

void CVFramebuffer::SetRenderTarget(CVPixelBufferRef pixel_buffer) {
    if (renderTarget) {
        CVPixelBufferRelease(renderTarget);
    }
    if (_glTexture) {
        CFRelease(_glTexture);
    }
    renderTarget = CVPixelBufferRetain(pixel_buffer);
    assert(_width == (int)CVPixelBufferGetWidth(renderTarget));
    assert(_height == (int)CVPixelBufferGetHeight(renderTarget));
    
    CVOpenGLESTextureCacheCreateTextureFromImage(kCFAllocatorDefault,
                                                 _context->iOSGLTextureCache,
                                                 renderTarget, NULL,
                                                 GL_TEXTURE_2D, GL_RGBA,
                                                 (GLsizei)_width,
                                                 (GLsizei)_height,
                                                 GL_BGRA, GL_UNSIGNED_BYTE,
                                                 0, &_glTexture);
    _texture = CVOpenGLESTextureGetName(_glTexture);

    _bindFramebuffer();
}

CVFramebuffer::~CVFramebuffer()
{
    Log("", "Delete Framebuffer");
    if (NULL != _glTexture) {
        CFRelease(_glTexture);
    }
    
    if (NULL != renderTarget) {
        CVPixelBufferUnlockBaseAddress(renderTarget, 0);
        CVPixelBufferRelease(renderTarget);
    }
    
    if (NULL != renderIOSurface) {
        if (@available(iOS 11.0, *)) {
            IOSurfaceDecrementUseCount(renderIOSurface);
            CFRelease(renderIOSurface);
        }
    }
    
    bool bDeleteTex = (_texture != -1);
    bool bDeleteFB = (_framebuffer != -1);
    
    for (auto const &framebuffer : _context->_framebuffers) {
        if (!framebuffer || framebuffer == this) continue;
        if (bDeleteTex) {
            if (_texture == framebuffer->getTexture()) {
                bDeleteTex = false;
            }
        }
        
        if (bDeleteFB) {
            if (framebuffer->hasFramebuffer() &&
                _framebuffer == framebuffer->getFramebuffer()) {
                bDeleteFB = false;
            }
        }
    }
    
    std::vector<Framebuffer*>::iterator itr = std::find(_context->_framebuffers.begin(),  _context->_framebuffers.end(), this);
    if (itr != _context->_framebuffers.end()) {
        _context->_framebuffers.erase(itr);
    }
    
    if (bDeleteTex) {
        CHECK_GL(glDeleteTextures(1, &_texture));
        _texture = -1;
    }
    if (bDeleteFB) {
        CHECK_GL(glDeleteFramebuffers(1, &_framebuffer));
        _framebuffer = -1;
    }
}

void CVFramebuffer::lockAddress()
{
    if (renderTarget != NULL) {
        CVPixelBufferLockBaseAddress(renderTarget, kCVPixelBufferLock_ReadOnly);
    }
}

void CVFramebuffer::unlockAddress()
{
    if (renderTarget != NULL) {
        CVPixelBufferUnlockBaseAddress(renderTarget, kCVPixelBufferLock_ReadOnly);
    }
}

int CVFramebuffer::getBytesPerRow()
{
    if (renderTarget != NULL) {
        return (int)CVPixelBufferGetBytesPerRow(renderTarget);
    } else {
        return 0;
    }
}

void* CVFramebuffer::frameBufferGetBaseAddress()
{
    if (renderTarget != NULL) {
        return CVPixelBufferGetBaseAddress(renderTarget);
    }
    return NULL;
}

void CVFramebuffer::_generateTexture()
{
  
    if (@available(iOS 11.0, *)) {
        CHECK_GL(glGenTextures(1, &_texture));
        CHECK_GL(glBindTexture(GL_TEXTURE_2D, _texture));
        CHECK_GL(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER,
                                 _textureAttributes.minFilter));
        CHECK_GL(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER,
                                 _textureAttributes.magFilter));
        CHECK_GL(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, _textureAttributes.wrapS));
        CHECK_GL(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, _textureAttributes.wrapT));
        
    } else {
        
        NSDictionary *cvBufferProperties = @{(id)kCVPixelBufferIOSurfacePropertiesKey : @{},
                                             (id)kCVPixelBufferIOSurfaceOpenGLESTextureCompatibilityKey: @(YES),
                                             (id)kCVPixelBufferOpenGLCompatibilityKey : @(YES),
        };
        __unused CVReturn cvret = CVPixelBufferCreate(kCFAllocatorDefault,
                                                      _width,
                                                      _height,
                                                      kCVPixelFormatType_32BGRA,
                                                      (__bridge  CFDictionaryRef)cvBufferProperties,
                                                      &renderTarget);
        
        if (cvret != kCVReturnSuccess) {
            Log("", "Failed to create CVPixelBuffer");
        }
        
        cvret = CVOpenGLESTextureCacheCreateTextureFromImage(kCFAllocatorDefault,
                                                             _context->iOSGLTextureCache,
                                                             renderTarget,
                                                             nil,
                                                             GL_TEXTURE_2D,
                                                             GL_RGBA,
                                                             _width, _height,
                                                             GL_BGRA,
                                                             GL_UNSIGNED_BYTE,
                                                             0,
                                                             &_glTexture);
        if (cvret != kCVReturnSuccess) {
            Log("", "Failed to create _glTexture");
        }
        
        _texture = CVOpenGLESTextureGetName(_glTexture);
        
        CHECK_GL(glBindTexture(GL_TEXTURE_2D, _texture));
        CHECK_GL(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER,
                                 _textureAttributes.minFilter));
        CHECK_GL(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER,
                                 _textureAttributes.magFilter));
        CHECK_GL(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, _textureAttributes.wrapS));
        CHECK_GL(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, _textureAttributes.wrapT));
        CHECK_GL(glBindTexture(GL_TEXTURE_2D, 0));
    }
}

void CVFramebuffer::_generateFramebuffer(bool needGenerateTexture)
{
    
    
    CHECK_GL(glGenFramebuffers(1, &_framebuffer));
    
    if (needGenerateTexture) {
        _generateTexture();
    }
    
    _bindFramebuffer();
    
}


void CVFramebuffer::_bindFramebuffer() {
    if (@available(iOS 11.0, *)) {
        NSDictionary *cvBufferProperties = @{(id)kCVPixelBufferIOSurfacePropertiesKey : @{},
                                             (id)kCVPixelBufferIOSurfaceOpenGLESTextureCompatibilityKey: @(YES),
                                             (id)kCVPixelBufferOpenGLCompatibilityKey : @(YES),
        };
        CHECK_GL(glBindTexture(GL_TEXTURE_2D, _texture));
        EAGLContext *currentContext = this->getContext()->getEglContext();
        
        unsigned bytesPerElement = 4;
        
        size_t bytesPerRow = IOSurfaceAlignProperty(kIOSurfaceBytesPerRow, _width * bytesPerElement);
        size_t totalBytes = IOSurfaceAlignProperty(kIOSurfaceAllocSize,  _height * bytesPerRow);
        id cvformat = @(kCVPixelFormatType_32BGRA);
        
        if (_textureAttributes.format == GL_LUMINANCE) {
            cvformat = @(kCVPixelFormatType_16Gray);
        }
        
        NSDictionary *dict = @{
            (id)kIOSurfaceWidth : @(_width),
            (id)kIOSurfaceHeight : @(_height),
            (id)kIOSurfacePixelFormat : cvformat,
            (id)kIOSurfaceBytesPerElement : @(bytesPerElement),
            (id)kIOSurfaceBytesPerRow : @(bytesPerRow),
            (id)kIOSurfaceAllocSize : @(totalBytes),
            (id)kIOSurfaceIsGlobal: @YES
        };
        
        renderIOSurface = IOSurfaceCreate((CFDictionaryRef)dict);
        _ioSurfaceId = IOSurfaceGetID(renderIOSurface);
        
        GLenum internalFormat = _textureAttributes.internalFormat;
        GLenum extformat = GL_BGRA_EXT;
        
        if (internalFormat == GL_LUMINANCE) {
            extformat = GL_R16F_EXT;
        }
        
        CVPixelBufferCreateWithIOSurface(kCFAllocatorDefault, renderIOSurface,
                                         (__bridge CFDictionaryRef)cvBufferProperties, &renderTarget);
#if !TARGET_OS_SIMULATOR
        BOOL rs = [currentContext texImageIOSurface:renderIOSurface target:GL_TEXTURE_2D
                                     internalFormat:internalFormat
                                              width:_width height:_height
                                             format:extformat
                                               type:_textureAttributes.type plane:0];
        IOSurfaceIncrementUseCount(renderIOSurface);
        if (rs) {
            LogE("CVFramebuffer", "IOSurface binding 成功");
        }
#endif
    }
    CHECK_GL(glBindTexture(GL_TEXTURE_2D, 0));
    
    CHECK_GL(glBindFramebuffer(GL_FRAMEBUFFER, _framebuffer));
    CHECK_GL(glBindTexture(GL_TEXTURE_2D, _texture));
    
    CHECK_GL(glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D,
                                    _texture, 0));
    
    GLenum status = glCheckFramebufferStatus(GL_FRAMEBUFFER);
    assert(status == GL_FRAMEBUFFER_COMPLETE);
    
    CHECK_GL(glBindTexture(GL_TEXTURE_2D, 0));
    CHECK_GL(glBindFramebuffer(GL_FRAMEBUFFER, 0));
    //    Opipe::Log("Quaramera", "_generateFramebuffer %d ", _framebuffer);
    assert(_framebuffer < 100);
}


NS_GI_END
