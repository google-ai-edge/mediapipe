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

#include "Framebuffer.hpp"
#include <assert.h>
#include <algorithm>
#include "GPUImageUtil.h"
#include "Context.hpp"
#include "GPUImageMacros.h"


#define useCVPB defined(__APPLE__)


namespace Opipe {
    TextureAttributes Framebuffer::defaultTextureAttribures =
    {
    .minFilter = GL_LINEAR,
    .magFilter = GL_LINEAR,
    .wrapS = GL_CLAMP_TO_EDGE,
    .wrapT = GL_CLAMP_TO_EDGE,
    .internalFormat = GL_RGBA,
    .format = GL_RGBA,
    .type = GL_UNSIGNED_BYTE
    };

    Framebuffer::Framebuffer() {
        
    }

    Framebuffer::Framebuffer(Context *context, int width, int height,
                             const TextureAttributes textureAttributes, GLuint textureId)
    : _texture(-1), _hasFB(true), _framebuffer(-1), _context(context) {
        
        
        _width = width;
        _height = height;
        _textureAttributes = textureAttributes;
        _texture = textureId;
        _useExternalTexture = true;
        _generateFramebuffer(false);
        _context->_framebuffers.push_back(this);
    }

    Framebuffer::Framebuffer(Context *context, int width, int height,
                             bool onlyGenerateTexture/* = false*/,
                             const TextureAttributes textureAttributes) : _texture(-1),
    _framebuffer(-1),
    _context(context) {
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

    Framebuffer::Framebuffer(Context *context, int width, int height, GLuint handle,
                             const TextureAttributes textureAttributes) : _texture(handle),
    _framebuffer(-1),
    _context(context) {
        _width = width;
        _height = height;
        _textureAttributes = textureAttributes;
    }

    Framebuffer::~Framebuffer() {
        if (isDealloc) {
            return;
        }
        Opipe::Log("Framebuffer", "delete Framebuffer(%d,%d) ", _width, _height);
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
        
        if (bDeleteTex && !_useExternalTexture) {
            CHECK_GL(glDeleteTextures(1, &_texture));
            _texture = -1;
        }
        if (bDeleteFB) {
            CHECK_GL(glDeleteFramebuffers(1, &_framebuffer));
            _framebuffer = -1;
        }
        isDealloc = true;
        
    }

    void Framebuffer::active() {
        CHECK_GL(glBindFramebuffer(GL_FRAMEBUFFER, _framebuffer));
        CHECK_GL(glViewport(0, 0, _width, _height));
    }

    void Framebuffer::inactive() {
        CHECK_GL(glBindFramebuffer(GL_FRAMEBUFFER, 0));
    }

    void Framebuffer::lock(std::string lockKey) {
        if (lockKey == "Unknow") {
//            Log("Framebuffer LOCK", "未知锁 【hasCode :%s】", _hashCode.c_str());
        } else if (lockKey != _lockKey) {
//            Log("Framebuffer LOCK", "Key变更:%s 【hasCode :%s】", lockKey.c_str(), _hashCode.c_str());
        }
        
        _lockKey = lockKey;
        _framebufferRetainCount++;
//        Log("Framebuffer LOCK", "lock retainCount == :%d lockKey:%s 【framebufferCode:%s】",
//            _framebufferRetainCount,
//            lockKey.c_str(), _hashCode.c_str());
    }

    void Framebuffer::unlock(std::string lockKey) {
        if (_framebufferRetainCount > 0) {
            _framebufferRetainCount--;
        } else {
    //            assert("过度释放 请检查"); 此处不要崩溃，引用计数管理Framebuffer不会导致过度释放。
        }
        
        if (lockKey != _lockKey) {
//            Log("Framebuffer UNLOCK", "可能是多次Lock后Unlock retainCount:%d lockKey:%s 【framebufferCode:%s】",
//                _framebufferRetainCount,
//                lockKey.c_str(),
//                _hashCode.c_str());
        }
        
//        Log("Framebuffer UNLOCK", "unlock retainCount == :%d lockKey:%s 【framebufferCode:%s】"
//            , _framebufferRetainCount,
//            lockKey.c_str(),
//            _hashCode.c_str());
    }

    void Framebuffer::resetRetainCount() {
        _framebufferRetainCount = 0;
    }

    void *Framebuffer::frameBufferGetBaseAddress() {
        //#if HARDWARE_BUFFER_ENABLE
        //        return _hardwareBufferReadData;
        //#endif
        return NULL;
    }

    int Framebuffer::getBytesPerRow() {
        return _width * 4;
    }

    //#if PLATFORM == PLATFORM_ANDROID
    //    AHardwareBuffer_Desc& Framebuffer::getAHardwareBufferDesc(){
    //        return _graphicBufDes;
    //    }
    //#endif

    void Framebuffer::_generateTexture() {
        CHECK_GL(glGenTextures(1, &_texture));
        
        CHECK_GL(glBindTexture(GL_TEXTURE_2D, _texture));
        CHECK_GL(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER,
                                 _textureAttributes.minFilter));
        CHECK_GL(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER,
                                 _textureAttributes.magFilter));
        CHECK_GL(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, _textureAttributes.wrapS));
        CHECK_GL(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, _textureAttributes.wrapT));
        
        // TODO: Handle mipmaps
        CHECK_GL(glBindTexture(GL_TEXTURE_2D, 0));
    }


    void Framebuffer::_generateFramebuffer(bool needGenerateTexture) {
        
        CHECK_GL(glGenFramebuffers(1, &_framebuffer));
        CHECK_GL(glBindFramebuffer(GL_FRAMEBUFFER, _framebuffer));
        
        if (needGenerateTexture) {
            _generateTexture();
        }
        
        CHECK_GL(glBindTexture(GL_TEXTURE_2D, _texture));
        
        if (needGenerateTexture) {
            
            CHECK_GL(glTexImage2D(GL_TEXTURE_2D, 0, _textureAttributes.internalFormat, _width,
                                  _height, 0, _textureAttributes.format, _textureAttributes.type,
                                  0));
        }
        
        CHECK_GL(glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D,
                                        _texture, 0));
        
        
        CHECK_GL(glBindTexture(GL_TEXTURE_2D, 0));
        CHECK_GL(glBindFramebuffer(GL_FRAMEBUFFER, 0));
    //        Opipe::Log("QuarameraGL", "_generateFramebuffer %d ", _framebuffer);
    }

    Context *Framebuffer::getContext() {
        if (_context) {
            return _context;
        }
        
        return NULL;
    }
}
