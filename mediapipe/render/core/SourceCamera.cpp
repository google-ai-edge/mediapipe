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

#include <math.h>
#include "SourceCamera.hpp"
#include "Context.hpp"
#include "util.h"
#if defined(__APPLE__)
#include "CVFramebuffer.hpp"
#endif

USING_NS_GI
SourceCamera::SourceCamera(Context *context) : Source(context)
,_UVFrameBuffer(0)
,_VFrameBuffer(0)
{
    _context = context;
}

SourceCamera::~SourceCamera()
{
    removeAllTargets();
    if(_framebuffer) {
        if (_customTexture  && !_framebuffer->isDealloc) {
            delete _framebuffer;
        }
    }
    _framebuffer = 0;
    if (_UVFrameBuffer != 0) {
        _UVFrameBuffer = 0;
    }
    if (_VFrameBuffer != 0) {
        _VFrameBuffer = 0;
    }
}

SourceCamera* SourceCamera::create(Context *context)
{
    SourceCamera* sourceCamera = new SourceCamera(context);
    return sourceCamera;
}

void SourceCamera::updateTargets(float frameTime)
{
    for (auto& it : _targets){
        Target* target = it.first;
        target->setInputFramebuffer(_framebuffer, _outputRotation, _targets[target]);
        
        if (_UVFrameBuffer) {
            target->setInputFramebuffer(_UVFrameBuffer, _outputRotation, _targets[target] + 1);
        }
        
        if (_VFrameBuffer) {
            target->setInputFramebuffer(_VFrameBuffer, _outputRotation, _targets[target] + 2);
        }
        
        if (target->isPrepared()) {
            target->update(frameTime);
            target->unPrepear();
        }
    }
}

#if defined(__APPLE__)
void SourceCamera::setIORenderTexture(IOSurfaceID surfaceID,
                                      GLuint texture,
                                      int width,
                                      int height,
                                      Opipe::RotationMode outputRotation,
                                      SourceType sourceType,
                                      TextureAttributes textureAttributes) {
    //纹理发生变化，使用新的framebuffer
    if(_inputTexture != texture){
        this->setFramebuffer(nullptr);
    }

    if(_framebuffer == nullptr || (_framebuffer && _framebuffer->getTexture() != texture)) {
        if (_framebuffer) {
            delete _framebuffer;
            _framebuffer = 0;
        }
        //相机输入的FBO不要回收回去,回收回去会导致相机纹理ID不对
        /**
         * TODO：还存在坑 ,输入的纹理其实没有必要使用FBCache和FBO
         * 目前这种方式（CameraSource销毁的时候,才将FB返回FBCache）依旧存在坑
         * 因为一般Filter使用FBO，是希望能够直接绘制里面的内容，那个如果这个输入的texture的FBO给其他绘制filter使用（非Source），那么就会导致输入的texture的纹理给覆盖了
         */

        _inputTexture = texture;
        
        CVFramebuffer *framebuffer = new CVFramebuffer(_context, width, height,
                                                       texture, surfaceID, textureAttributes);
//        Framebuffer *framebuffer =  getContext()->getFramebufferCache()->fetchFramebufferUseTextureId(
//                _context, width, height, texture);
        _customTexture = true;
        this->setFramebuffer(framebuffer, outputRotation);
    }

    CHECK_GL(glBindTexture(GL_TEXTURE_2D, this->getFramebuffer()->getTexture()));
}
#endif

void SourceCamera::setRenderTexture(GLuint texture, int width, int height,
                                    RotationMode outputRotation,
                                    SourceType sourceType,
                                    TextureAttributes textureAttributes)
{

    //纹理发生变化，使用新的framebuffer
    if(_inputTexture != texture){
        this->setFramebuffer(nullptr);
    }

    if(_framebuffer == nullptr || (_framebuffer && _framebuffer->getTexture() != texture)) {
        if (_framebuffer) {
            delete _framebuffer;
            _framebuffer = 0;
        }
        //相机输入的FBO不要回收回去,回收回去会导致相机纹理ID不对
        /**
         * TODO：还存在坑 ,输入的纹理其实没有必要使用FBCache和FBO
         * 目前这种方式（CameraSource销毁的时候,才将FB返回FBCache）依旧存在坑
         * 因为一般Filter使用FBO，是希望能够直接绘制里面的内容，那个如果这个输入的texture的FBO给其他绘制filter使用（非Source），那么就会导致输入的texture的纹理给覆盖了
         */

        _inputTexture = texture;
        Framebuffer *framebuffer =  getContext()->getFramebufferCache()->fetchFramebufferUseTextureId(
                _context, width, height, texture);
        _customTexture = true;
        this->setFramebuffer(framebuffer, outputRotation);
    }

    CHECK_GL(glBindTexture(GL_TEXTURE_2D, this->getFramebuffer()->getTexture()));
}


void SourceCamera::setFrameData(int width,
                                int height,
                                const void* pixels,
                                GLenum pixelsType,
                                GLuint texture,
                                RotationMode outputRotation,
                                SourceType sourceType,
                                const void* upixels,
                                const void* vpixels,
                                bool keep_white)
{
    this->setFramebuffer(0);
    Framebuffer* framebuffer = getContext()->getFramebufferCache()->fetchFramebuffer(_context, width, height, true);
    this->setFramebuffer(framebuffer, outputRotation);
    CHECK_GL(glBindTexture(GL_TEXTURE_2D, this->getFramebuffer()->getTexture()));

       
    switch (sourceType) {
        case SourceType_RGBA:
            if (pixels) {
                CHECK_GL(glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, pixelsType, GL_UNSIGNED_BYTE, pixels));
            }
            break;
        case SourceType_YUV420SP:
            if (upixels) {
                CHECK_GL(glTexImage2D(GL_TEXTURE_2D, 0, GL_LUMINANCE, width, height , 0, GL_LUMINANCE, GL_UNSIGNED_BYTE, pixels));
                
                _UVFrameBuffer = getContext()->getFramebufferCache()->fetchFramebuffer(_context, width * 0.5, height * 0.5, true);
                CHECK_GL(glBindTexture(GL_TEXTURE_2D, _UVFrameBuffer->getTexture()));
                CHECK_GL(glTexImage2D(GL_TEXTURE_2D, 0, GL_LUMINANCE_ALPHA, width * 0.5, height * 0.5, 0, GL_LUMINANCE_ALPHA, GL_UNSIGNED_BYTE, upixels));
            }
            break;
        case SourceType_YUV420P:
            if (upixels && pixels) {
                int w = width * 0.5;
                int h = height * 0.5;
                CHECK_GL(glTexImage2D(GL_TEXTURE_2D, 0, GL_LUMINANCE, width, height , 0, GL_LUMINANCE, GL_UNSIGNED_BYTE, pixels));
                _UVFrameBuffer = getContext()->getFramebufferCache()->fetchFramebuffer(_context, w, h, true);
                CHECK_GL(glBindTexture(GL_TEXTURE_2D, _UVFrameBuffer->getTexture()));

                CHECK_GL(glTexImage2D(GL_TEXTURE_2D, 0, GL_LUMINANCE, w, h, 0, GL_LUMINANCE, GL_UNSIGNED_BYTE, upixels));
                _VFrameBuffer = getContext()->getFramebufferCache()->fetchFramebuffer(_context, w, h, true);
                CHECK_GL(glBindTexture(GL_TEXTURE_2D, _VFrameBuffer->getTexture()));
                CHECK_GL(glTexImage2D(GL_TEXTURE_2D, 0, GL_LUMINANCE, w, h, 0, GL_LUMINANCE, GL_UNSIGNED_BYTE, vpixels));
            }
            break;
        default:
            break;
    }
    CHECK_GL(glBindTexture(GL_TEXTURE_2D, 0));

}
