//
//  OlaShareTextureFilter.cpp
//  AREmotion
//
//  Created by Renzhu Wang on 07/12/2017.
//  Copyright © 2022 olachat. All rights reserved.
//

#include "OlaShareTextureFilter.hpp"


namespace Opipe {
    
    const std::string kOnScreenFragmentShaderString = SHADER_STRING
    (
     varying highp vec2 vTexCoord;
     uniform sampler2D colorMap;
     void main() {
        lowp
        vec4 textureColor = texture2D(colorMap, vTexCoord);
        gl_FragColor = vec4(textureColor.rgb, textureColor.a);
    });
    
    OlaShareTextureFilter::OlaShareTextureFilter(Context *context) : Opipe::Filter(context), targetTextureId(-1) {}
    
    OlaShareTextureFilter *OlaShareTextureFilter::create(Context *context) {
        OlaShareTextureFilter *ret = new(std::nothrow) OlaShareTextureFilter(context);
        if (!ret || !ret->init(context)) {
            delete ret;
            ret = 0;
        }
        return ret;
    }
    
    OlaShareTextureFilter *
    OlaShareTextureFilter::create(Context *context, GLuint targetTextureId, TextureAttributes attributes) {
        auto *ret = new(std::nothrow) OlaShareTextureFilter(context);
        if (!ret || !ret->init(context)) {
            delete ret;
            ret = nullptr;
        }
        ret->targetTextureId = targetTextureId;
        ret->targetTextureAttr = attributes;
        return ret;
    }
    
    bool OlaShareTextureFilter::init(Context *context) {
        if (!Opipe::Filter::initWithFragmentShaderString(context, kOnScreenFragmentShaderString, 1)) {
            return false;
        }
        return true;
    }
    
    
    bool OlaShareTextureFilter::proceed(float frameTime, bool bUpdateTargets/* = true*/) {
        if (!_filterProgram->isValid()) {
            delete _filterProgram;
            _filterProgram = 0;
            _filterProgram = GLProgram::createByShaderString(_context,
                                                             kDefaultVertexShader,
                                                             kOnScreenFragmentShaderString);
        }
        
        return Filter::proceed(frameTime, bUpdateTargets);
    }
    
    OlaShareTextureFilter::~OlaShareTextureFilter() noexcept {
        //不用去释放 交给Context 去Purge
        if (_targetFramebuffer) {
            delete _framebuffer;
            _framebuffer = 0;
        } else {
            _framebuffer = 0;
        }
    }
    
    void OlaShareTextureFilter::updateTargetId(GLuint targetId) {
        targetTextureId = targetId;
    }
    
    void OlaShareTextureFilter::update(float frameTime) {
        if (_inputFramebuffers.empty()) return;
        
        Framebuffer *firstInputFramebuffer = _inputFramebuffers.begin()->second.frameBuffer;
        RotationMode firstInputRotation = _inputFramebuffers.begin()->second.rotationMode;
        if (!firstInputFramebuffer) return;
        
        int rotatedFramebufferWidth = firstInputFramebuffer->getWidth();
        int rotatedFramebufferHeight = firstInputFramebuffer->getHeight();
        if (rotationSwapsSize(firstInputRotation)) {
            rotatedFramebufferWidth = firstInputFramebuffer->getHeight();
            rotatedFramebufferHeight = firstInputFramebuffer->getWidth();
        }
        
        if (_framebufferScale != 1.0) {
            rotatedFramebufferWidth = int(rotatedFramebufferWidth * _framebufferScale);
            rotatedFramebufferHeight = int(rotatedFramebufferHeight * _framebufferScale);
        }
        
        
        if (_framebuffer != nullptr && (_framebuffer->getWidth() != rotatedFramebufferWidth ||
                                        _framebuffer->getHeight() != rotatedFramebufferHeight)) {
            _framebuffer = 0;
        }
        
        if (_framebuffer == nullptr || _framebuffer->isDealloc) {
            _framebuffer = getContext()->getFramebufferCache()->
            fetchFramebuffer(_context,
                             rotatedFramebufferWidth,
                             rotatedFramebufferHeight,
                             false,
                             targetTextureAttr);
            _framebuffer->lock();
        }

        if (_framebuffer) {
            targetTextureId = _framebuffer->getTexture();
        }
        
        proceed(frameTime);
    }
    
}
