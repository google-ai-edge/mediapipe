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

#include "Source.hpp"
#include "GPUImageUtil.h"
#include "Context.hpp"
#if defined(__APPLE__)
#include "IOSTarget.hpp"
#endif

NS_GI_BEGIN


Source::Source(Context *context)
:_framebuffer(0)
,_outputRotation(RotationMode::NoRotation)
,_framebufferScale(1.0)
,_context(context)
{
    
}

Source::Source() {
    
}

Source::~Source() {
    _framebuffer = 0;

    removeAllTargets();
}

Source* Source::addTarget(Target* target) {
    int targetTexIdx = target->getNextAvailableTextureIndex();
    return addTarget(target, targetTexIdx);
}

Source* Source::addTarget(Target* target, int texIdx) {
    if (!hasTarget(target)) {
        _targets[target] = texIdx;
        if (_framebuffer) {
            target->setInputFramebuffer(_framebuffer, RotationMode::NoRotation, texIdx);
        }
        target->retain();
    }
    return dynamic_cast<Source*>(target);
}

Source* Source::addTarget(Target* target, int texIdx, bool ignoreForPrepared) {
    if (!hasTarget(target)) {
        _targets[target] = texIdx;
        if (_framebuffer) {
            target->setInputFramebuffer(_framebuffer, RotationMode::NoRotation, texIdx,
                                        ignoreForPrepared);
        }
        target->retain();
    }
    return dynamic_cast<Source*>(target);
}

#if defined(__APPLE__)
Source* Source::addTarget(id<GPUImageTarget> target) {
    IOSTarget* iosTarget = new IOSTarget(target);
    addTarget(iosTarget);
    iosTarget->release();
    return 0;
}
void Source::removeTarget(id<GPUImageTarget> target) {
    
    for (std::map<Target*, int>::iterator itr = _targets.begin(); itr != _targets.end(); itr++)
    {
        Target* ref = (Target*)(itr->first);
        if (typeid(ref) == typeid(IOSTarget) && ((IOSTarget*)ref)->getRealTarget() == target)
        {
            ref->release();
            _targets.erase(itr);
            break;
        }
    }
    
}

#endif

bool Source::hasTarget(const Target* target) const {
    if (_targets.find(const_cast<Target*>(target)) != _targets.end())
        return true;
    else
        return false;
}

void Source::removeTarget(Target* target) {
    std::map<Target*, int>::iterator itr = _targets.find(target);
    if (itr != _targets.end()) {
        Ref* ref = (Ref*)(itr->first);
        if (ref) {
            ref->release();
        }
        _targets.erase(itr);
    }
}

void Source::removeAllTargets() {
    for (auto const& target : _targets ) {
        Ref* ref = (Ref*)(target.first);
        if (ref) {
            ref->release();
        }
    }
    _targets.clear();
}

bool Source::proceed(float frameTime, bool bUpdateTargets/* = true*/) {
    if (bUpdateTargets)
        updateTargets(frameTime);
    return true;
}

void Source::updateTargets(float frameTime) {
    for(auto& it : _targets){
        Target* target = it.first;
        if (target == NULL) {
            return;
        }
        
        target->setInputFramebuffer(_framebuffer, _outputRotation, _targets[target]);
    }

    for(auto& it : _targets) {
        Target* target = it.first;
        if (target == NULL) {
            return;
        }
        if (target->isPrepared()) {
            target->update(frameTime);
        }
    }

}

unsigned char* Source::captureAProcessedFrameData(Filter* upToFilter, int width/* = 0*/,
                                                  int height/* = 0*/) {
    if (getContext()->isCapturingFrame) return 0 ;

    if (width <= 0 || height <= 0) {
        if (!_framebuffer) return 0;
        width = getRotatedFramebufferWidth();
        height = getRotatedFramebufferHeight();
    }
    
    getContext()->isCapturingFrame = true;
    getContext()->captureWidth = width;
    getContext()->captureHeight = height;
    getContext()->captureUpToFilter = upToFilter;

    proceed(true);
    unsigned char* processedFrameData = getContext()->capturedFrameData;

    getContext()->capturedFrameData = 0;
    getContext()->captureWidth = 0;
    getContext()->captureHeight = 0;
    getContext()->isCapturingFrame = false;
    
    return processedFrameData;
}

void Source::setFramebuffer(Framebuffer* fb,
                            RotationMode outputRotation/* = RotationMode::NoRotation*/) {
    if (_framebuffer != fb && _framebuffer != 0) {
        _framebuffer = 0;
    }
    _framebuffer = fb;
    _outputRotation = outputRotation;
}

int Source::getRotatedFramebufferWidth() const {
    if (_framebuffer)
        if (rotationSwapsSize(_outputRotation))
            return _framebuffer->getHeight();
        else
            return _framebuffer->getWidth();
    else
        return 0;
}

int Source::getRotatedFramebufferHeight() const {
    if (_framebuffer)
        if (rotationSwapsSize(_outputRotation))
            return _framebuffer->getWidth();
        else
            return _framebuffer->getHeight();
    else
        return 0;
}

Framebuffer* Source::getFramebuffer() const {
    return _framebuffer;
}

Context *Source::getContext() {
    if (_context) {
        return _context;
    }
    
    return NULL;
}
NS_GI_END
