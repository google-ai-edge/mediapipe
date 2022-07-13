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

#ifndef Source_hpp
#define Source_hpp
#if defined(__APPLE__)

#import "GPUImageTarget.h"

#endif
#include "GPUImageMacros.h"
#include "Target.hpp"
#include <map>
#include <functional>


NS_GI_BEGIN

class Context;
class Filter;
class Source : public virtual Ref {
public:
    Source();
    Source(Context *context);
    virtual ~Source();
    virtual Source* addTarget(Target* target);
    virtual Source* addTarget(Target* target, int texIdx);
    virtual Source* addTarget(Target* target, int texIdx, bool ignoreForPrepared);
#if defined(__APPLE__)
    virtual Source* addTarget(id<GPUImageTarget> target);
    virtual void removeTarget(id<GPUImageTarget> target);
#endif
    virtual void removeTarget(Target* target);
    virtual void removeAllTargets();
    virtual bool hasTarget(const Target* target) const;
    virtual std::map<Target*, int>& getTargets() { return _targets; };

    virtual void setFramebuffer(Framebuffer* fb, RotationMode outputRotation = RotationMode::NoRotation);
    virtual Framebuffer* getFramebuffer() const;
    
    void setFramebufferScale(float framebufferScale) { _framebufferScale = framebufferScale; }
    int getRotatedFramebufferWidth() const;
    int getRotatedFramebufferHeight() const;
    
    virtual bool proceed(float frameTime = 0.0, bool bUpdateTargets = true);
    virtual void updateTargets(float frameTime);

    virtual unsigned char* captureAProcessedFrameData(Filter* upToFilter, int width = 0, int height = 0);
    
    Context *getContext();
protected:
    Framebuffer* _framebuffer = 0;
    RotationMode _outputRotation = NoRotation;
    std::map<Target*, int> _targets;
    float _framebufferScale = 1.0;
    Context *_context = 0;
};


NS_GI_END

#endif /* Source_hpp */
