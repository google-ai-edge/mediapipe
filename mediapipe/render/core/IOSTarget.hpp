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

#ifndef IOSTarget_hpp
#define IOSTarget_hpp

#include "Target.hpp"
#import "GPUImageTarget.h"

NS_GI_BEGIN

class IOSTarget : public Target {
public:
    IOSTarget(id<GPUImageTarget> realTarget) {
        _realTarget = realTarget;
    }
    
    id<GPUImageTarget> getRealTarget(){return _realTarget;};
    
    virtual ~IOSTarget() { _realTarget = 0; }
    
    virtual void update(float frameTime) override {
        [_realTarget update:frameTime];
    };
    
    virtual void setInputFramebuffer(Framebuffer* framebuffer, RotationMode rotationMode = NoRotation,
                                     int texIdx = 0, bool ignoreForPrepared = false) override {
        [ _realTarget setInputFramebuffer:framebuffer withRotation:rotationMode atIndex:texIdx];
    };
    
    virtual bool isPrepared() const override {
        if ([_realTarget respondsToSelector:@selector(isPrepared)])
            return [_realTarget isPrepared];
        else
            return true;
    }
    
    virtual void unPrepear() override {
        if ([_realTarget respondsToSelector:@selector(unPrepared)])
            [_realTarget unPrepared];
    }
    
private:
    id<GPUImageTarget> _realTarget;
    
};

NS_GI_END

#endif // IOSTarget_hpp

#endif
