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

#ifndef Target_hpp
#define Target_hpp

#include "GPUImageMacros.h"
#include "Framebuffer.hpp"
#include <map>

NS_GI_BEGIN

enum RotationMode {
    NoRotation = 0,
    RotateLeft,
    RotateRight,
    FlipVertical,
    FlipHorizontal,
    RotateRightFlipVertical,
    RotateRightFlipHorizontal,
    Rotate180
};

class Target : public virtual Ref {
public:
    Target(int inputNumber = 1);
    virtual ~Target();
    virtual void setInputFramebuffer(Framebuffer* framebuffer,
                                     RotationMode rotationMode = NoRotation,
                                     int texIdx = 0,
                                     bool ignoreForPrepared = false);
    virtual bool isPrepared() const;
    virtual void unPrepear();
    virtual void update(float frameTime) {};
    virtual int getNextAvailableTextureIndex() const;
    //virtual void setInputSizeWithIdx(int width, int height, int textureIdx) {};
protected:
    struct InputFrameBufferInfo {
        Framebuffer* frameBuffer;
        RotationMode rotationMode;
        int texIndex;
        bool ignoreForPrepare;
    };
    
    std::map<int, InputFrameBufferInfo> _inputFramebuffers;
    int _inputNum;
};

NS_GI_END

#endif /* Target_hpp */
