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

#include "Target.hpp"
#include "GPUImageUtil.h"

NS_GI_BEGIN

Target::Target(int inputNumber/* = 1*/)
:_inputNum(inputNumber)
{
}

Target::~Target()
{
    for (std::map<int, InputFrameBufferInfo>::iterator it = _inputFramebuffers.begin();
         it != _inputFramebuffers.end(); ++it) {
        if (it->second.frameBuffer) {
            it->second.frameBuffer = 0;
        }
    }
    _inputFramebuffers.clear();
}

void Target::setInputFramebuffer(Framebuffer* framebuffer,
                                 RotationMode rotationMode/* = NoRotation*/,
                                 int texIdx/* = 0*/,
                                 bool ignoreForPrepared/* = false*/) {
    InputFrameBufferInfo inputFrameBufferInfo;
    inputFrameBufferInfo.frameBuffer = framebuffer;
    inputFrameBufferInfo.rotationMode = rotationMode;
    inputFrameBufferInfo.texIndex = texIdx;
    inputFrameBufferInfo.ignoreForPrepare = ignoreForPrepared;
    if (_inputFramebuffers.find(texIdx) != _inputFramebuffers.end() &&
        _inputFramebuffers[texIdx].frameBuffer) {
        _inputFramebuffers[texIdx].frameBuffer->unlock(typeid(*this).name());
        _inputFramebuffers[texIdx].frameBuffer = 0;
    }
    _inputFramebuffers[texIdx] = inputFrameBufferInfo;
    if (_inputFramebuffers[texIdx].frameBuffer && !_inputFramebuffers[texIdx].frameBuffer->isDealloc) {
        _inputFramebuffers[texIdx].frameBuffer->lock(typeid(*this).name());
    }
}

int Target::getNextAvailableTextureIndex() const {
    for (int i = 0; i < _inputNum; ++i) {
        if (_inputFramebuffers.find(i) == _inputFramebuffers.end())
            return i;
    }
    return _inputNum - 1;
}

bool Target::isPrepared() const {
    int preparedNum = 0;
    int ignoreForPrepareNum = 0;
    for (std::map<int, InputFrameBufferInfo>::const_iterator it = _inputFramebuffers.begin();
         it != _inputFramebuffers.end(); ++it) {
        if (it->second.ignoreForPrepare)
            ignoreForPrepareNum++;
        else if (it->second.frameBuffer)
            preparedNum++;
    }
    if (ignoreForPrepareNum + preparedNum >= _inputNum)
        return true;
    else
        return false;
}

void Target::unPrepear() {
    for (std::map<int, InputFrameBufferInfo>::iterator it = _inputFramebuffers.begin();
         it != _inputFramebuffers.end(); ++it) {
        if (!it->second.ignoreForPrepare) {
            if (it->second.frameBuffer && !it->second.frameBuffer->isDealloc) {
                it->second.frameBuffer->unlock(typeid(*this).name());
                it->second.frameBuffer = 0;
            }
        }
    }
}

NS_GI_END
