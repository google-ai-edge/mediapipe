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


#include <assert.h>
#include "GPUImageUtil.h"
#include "Ref.hpp"

NS_GI_BEGIN

Ref::Ref()
:_referenceCount(1)
{
}

Ref::~Ref() {
}

void Ref::retain() {
    ++_referenceCount;
}

void Ref::release() {

//    assert(_referenceCount > 0);
    if (_referenceCount == 0) {
        delete this;
    } else {
        --_referenceCount;
        if (_referenceCount == 0) {
            delete this;
        }
    }
    
}

void Ref::resetRefenceCount() {
    _referenceCount = 1;
}

unsigned int Ref::getReferenceCount() const {
    return _referenceCount;
}


NS_GI_END
