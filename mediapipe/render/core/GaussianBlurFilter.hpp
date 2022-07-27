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

#ifndef GaussianBlurFilter_hpp
#define GaussianBlurFilter_hpp

#include "GPUImageMacros.h"
#include "FilterGroup.hpp"
#include "GaussianBlurMonoFilter.hpp"

NS_GI_BEGIN

class GaussianBlurFilter : public FilterGroup {
public:
    virtual ~GaussianBlurFilter();
    
    static GaussianBlurFilter* create(Context *context, int radius = 4, float sigma = 2.0, float multiplier = 1.0);
    bool init(Context *context, int radius, float sigma, float multiplier);
    void setRadius(int radius);
    void setSigma(float sigma);
    void setSigma_h(float sigma);
    void setSigma_v(float sigma);
    
protected:
    GaussianBlurFilter(Context *context);
    
private:
    GaussianBlurMonoFilter* _hBlurFilter;
    GaussianBlurMonoFilter* _vBlurFilter;
};


NS_GI_END

#endif /* GaussianBlurFilter_hpp */
