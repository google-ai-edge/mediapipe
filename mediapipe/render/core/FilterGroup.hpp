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

#ifndef FilterGroup_hpp
#define FilterGroup_hpp

#include "GPUImageMacros.h"
#include "Source.hpp"
#include "Target.hpp"
#include <vector>
#include "Filter.hpp"

NS_GI_BEGIN

class Context;
class FilterGroup : public Filter {
public:
    virtual ~FilterGroup();
    
    static FilterGroup* create(Context *context);
    static FilterGroup* create(Context *context, std::vector<Filter*> filters);
    
    bool init(Context *context);
    bool init(Context *context, std::vector<Filter*> filters);
    bool hasFilter(const Filter* filter) const;
    void addFilter(Filter* filter);
    void removeFilter(Filter* filter);
    void removeAllFilters();
    
    // Manually specify the terminal filter, which is the final output filter of sequence
    // Most often, it's not necessary to specify the terminal filter manually,
    // as the terminal filter will be specified automatically.
    void setTerminalFilter(Filter* filter) { _terminalFilter = filter; }
    
    virtual Source* addTarget(Target* target) override;
    virtual Source* addTarget(Target* target, int texIdx) override;
#if defined(__APPLE__)
    virtual Source* addTarget(id<GPUImageTarget> target) override;
#endif
    virtual void removeTarget(Target* target) override;
    virtual void removeAllTargets() override;
    virtual bool hasTarget(const Target* target) const override;
    virtual std::map<Target*, int>& getTargets() override;
    virtual bool proceed(float frameTime = 0, bool bUpdateTargets = true) override;
    virtual void update(float frameTime) override;
    virtual void updateTargets(float frameTime) override;
    virtual void setFramebuffer(Framebuffer* fb, RotationMode outputRotation = RotationMode::NoRotation) override;
    virtual Framebuffer* getFramebuffer() const override;
    virtual void setInputFramebuffer(Framebuffer* framebuffer, RotationMode rotationMode = NoRotation,
                                     int texIdx = 0, bool ignoreForPrepared = false) override;
    virtual bool isPrepared() const override;
    virtual void unPrepear() override;
    
public:
    std::vector<Filter*> _filters;
    Filter* _terminalFilter;
    
    FilterGroup(Context *context);
    static Filter* _predictTerminalFilter(Filter* filter);
    
};

NS_GI_END

#endif /* FilterGroup_hpp */
