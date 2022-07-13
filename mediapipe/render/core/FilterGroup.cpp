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
#include <algorithm>
#include "FilterGroup.hpp"
#include "Context.hpp"

NS_GI_BEGIN

REGISTER_FILTER_CLASS(FilterGroup)

FilterGroup::FilterGroup(Context *context) : Filter(context)
, _terminalFilter(0)
{
}

FilterGroup::~FilterGroup() {
    removeAllFilters();
    _terminalFilter = 0;
}

FilterGroup* FilterGroup::create(Context *context) {
    FilterGroup* ret = new (std::nothrow) FilterGroup(context);
    if (ret && ret->init(context)) {
        //ret->autorelease();
    } else {
        delete ret;
        ret = 0;
    }
    return ret;
}

FilterGroup* FilterGroup::create(Context *context, std::vector<Filter*> filters) {
    FilterGroup* ret = new (std::nothrow) FilterGroup(context);
    if (ret && ret->init(context, filters)) {
        //ret->autorelease();
    } else {
        delete ret;
        ret = 0;
    }
    return ret;
}

bool FilterGroup::init(Context *context) {
    return true;
}


bool FilterGroup::init(Context *context, std::vector<Filter*> filters) {
    if (filters.size() == 0) return true;
    _filters = filters;
    
    for (auto const& filter : filters ) {
        Ref* ref = dynamic_cast<Ref*>(filter);
        if (ref) {
            ref->retain();
        }
    }
    
    setTerminalFilter(_predictTerminalFilter(filters[filters.size() - 1]));
    return true;
}

bool FilterGroup::hasFilter(const Filter* filter) const {
    std::vector<Filter*>::const_iterator it = std::find(_filters.begin(), _filters.end(), filter);
    if (it != _filters.end())
        return true;
    else
        return false;
}

void FilterGroup::addFilter(Filter* filter) {
    if (hasFilter(filter)) return;
    
    _filters.push_back(filter);
    
    Ref* ref = dynamic_cast<Ref*>(filter);
    if (ref) {
        ref->retain();
    }
    
    setTerminalFilter(_predictTerminalFilter(filter));
}

void FilterGroup::removeFilter(Filter* filter) {
    std::vector<Filter*>::iterator itr = std::find(_filters.begin(), _filters.end(), filter);
    if (itr != _filters.end()) {
        Ref* ref = dynamic_cast<Ref*>(*itr);
        if (ref) {
            ref->release();
        }
        _filters.erase(itr);
    }
}

void FilterGroup::removeAllFilters() {
    for (auto const& filter : _filters ) {
        Ref* ref = dynamic_cast<Ref*>(filter);
        if (ref) {
            ref->release();
        }
    }
    _filters.clear();
}

Filter* FilterGroup::_predictTerminalFilter(Filter* filter) {
    if (filter->getTargets().size() == 0)
        return filter;
    else
        return _predictTerminalFilter(dynamic_cast<Filter*>(filter->getTargets().begin()->first));
}

Source* FilterGroup::addTarget(Target* target) {
    if (_terminalFilter)
        return _terminalFilter->addTarget(target);
    else
        return 0;
}

Source* FilterGroup::addTarget(Target* target, int texIdx) {
    if (_terminalFilter)
        return _terminalFilter->addTarget(target, texIdx);
    else
        return 0;
}

#if defined(__APPLE__)
Source* FilterGroup::addTarget(id<GPUImageTarget> target) {
    if (_terminalFilter)
        return _terminalFilter->addTarget(target);
    else
        return 0;
}
#endif

void FilterGroup::removeTarget(Target* target) {
    if (_terminalFilter)
        _terminalFilter->removeTarget(target);
}

void FilterGroup::removeAllTargets() {
    if (_terminalFilter)
        _terminalFilter->removeAllTargets();
}

bool FilterGroup::hasTarget(const Target* target) const {
    if (_terminalFilter)
        return _terminalFilter->hasTarget(target);
    else
        return false;
}

std::map<Target*, int>& FilterGroup::getTargets() {
    assert(_terminalFilter);
    return _terminalFilter->getTargets();
}

bool FilterGroup::proceed(float frameTime, bool bUpdateTargets/* = true*/) {
    
    return true;
}

void FilterGroup::update(float frameTime) {
    proceed(frameTime);
    if (getContext()->isCapturingFrame && this == getContext()->captureUpToFilter) {
        getContext()->captureUpToFilter = _terminalFilter;
    }
    
    for(auto& filter : _filters){
        if (filter->isPrepared()) {
            filter->update(frameTime);
            filter->unPrepear();
        }
    }
}

void FilterGroup::updateTargets(float frameTime) {
    if (_terminalFilter)
        _terminalFilter->updateTargets(frameTime);
}

void FilterGroup::setFramebuffer(Framebuffer* fb, RotationMode outputRotation/* = RotationMode::NoRotation*/) {
    //if (_terminalFilter)
    //    _terminalFilter->setFramebuffer(fb);
}

Framebuffer* FilterGroup::getFramebuffer() const {
    //if (_terminalFilter)
    //    return _terminalFilter->getFramebuffer();
    return 0;
}

void FilterGroup::setInputFramebuffer(Framebuffer* framebuffer,
                                      RotationMode rotationMode/* = NoRotation*/,
                                      int texIdx/* = 0*/, bool ignoreForPrepared) {
    for (auto& filter : _filters) {
        filter->setInputFramebuffer(framebuffer, rotationMode, texIdx);
    }
}

bool FilterGroup::isPrepared() const {
    //    for (auto& filter : _filters) {
    //        if (!filter->isPrepared())
    //            return false;
    //    }
    return true;
}

void FilterGroup::unPrepear() {
    //for (auto& filter : _filters) {
    //    filter->unPrepeared();
    //}
}

NS_GI_END
