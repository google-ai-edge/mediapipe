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

#include "FramebufferCache.hpp"
#include "GPUImageUtil.h"

#if defined(__APPLE__)
#include "CVFramebuffer.hpp"
#endif

NS_GI_BEGIN

FramebufferCache::FramebufferCache(Context *context)
: _context(context) {
}

FramebufferCache::~FramebufferCache() {
    purge();
}

Framebuffer* FramebufferCache::fetchFramebufferUseTextureId(Context *context,
                                                            int width,
                                                            int height,
                                                            int textureId,
                                                            bool onlyTexture,
                                                            const TextureAttributes textureAttributes,
                                                            bool useTextureCache) {
    Framebuffer* framebufferFromCache = 0;
    if (useTextureCache) {
#if defined(__APPLE__)
        //分平台创建 TextureCache
        framebufferFromCache = new CVFramebuffer(context,
                                                 width,
                                                 height,
                                                 textureAttributes,
                                                 textureId);
#elif defined(__ANDROID__) || defined(ANDROID)
        assert("Android HardwareBuffer not support reuse now");
#endif
    } else {
        framebufferFromCache = new Framebuffer(context,
                                               width,
                                               height,
                                               textureAttributes,
                                               textureId);
    }
    return framebufferFromCache;
    
}



Framebuffer* FramebufferCache::fetchFramebuffer(Context *context,
                                                int width,
                                                int height,
                                                bool onlyTexture/* = false*/,
                                                const TextureAttributes textureAttributes/* = defaultTextureAttribure*/,
                                                bool useTextureCache) {
    
    Framebuffer* framebufferFromCache = 0;
    std::string lookupHash = _getHash(width, height, onlyTexture, textureAttributes, useTextureCache);
    
    auto matchFramebuffersHashCode = _framebufferTypeMap[lookupHash];
    
    if (matchFramebuffersHashCode.size() > 0) {
        for (const auto &framebufferHashCodeKey : matchFramebuffersHashCode) {
            auto *framebuffer = _framebuffers[framebufferHashCodeKey.first];
            if (framebuffer == NULL) {
                framebuffer = 0;
                break;
            }
            if (framebuffer->getWidth() != width || framebuffer->getHeight() != height) {
                forceCleanFramebuffer(framebuffer);
                framebuffer = 0;
            } else if (framebuffer->framebufferRetainCount() == 0 && !framebuffer->isDealloc) {
//                Log("Framebuffer 【命中缓存】", "hashcode:%s count:%d",
//                    framebufferHashCodeKey.first.c_str(),
//                    framebuffer->framebufferRetainCount());
                return framebuffer;
            }
        }
    }
//    Log("Framebuffer 所有缓存【未命中】", "hashcode:%s count:%d",
//        lookupHash.c_str(),
//        matchFramebuffersHashCode.size());
    // 如果都被占用了 或者找不到对应的Framebuffer 则需要创建一个新的
    
    if (useTextureCache) {
#if defined(__APPLE__)
        //分平台创建 TextureCache
        framebufferFromCache = new CVFramebuffer(context,
                                                 width,
                                                 height,
                                                 onlyTexture,
                                                 textureAttributes);
#elif defined(__ANDROID__) || defined(ANDROID)
        assert("Android HardwareBuffer not support reuse now");
#endif
    } else {
        framebufferFromCache = new Framebuffer(context,
                                               width,
                                               height,
                                               onlyTexture,
                                               textureAttributes);
    }
    
    std::string framebufferHash = str_format("%s-%ld", lookupHash.c_str(),
                                             framebufferFromCache->getTexture());
    Log("Framebuffer 创建新的Framebuffer", "hashcode:%s numberOfMatchingFramebuffers:%d",
        framebufferHash.c_str(),
        matchFramebuffersHashCode.size());
    framebufferFromCache->_hashCode = framebufferHash;
    framebufferFromCache->_typeCode = lookupHash;
    _framebuffers[framebufferHash] = framebufferFromCache;
    _framebufferTypeMap[lookupHash][framebufferHash] = 0;
    return framebufferFromCache;
}

void FramebufferCache::forceCleanFramebuffer(Framebuffer *framebuffer) {
    if (_framebuffers.find(framebuffer->_hashCode) != _framebuffers.end()) {
        _framebuffers.erase(framebuffer->_hashCode);
    }
    
    if (_framebufferTypeMap[framebuffer->_typeCode].find(framebuffer->_hashCode) !=
        _framebufferTypeMap[framebuffer->_typeCode].end()) {
        _framebufferTypeMap[framebuffer->_typeCode].erase(framebuffer->_hashCode);
    }
    
    delete framebuffer;
    framebuffer = 0;
}

void FramebufferCache::returnFramebuffer(Framebuffer* framebuffer, int maxCacheSize) {
    if (framebuffer->framebufferRetainCount() == 0) {
        Log("准备回收 retainCount == 0 的Framebuffer", "cacheHash:%s cacheReferenceCount:%d",
            framebuffer->_hashCode.c_str(),
            framebuffer->framebufferRetainCount());
        if (_framebuffers.find(framebuffer->_hashCode) != _framebuffers.end()) {
            
            if (_framebufferTypeMap[framebuffer->_typeCode].size() > maxCacheSize) {
                _framebuffers.erase(framebuffer->_hashCode);
                if (_framebufferTypeMap[framebuffer->_typeCode].find(framebuffer->_hashCode) !=
                    _framebufferTypeMap[framebuffer->_typeCode].end()) {
                    _framebufferTypeMap[framebuffer->_typeCode].erase(framebuffer->_hashCode);
                }
                delete framebuffer;
                framebuffer = 0;
            }
        }
    }
}

std::string FramebufferCache::_getHash(int width,
                                       int height,
                                       bool onlyTexture,
                                       const TextureAttributes textureAttributes,
                                       bool useTextureCache) const {
    const char *formatStr = "";
    if (onlyTexture) {
        formatStr = "%.1dx%.1d-%d:%d:%d:%d:%d:%d:%d-NOFB";
    } else {
        formatStr = "%.1dx%.1d-%d:%d:%d:%d:%d:%d:%d";
    }
    
    return str_format(formatStr,
                      width, height,
                      textureAttributes.minFilter, textureAttributes.magFilter,
                      textureAttributes.wrapS, textureAttributes.wrapT,
                      textureAttributes.internalFormat, textureAttributes.format,
                      textureAttributes.type);
}

Framebuffer* FramebufferCache::_getFramebufferByHash(const std::string& hash) {
    return _framebuffers[hash];
}

void FramebufferCache::purge(bool force) {
    if (_framebuffers.size() == 0) {
        return;
    }
    for(auto &kvp : _framebuffers) {
        if (kvp.second && !kvp.second->isDealloc) {
            delete kvp.second;
            kvp.second = nullptr;
        }
    }
    _framebuffers.clear();
    _framebufferTypeMap.clear();
}

void FramebufferCache::clearCache() {
    _framebuffers.clear();
    _framebufferTypeMap.clear();
}

NS_GI_END
