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

#ifndef FramebufferCache_hpp
#define FramebufferCache_hpp

#include "GPUImageMacros.h"
#include "Framebuffer.hpp"
#include <string>
#include <map>

#if defined(__APPLE__)
#define useCVFramebuffer true  //默认关闭 调试的时候再打开，因为CVPixelBuffer 回收较慢会阻塞CPU
#else
#define useCVFramebuffer false
#endif
namespace Opipe {
    
    class Context;
    class FramebufferCache {
    public:
        FramebufferCache(Context *context);
        ~FramebufferCache();
        Framebuffer* fetchFramebuffer(Context *context,
                                      int width,
                                      int height,
                                      bool onlyTexture = false,
                                      const TextureAttributes textureAttributes =
                                      Framebuffer::defaultTextureAttribures,
#if defined(__APPLE__)
                                      bool useTextureCache = useCVFramebuffer);
#else
        bool useTextureCache = false);
#endif
        
        
        /// 通过外部传入的TextureId生成FBO
        /// @param width 宽
        /// @param height 高
        /// @param textureId textureId
        Framebuffer* fetchFramebufferUseTextureId(Context *context,
                                                  int width,
                                                  int height,
                                                  int textureId,
                                                  bool onlyTexture = false,
                                                  const TextureAttributes textureAttributes = Framebuffer::defaultTextureAttribures,
#if defined(__APPLE__)
                                                  bool useTextureCache = useCVFramebuffer);
#else
        bool useTextureCache = false);
#endif
        
        
        void returnFramebuffer(Framebuffer* framebuffer, int maxCacheSize = 1);
        void forceCleanFramebuffer(Framebuffer* framebuffer);
        
        void purge(bool force = false);
        void clearCache();
        
        std::map<std::string, Framebuffer*> allCaches() {
            return _framebuffers;
        }
        
        std::map<std::string, std::map<std::string, int>> allCachesTypeMap() {
            return _framebufferTypeMap;
        }
        
    private:
        std::string _getHash(int width,
                             int height,
                             bool onlyTexture,
                             const TextureAttributes textureAttributes,
                             bool useTextureCache) const;
        Framebuffer* _getFramebufferByHash(const std::string& hash);
        
        std::map<std::string, Framebuffer*> _framebuffers;
        std::map<std::string, std::map<std::string, int>> _framebufferTypeMap;
        Context *_context;
        
    };
    
}

#endif /* FramebufferCache_hpp */
