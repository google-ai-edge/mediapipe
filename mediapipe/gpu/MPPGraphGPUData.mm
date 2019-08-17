// Copyright 2019 The MediaPipe Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#import "mediapipe/gpu/MPPGraphGPUData.h"

#import "GTMDefines.h"

#include "mediapipe/gpu/gl_context.h"
#include "mediapipe/gpu/gpu_buffer_multi_pool.h"

#if TARGET_OS_OSX
#import <AppKit/NSOpenGL.h>
#else
#import <OpenGLES/EAGL.h>
#endif  // TARGET_OS_OSX

@implementation MPPGraphGPUData

@synthesize textureCache = _textureCache;
@synthesize mtlDevice = _mtlDevice;
@synthesize mtlCommandQueue = _mtlCommandQueue;
#if COREVIDEO_SUPPORTS_METAL
@synthesize mtlTextureCache = _mtlTextureCache;
#endif

#if TARGET_OS_OSX
typedef CVOpenGLTextureCacheRef CVTextureCacheType;
#else
typedef CVOpenGLESTextureCacheRef CVTextureCacheType;
#endif  // TARGET_OS_OSX

- (instancetype)initWithContext:(mediapipe::GlContext*)context
                      multiPool:(mediapipe::GpuBufferMultiPool*)pool {
  self = [super init];
  if (self) {
    _gpuBufferPool = pool;
    _glContext = context;
  }
  return self;
}

- (void)dealloc {
  if (_textureCache) {
    _textureCache = NULL;
  }
#if COREVIDEO_SUPPORTS_METAL
  if (_mtlTextureCache) {
    CFRelease(_mtlTextureCache);
    _mtlTextureCache = NULL;
  }
#endif
}

#if TARGET_OS_OSX
- (NSOpenGLContext *)glContext {
  return _glContext->nsgl_context();
}

- (NSOpenGLPixelFormat *) glPixelFormat {
  return _glContext->nsgl_pixel_format();
}
#else
- (EAGLContext *)glContext {
  return _glContext->eagl_context();
}
#endif  // TARGET_OS_OSX

- (CVTextureCacheType)textureCache {
  @synchronized(self) {
    if (!_textureCache) {
      _textureCache = _glContext->cv_texture_cache();
    }
  }
  return _textureCache;
}

- (mediapipe::GpuBufferMultiPool *)gpuBufferPool {
  return _gpuBufferPool;
}

- (id<MTLDevice>)mtlDevice {
  @synchronized(self) {
    if (!_mtlDevice) {
      _mtlDevice = MTLCreateSystemDefaultDevice();
    }
  }
  return _mtlDevice;
}

- (id<MTLCommandQueue>)mtlCommandQueue {
  @synchronized(self) {
    if (!_mtlCommandQueue) {
      _mtlCommandQueue = [self.mtlDevice newCommandQueue];
    }
  }
  return _mtlCommandQueue;
}

#if COREVIDEO_SUPPORTS_METAL
- (CVMetalTextureCacheRef)mtlTextureCache {
  @synchronized(self) {
    if (!_mtlTextureCache) {
      CVReturn err = CVMetalTextureCacheCreate(NULL, NULL, self.mtlDevice, NULL, &_mtlTextureCache);
      NSAssert(err == kCVReturnSuccess, @"Error at CVMetalTextureCacheCreate %d", err);
      // TODO: register and flush metal caches too.
    }
  }
  return _mtlTextureCache;
}
#endif

@end
