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

#ifndef MEDIAPIPE_GPU_DRISHTIGRAPHGPUDATA_H_
#define MEDIAPIPE_GPU_DRISHTIGRAPHGPUDATA_H_

#import <CoreVideo/CVMetalTextureCache.h>
#import <CoreVideo/CoreVideo.h>
#import <Metal/Metal.h>

#import "mediapipe/gpu/gl_base.h"
#import "mediapipe/gpu/gl_context.h"

namespace mediapipe {
class GlContext;
class GpuBufferMultiPool;
}  // namespace mediapipe

@interface MPPGraphGPUData : NSObject {
  // Shared buffer pool for GPU calculators.
  mediapipe::GpuBufferMultiPool* _gpuBufferPool;
  mediapipe::GlContext* _glContext;
}

- (instancetype)init NS_UNAVAILABLE;

/// Initialize. The provided multipool pointer must remain valid throughout
/// this object's lifetime.
- (instancetype)initWithContext:(mediapipe::GlContext*)context
                      multiPool:(mediapipe::GpuBufferMultiPool*)pool NS_DESIGNATED_INITIALIZER;

/// Shared texture pool for GPU calculators.
/// For internal use by GlCalculatorHelper.
@property(readonly) mediapipe::GpuBufferMultiPool* gpuBufferPool;

/// Shared OpenGL context.
#if TARGET_OS_OSX
@property(readonly) NSOpenGLContext* glContext;
@property(readonly) NSOpenGLPixelFormat* glPixelFormat;
#else
@property(readonly) EAGLContext* glContext;
#endif  // TARGET_OS_OSX

/// Shared texture cache.
#if TARGET_OS_OSX
@property(readonly) CVOpenGLTextureCacheRef textureCache;
#else
@property(readonly) CVOpenGLESTextureCacheRef textureCache;
#endif  // TARGET_OS_OSX

/// Shared Metal resources.
@property(readonly) id<MTLDevice> mtlDevice;
@property(readonly) id<MTLCommandQueue> mtlCommandQueue;
#if COREVIDEO_SUPPORTS_METAL
@property(readonly) CVMetalTextureCacheRef mtlTextureCache;
#endif

@end

#endif  // MEDIAPIPE_GPU_DRISHTIGRAPHGPUDATA_H_
