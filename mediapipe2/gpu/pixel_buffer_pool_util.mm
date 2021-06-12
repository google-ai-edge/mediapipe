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

#include "mediapipe/gpu/pixel_buffer_pool_util.h"

#import <Foundation/Foundation.h>

#include "mediapipe/objc/util.h"

#if !defined(ENABLE_MEDIAPIPE_GPU_BUFFER_THRESHOLD_CHECK) && !defined(NDEBUG)
#define ENABLE_MEDIAPIPE_GPU_BUFFER_THRESHOLD_CHECK 1
#endif  // defined(ENABLE_MEDIAPIPE_GPU_BUFFER_THRESHOLD_CHECK)

namespace mediapipe {

CVPixelBufferPoolRef CreateCVPixelBufferPool(
    int width, int height, OSType pixelFormat, int keepCount,
    CFTimeInterval maxAge) {
  CVPixelBufferPoolRef pool = NULL;

  NSMutableDictionary *sourcePixelBufferOptions =
      [(__bridge NSDictionary*)GetCVPixelBufferAttributesForGlCompatibility() mutableCopy];
  [sourcePixelBufferOptions addEntriesFromDictionary:@{
    (id)kCVPixelBufferPixelFormatTypeKey : @(pixelFormat),
    (id)kCVPixelBufferWidthKey : @(width),
    (id)kCVPixelBufferHeightKey : @(height),
  }];

  NSMutableDictionary *pixelBufferPoolOptions = [[NSMutableDictionary alloc] init];
  pixelBufferPoolOptions[(id)kCVPixelBufferPoolMinimumBufferCountKey] = @(keepCount);
  if (maxAge > 0) {
    pixelBufferPoolOptions[(id)kCVPixelBufferPoolMaximumBufferAgeKey] = @(maxAge);
  }

  CVPixelBufferPoolCreate(
      kCFAllocatorDefault, (__bridge CFDictionaryRef)pixelBufferPoolOptions,
      (__bridge CFDictionaryRef)sourcePixelBufferOptions, &pool);

  return pool;
}

OSStatus PreallocateCVPixelBufferPoolBuffers(
    CVPixelBufferPoolRef pool, int count, CFDictionaryRef auxAttributes) {
  CVReturn err = kCVReturnSuccess;
  NSMutableArray *pixelBuffers = [[NSMutableArray alloc] init];
  for (int i = 0; i < count && err == kCVReturnSuccess; i++) {
    CVPixelBufferRef pixelBuffer = NULL;
    err = CVPixelBufferPoolCreatePixelBufferWithAuxAttributes(
        kCFAllocatorDefault, pool, auxAttributes, &pixelBuffer);
    if (err != kCVReturnSuccess) {
      break;
    }

    [pixelBuffers addObject:(__bridge id)pixelBuffer];
    CFRelease(pixelBuffer);
  }
  return err;
}

CFDictionaryRef CreateCVPixelBufferPoolAuxiliaryAttributesForThreshold(int allocationThreshold) {
  if (allocationThreshold > 0) {
    return (CFDictionaryRef)CFBridgingRetain(
        @{(id)kCVPixelBufferPoolAllocationThresholdKey: @(allocationThreshold)});
  } else {
    return nil;
  }
}

CVReturn CreateCVPixelBufferWithPool(
    CVPixelBufferPoolRef pool, CFDictionaryRef auxAttributes,
    CVTextureCacheType textureCache, CVPixelBufferRef* outBuffer) {
  return CreateCVPixelBufferWithPool(pool, auxAttributes, [textureCache](){
#if TARGET_OS_OSX
      CVOpenGLTextureCacheFlush(textureCache, 0);
#else
      CVOpenGLESTextureCacheFlush(textureCache, 0);
#endif  // TARGET_OS_OSX
  }, outBuffer);
}

CVReturn CreateCVPixelBufferWithPool(
    CVPixelBufferPoolRef pool, CFDictionaryRef auxAttributes,
    std::function<void(void)> flush, CVPixelBufferRef* outBuffer) {
  CVReturn err = CVPixelBufferPoolCreatePixelBufferWithAuxAttributes(
      kCFAllocatorDefault, pool, auxAttributes, outBuffer);
  if (err == kCVReturnWouldExceedAllocationThreshold) {
    if (flush) {
      // Call the flush function to potentially release the retained buffers
      // and try again to create a pixel buffer.
      flush();
      err = CVPixelBufferPoolCreatePixelBufferWithAuxAttributes(
          kCFAllocatorDefault, pool, auxAttributes, outBuffer);
    }
    if (err == kCVReturnWouldExceedAllocationThreshold) {
      // TODO: allow the application to set the threshold. For now, disable it by
      // default, since the threshold we are using is arbitrary and some graphs routinely cross it.
#ifdef ENABLE_MEDIAPIPE_GPU_BUFFER_THRESHOLD_CHECK
      NSLog(@"Using more buffers than expected! This is a debug-only warning, "
            "you can ignore it if your app works fine otherwise.");
#ifdef DEBUG
      NSLog(@"Pool status: %@", ((__bridge NSObject *)pool).description);
#endif  // DEBUG
#endif  // defined(ENABLE_MEDIAPIPE_GPU_BUFFER_THRESHOLD_CHECK)
      // Try again and ignore threshold.
      // TODO drop a frame instead?
      err = CVPixelBufferPoolCreatePixelBufferWithAuxAttributes(
          kCFAllocatorDefault, pool, NULL, outBuffer);
    }
  }
  return err;
}

#if TARGET_IPHONE_SIMULATOR
static void FreeRefConReleaseCallback(void* refCon, const void* baseAddress) {
  free(refCon);
}
#endif

CVReturn CreateCVPixelBufferWithoutPool(
    int width, int height, OSType pixelFormat, CVPixelBufferRef* outBuffer) {
#if TARGET_IPHONE_SIMULATOR
  // On the simulator, syncing the texture with the pixelbuffer does not work,
  // and we have to use glReadPixels. Since GL_UNPACK_ROW_LENGTH is not
  // available in OpenGL ES 2, we should create the buffer so the pixels are
  // contiguous.
  //
  // TODO: verify if we can use kIOSurfaceBytesPerRow to force
  // CoreVideo to give us contiguous data.
  size_t bytes_per_row = width * 4;
  void* data = malloc(bytes_per_row * height);
  return CVPixelBufferCreateWithBytes(
      kCFAllocatorDefault, width, height, pixelFormat, data, bytes_per_row,
      FreeRefConReleaseCallback, data, GetCVPixelBufferAttributesForGlCompatibility(),
      outBuffer);
#else
  return CVPixelBufferCreate(
      kCFAllocatorDefault, width, height, pixelFormat,
      GetCVPixelBufferAttributesForGlCompatibility(), outBuffer);
#endif
}

}  // namespace mediapipe
