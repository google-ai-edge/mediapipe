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

#import <UIKit/UIKit.h>
#import <XCTest/XCTest.h>

#include <memory>

#include "absl/memory/memory.h"
#include "mediapipe/framework/port/threadpool.h"

#import "mediapipe/gpu/MPPGraphGPUData.h"
#import "mediapipe/gpu/gpu_shared_data_internal.h"

@interface MPPGraphGPUDataTests : XCTestCase {
}
@end

@implementation MPPGraphGPUDataTests

// This test verifies that the internal Objective-C object is correctly
// released when the C++ wrapper is released.
- (void)testCorrectlyReleased {
  __weak id gpuData = nil;
  std::weak_ptr<mediapipe::GpuResources> gpuRes;
  @autoreleasepool {
    mediapipe::GpuSharedData gpu_shared;
    gpuRes = gpu_shared.gpu_resources;
    gpuData = gpu_shared.gpu_resources->ios_gpu_data();
    XCTAssertNotEqual(gpuRes.lock(), nullptr);
    XCTAssertNotNil(gpuData);
  }
  XCTAssertEqual(gpuRes.lock(), nullptr);
  XCTAssertNil(gpuData);
}

// This test verifies that the lazy initialization of the glContext instance
// variable is thread-safe. All threads should read the same value.
- (void)testGlContextThreadSafeLazyInitialization {
  mediapipe::GpuSharedData gpu_shared;
  constexpr int kNumThreads = 10;
  EAGLContext* ogl_context[kNumThreads];
  auto pool = absl::make_unique<mediapipe::ThreadPool>(kNumThreads);
  pool->StartWorkers();
  for (int i = 0; i < kNumThreads; ++i) {
    pool->Schedule([&gpu_shared, &ogl_context, i] {
      ogl_context[i] = gpu_shared.gpu_resources->ios_gpu_data().glContext;
    });
  }
  pool.reset();
  for (int i = 0; i < kNumThreads - 1; ++i) {
    XCTAssertEqual(ogl_context[i], ogl_context[i + 1]);
  }
}

// This test verifies that the lazy initialization of the textureCache instance
// variable is thread-safe. All threads should read the same value.
- (void)testTextureCacheThreadSafeLazyInitialization {
  mediapipe::GpuSharedData gpu_shared;
  constexpr int kNumThreads = 10;
  CFHolder<CVOpenGLESTextureCacheRef> texture_cache[kNumThreads];
  auto pool = absl::make_unique<mediapipe::ThreadPool>(kNumThreads);
  pool->StartWorkers();
  for (int i = 0; i < kNumThreads; ++i) {
    pool->Schedule([&gpu_shared, &texture_cache, i] {
      texture_cache[i].reset(gpu_shared.gpu_resources->ios_gpu_data().textureCache);
    });
  }
  pool.reset();
  for (int i = 0; i < kNumThreads - 1; ++i) {
    XCTAssertEqual(*texture_cache[i], *texture_cache[i + 1]);
  }
}

@end
