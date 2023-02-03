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

#import "mediapipe/gpu/gpu_shared_data_internal.h"
#import "mediapipe/gpu/metal_shared_resources.h"

@interface MPPMetalSharedResourcesTests : XCTestCase {
}
@end

@implementation MPPMetalSharedResourcesTests

// This test verifies that the internal Objective-C object is correctly
// released when the C++ wrapper is released.
- (void)testCorrectlyReleased {
  __weak id metalRes = nil;
  std::weak_ptr<mediapipe::GpuResources> weakGpuRes;
  @autoreleasepool {
    auto maybeGpuRes = mediapipe::GpuResources::Create();
    XCTAssertTrue(maybeGpuRes.ok());
    weakGpuRes = *maybeGpuRes;
    metalRes = (**maybeGpuRes).metal_shared().resources();
    XCTAssertNotEqual(weakGpuRes.lock(), nullptr);
    XCTAssertNotNil(metalRes);
  }
  XCTAssertEqual(weakGpuRes.lock(), nullptr);
  XCTAssertNil(metalRes);
}

@end
