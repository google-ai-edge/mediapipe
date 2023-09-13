// Copyright 2023 The MediaPipe Authors.
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

#import "mediapipe/tasks/ios/vision/core/sources/MPPMask.h"

#import <XCTest/XCTest.h>

/** Unit tests for `MPPMask`. */
@interface MPPMaskTests : XCTestCase

@end

@implementation MPPMaskTests

#pragma mark - Tests

- (void)testInitWithUInt8ArrayNoCopySucceeds {

  NSInteger width = 2;
  NSInteger height = 3;

  UInt8 uint8Data[] = {128, 128, 128, 128, 128, 128};
  float float32Data[] = {0.501f, 0.501f, 0.501f, 0.501f, 0.501f, 0.501f};

  MPPMask *mask = [[MPPMask alloc] initWithUInt8Data:uint8Data width:width height:height shouldCopy:NO];

  XCTAssertEqual(mask.width, width);
  XCTAssertEqual(mask.height, height);

  // Test if UInt8 mask is not copied.
  XCTAssertEqual(mask.uint8Data, (const UInt8*)uint8Data);
  XCTAssertNotEqual(mask.float32Data, NULL);

  for (int i = 0 ; i < width * height ; i ++) {
    XCTAssertEqualWithAccuracy(mask.float32Data[i], float32Data[i], 1e-3f, @"index i = %d", i);
  }

  // Test if repeated Float32 mask accesses return the same array in memory.
  XCTAssertEqual(mask.float32Data, mask.float32Data);
}

- (void)testInitWithUInt8ArrayCopySucceeds {

  NSInteger width = 2;
  NSInteger height = 3;

  UInt8 uint8Data[] = {128, 128, 128, 128, 128, 128};
  float float32Data[] = {0.501f, 0.501f, 0.501f, 0.501f, 0.501f, 0.501f};

  MPPMask *mask = [[MPPMask alloc] initWithUInt8Data:uint8Data width:width height:height shouldCopy:YES];

  XCTAssertEqual(mask.width, width);
  XCTAssertEqual(mask.height, height);

  // Test if UInt8 mask is copied.
  XCTAssertNotEqual(mask.uint8Data, (const UInt8*)uint8Data);
  XCTAssertNotEqual(mask.float32Data, NULL);

  for (int i = 0 ; i < width * height ; i ++) {
    XCTAssertEqualWithAccuracy(mask.float32Data[i], float32Data[i], 1e-3f);
  }

  // Test if repeated Float32 mask accesses return the same array in memory.
  XCTAssertEqual(mask.float32Data, mask.float32Data);
}

- (void)testInitWithFloat32ArrayNoCopySucceeds {

  NSInteger width = 2;
  NSInteger height = 3;

  UInt8 uint8Data[] = {132, 132, 132, 132, 132, 132};
  float float32Data[] = {0.52f, 0.52f, 0.52f, 0.52f, 0.52f, 0.52f};
  MPPMask *mask = [[MPPMask alloc] initWithFloat32Data:float32Data width:width height:height shouldCopy:NO];

  XCTAssertEqual(mask.width, width);
  XCTAssertEqual(mask.height, height);

  // Test if Float32 mask is not copied.
  XCTAssertEqual(mask.float32Data, (const float*)float32Data);
  XCTAssertNotEqual(mask.uint8Data, NULL);

  for (int i = 0 ; i < width * height ; i ++) {
    XCTAssertEqual(mask.uint8Data[i], uint8Data[i]);
  }

  // Test if repeated UInt8 mask accesses return the same array in memory.
  XCTAssertEqual(mask.uint8Data, mask.uint8Data);
}

- (void)testInitWithFloat32ArrayCopySucceeds {

  NSInteger width = 2;
  NSInteger height = 3;

  UInt8 uint8Data[] = {132, 132, 132, 132, 132, 132};
  float float32Data[] = {0.52f, 0.52f, 0.52f, 0.52f, 0.52f, 0.52f};

  MPPMask *mask = [[MPPMask alloc] initWithFloat32Data:float32Data width:width height:height shouldCopy:YES];

  XCTAssertEqual(mask.width, width);
  XCTAssertEqual(mask.height, height);

  // Test if Float32 mask is copied.
  XCTAssertNotEqual(mask.float32Data, (const float*)float32Data);
  XCTAssertNotEqual(mask.uint8Data, NULL);

  for (int i = 0 ; i < width * height ; i ++) {
    XCTAssertEqual(mask.uint8Data[i], uint8Data[i]);
  }

  // Test if repeated UInt8 mask accesses return the same array in memory.
  XCTAssertEqual(mask.uint8Data, mask.uint8Data);
}

@end
