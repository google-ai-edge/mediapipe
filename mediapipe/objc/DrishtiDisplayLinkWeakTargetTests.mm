// Copyright 2021 The MediaPipe Authors.
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

#import <XCTest/XCTest.h>

#import "mediapipe/objc/MPPDisplayLinkWeakTarget.h"

@interface DummyTarget : NSObject

@property(nonatomic) BOOL updateCalled;
- (void)update:(id)sender;

@end

@implementation DummyTarget

@synthesize updateCalled = _updateCalled;

- (void)update:(id)sender {
  _updateCalled = YES;
}

@end

@interface MPPDisplayLinkWeakTargetTests : XCTestCase
@end

@implementation MPPDisplayLinkWeakTargetTests {
  DummyTarget *_dummyTarget;
}

- (void)setUp {
  _dummyTarget = [[DummyTarget alloc] init];
}

- (void)testCallingLiveTarget {
  XCTAssertFalse(_dummyTarget.updateCalled);

  MPPDisplayLinkWeakTarget *target =
      [[MPPDisplayLinkWeakTarget alloc] initWithTarget:_dummyTarget
                                                  selector:@selector(update:)];
  [target displayLinkCallback:nil];

  XCTAssertTrue(_dummyTarget.updateCalled);
}

- (void)testDoesNotCrashWhenTargetIsDeallocated {
  MPPDisplayLinkWeakTarget *target =
      [[MPPDisplayLinkWeakTarget alloc] initWithTarget:_dummyTarget
                                                  selector:@selector(update:)];
  _dummyTarget = nil;
  [target displayLinkCallback:nil];

  XCTAssertNil(_dummyTarget);
}

@end
