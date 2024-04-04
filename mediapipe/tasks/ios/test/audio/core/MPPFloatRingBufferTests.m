// Copyright 2024 The MediaPipe Authors.
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

#import "mediapipe/tasks/ios/audio/core/sources/MPPFloatRingBuffer.h"
#import "mediapipe/tasks/ios/common/sources/MPPCommon.h"

static NSString *const kExpectedErrorDomain = @"com.google.mediapipe.tasks";

#define AssertEqualErrors(error, expectedError)              \
  XCTAssertNotNil(error);                                    \
  XCTAssertEqualObjects(error.domain, expectedError.domain); \
  XCTAssertEqual(error.code, expectedError.code);            \
  XCTAssertEqualObjects(error.localizedDescription, expectedError.localizedDescription)

#define AssertEqualFloatBuffers(buffer, expectedBuffer)      \
  XCTAssertNotNil(buffer);                                  \
  XCTAssertNotNil(expectedBuffer);                          \
  XCTAssertEqual(buffer.length, expectedBuffer.length);     \
  for (int i = 0; i < buffer.length; i++) {                 \
    XCTAssertEqual(buffer.data[i], expectedBuffer.data[i]); \
  }

@interface MPPFloatRingBufferTests : XCTestCase
@end

@implementation MPPFloatRingBufferTests

- (void)testLoadSucceedsWithFullLengthBuffer {
  const NSUInteger inputDataLength = 5;
  MPPFloatRingBuffer *ringBuffer = [[MPPFloatRingBuffer alloc] initWithLength:inputDataLength];

  float inputData[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};

  // Expected State after load: [1.0, 2.0, 3.0, 4.0, 5.0]
  [self assertLoadSucceedsOnRingBuffer:ringBuffer
                         withInputData:&(inputData[0])inputDataLength:inputDataLength
                          expectedData:&(inputData[0])];
}

- (void)testLoadSucceedsWithPartialLengthBuffer {
  const NSUInteger ringBufferLength = 5;
  MPPFloatRingBuffer *ringBuffer = [[MPPFloatRingBuffer alloc] initWithLength:ringBufferLength];

  const NSUInteger inputDataLength = 3;
  float inputData[] = {1.0f, 2.0f, 3.0f};

  // State after load.
  const float expectedData[] = {0.0f, 0.0f, 1.0f, 2.0f, 3.0f};

  [self assertLoadSucceedsOnRingBuffer:ringBuffer
                         withInputData:&(inputData[0])inputDataLength:inputDataLength
                          expectedData:&(expectedData[0])];
}

- (void)testLoadSucceedsByShiftingOutOldElements {
  const NSUInteger ringBufferLength = 5;
  MPPFloatRingBuffer *ringBuffer = [[MPPFloatRingBuffer alloc] initWithLength:ringBufferLength];

  const NSUInteger firstInputDataLength = 4;
  const float firstInputData[] = {1.0f, 2.0f, 3.0f, 4.0f};

  // State after load
  const float firstExpectedData[] = {0.0f, 1.0f, 2.0f, 3.0f, 4.0f};
  [self assertLoadSucceedsOnRingBuffer:ringBuffer
                         withInputData:&(firstInputData[0])inputDataLength:firstInputDataLength
                          expectedData:&(firstExpectedData[0])];

  NSUInteger secondInputDataLength = 3;
  const float secondInputData[] = {5, 6, 7};

  // State after load
  const float secondExpectedData[] = {3.0f, 4.0f, 5.0f, 6.0f, 7.0f};
  [self assertLoadSucceedsOnRingBuffer:ringBuffer
                         withInputData:&(secondInputData[0])inputDataLength:secondInputDataLength
                          expectedData:&(secondExpectedData[0])];
}

- (void)testLoadSucceedsWithMostRecentElements {
  const NSUInteger ringBufferLength = 5;
  MPPFloatRingBuffer *ringBuffer = [[MPPFloatRingBuffer alloc] initWithLength:ringBufferLength];

  const float firstInputData[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};

  // State after load `[1.0f, 2.0f, 3.0f, 4.0f, 5.0f]`
  [self assertLoadSucceedsOnRingBuffer:ringBuffer
                         withInputData:&(firstInputData[0])inputDataLength:ringBufferLength
                          expectedData:&(firstInputData[0])];

  NSUInteger secondInputDataLength = 6;
  const float secondInputData[] = {6.0f, 7.0f, 8.0f, 9.0f, 10.0f, 11.0f};

  // State after load
  const float secondExpectedData[] = {7.0f, 8.0f, 9.0f, 10.0f, 11.0f};
  [self assertLoadSucceedsOnRingBuffer:ringBuffer
                         withInputData:&(secondInputData[0])inputDataLength:secondInputDataLength
                          expectedData:&(secondExpectedData[0])];
}

- (void)testLoadSucceedsWithOffset {
  const NSUInteger ringBufferLength = 5;
  MPPFloatRingBuffer *ringBuffer = [[MPPFloatRingBuffer alloc] initWithLength:ringBufferLength];

  const NSUInteger firstInputDataLength = 2;
  const float firstInputData[] = {1.0f, 2.0f};

  // State after load
  const float firstExpectedData[] = {0.0f, 0.0f, 0.0f, 1.0f, 2.0f};

  [self assertLoadSucceedsOnRingBuffer:ringBuffer
                         withInputData:&(firstInputData[0])inputDataLength:firstInputDataLength
                          expectedData:&(firstExpectedData[0])];

  NSUInteger secondInputDataLength = 4;
  const float secondInputData[] = {6.0f, 7.0f, 8.0f, 9.0f};
  const NSUInteger offset = 2;
  const NSUInteger lengthOfDataToBeLoaded = 2;

  // State after load
  const float secondExpectedData[] = {0.0f, 1.0f, 2.0f, 8.0f, 9.0f};
  [self assertLoadSucceedsOnRingBuffer:ringBuffer
                         withInputData:&(secondInputData[0])inputDataLength:secondInputDataLength
                                offset:offset
                      lengthToBeLoaded:lengthOfDataToBeLoaded
                          expectedData:&(secondExpectedData[0])];
}

- (void)testLoadSucceedsWithOffsetAndMostRecentElements {
  const NSUInteger ringBufferLength = 5;
  MPPFloatRingBuffer *ringBuffer = [[MPPFloatRingBuffer alloc] initWithLength:ringBufferLength];

  const float firstInputData[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};

  // State after load
  const float firstExpectedData[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
  [self assertLoadSucceedsOnRingBuffer:ringBuffer
                         withInputData:&(firstInputData[0])inputDataLength:ringBufferLength
                          expectedData:&(firstExpectedData[0])];

  NSUInteger secondInputDataLength = 8;
  const float secondInputData[] = {6.0f, 7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f, 13.0f};
  const NSUInteger offset = 2;
  const NSUInteger lengthOfDataToBeLoaded = 6;

  // State after load
  const float secondExpectedData[] = {9.0f, 10.0f, 11.0f, 12.0f, 13.0f};
  [self assertLoadSucceedsOnRingBuffer:ringBuffer
                         withInputData:&(secondInputData[0])inputDataLength:secondInputDataLength
                                offset:offset
                      lengthToBeLoaded:lengthOfDataToBeLoaded
                          expectedData:&(secondExpectedData[0])];
}

- (void)testLoadFailsWithIndexOutofBounds {
  const NSUInteger ringBufferLength = 5;
  MPPFloatRingBuffer *ringBuffer = [[MPPFloatRingBuffer alloc] initWithLength:ringBufferLength];

  const NSUInteger inputDataLength = 2;
  const float inputData[] = {1.0f, 2.0f};

  const NSUInteger offset = 1;
  const NSUInteger lengthOfDataToBeLoaded = 3;
  NSError *error = nil;

  XCTAssertFalse([self loadRingBuffer:ringBuffer
                             withData:inputData
                             ofLength:inputDataLength
                               offset:offset
                     lengthToBeLoaded:lengthOfDataToBeLoaded
                                error:&error]);

  NSError *expectedError = [NSError
      errorWithDomain:kExpectedErrorDomain
                 code:MPPTasksErrorCodeInvalidArgumentError
             userInfo:@{
               NSLocalizedDescriptionKey : [NSString
                   stringWithFormat:@"Index out of range. `offset` (%lu) + `length` (%lu) must be <= "
                                    @"`floatBuffer.length` (%lu)",

                                    offset, lengthOfDataToBeLoaded, inputDataLength]
             }];

  AssertEqualErrors(error, expectedError);
}

- (void)testClearSucceeds {
  const NSUInteger ringBufferLength = 5;
  MPPFloatRingBuffer *ringBuffer = [[MPPFloatRingBuffer alloc] initWithLength:ringBufferLength];

  const float firstInputDataLegth = 2;
  const float firstInputData[] = {1.0f, 2.0f};

  // State after load
  const float firstExpectedData[] = {0.0f, 0.0f, 0.0f, 1.0f, 2.0f};
  [self assertLoadSucceedsOnRingBuffer:ringBuffer
                         withInputData:&(firstInputData[0])inputDataLength:firstInputDataLegth
                          expectedData:&(firstExpectedData[0])];

  [ringBuffer clear];

  const float expectedData[] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
  [self assertFloatBufferOfRingBuffer:ringBuffer equalsExpectedData:&(expectedData[0])];
}

#pragma mark Assertions

// Convenience method to assert if loading ring buffer with input data of a certain length with
// offset 0 and all elements of input data succeeds and the resulting ring buffer equals the
// expected data. Expected data must be of the same length as the ring buffer.
- (void)assertLoadSucceedsOnRingBuffer:(MPPFloatRingBuffer *)ringBuffer
                         withInputData:(const float *)inputData
                       inputDataLength:(NSUInteger)inputDataLength
                          expectedData:(const float *)expectedData {
  [self assertLoadSucceedsOnRingBuffer:ringBuffer
                         withInputData:inputData
                       inputDataLength:inputDataLength
                                offset:0
                      lengthToBeLoaded:inputDataLength
                          expectedData:expectedData];
}

// Method to assert if loading ring buffer with input data of a certain length with
// the given offset and length of elements to be loaded succeeds and the resulting ring buffer
// equals the expected data. Expected data must be of the same length as the ring buffer.
- (void)assertLoadSucceedsOnRingBuffer:(MPPFloatRingBuffer *)ringBuffer
                         withInputData:(const float *)inputData
                       inputDataLength:(NSUInteger)inputDataLength
                                offset:(NSUInteger)offset
                      lengthToBeLoaded:(NSUInteger)lengthToBeLoaded
                          expectedData:(const float *)expectedData {
  XCTAssertTrue([self loadRingBuffer:ringBuffer
                            withData:inputData
                            ofLength:inputDataLength
                              offset:offset
                    lengthToBeLoaded:lengthToBeLoaded
                               error:nil]);

  [self assertFloatBufferOfRingBuffer:ringBuffer equalsExpectedData:expectedData];
}

- (void)assertFloatBufferOfRingBuffer:(MPPFloatRingBuffer *)ringBuffer
                   equalsExpectedData:(const float *)expectedData {
  MPPFloatBuffer *expectedFloatBuffer =
      [[MPPFloatBuffer alloc] initWithData:&(expectedData[0]) length:ringBuffer.length];

  MPPFloatBuffer *actualFloatBuffer = ringBuffer.floatBuffer;
  AssertEqualFloatBuffers(actualFloatBuffer, expectedFloatBuffer);
}

#pragma mark Loading Helpers

- (BOOL)loadRingBuffer:(MPPFloatRingBuffer *)ringBuffer
              withData:(const float *)data
              ofLength:(NSUInteger)length
                offset:(NSUInteger)offset
      lengthToBeLoaded:(NSUInteger)lengthToBeLoaded
                 error:(NSError **)error {
  MPPFloatBuffer *inputBuffer = [[MPPFloatBuffer alloc] initWithData:data length:length];
  return [ringBuffer loadFloatBuffer:inputBuffer offset:offset length:lengthToBeLoaded error:error];
}
@end
