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

#import <AVFoundation/AVFoundation.h>
#import <XCTest/XCTest.h>

#import "mediapipe/tasks/ios/audio/core/sources/MPPAudioRecord.h"
#import "mediapipe/tasks/ios/common/sources/MPPCommon.h"
#import "mediapipe/tasks/ios/test/utils/sources/MPPFileInfo.h"

static MPPFileInfo *const kSpeech16KHzMonoFileInfo =
    [[MPPFileInfo alloc] initWithName:@"speech_16000_hz_mono" type:@"wav"];

static NSString *const kExpectedErrorDomain = @"com.google.mediapipe.tasks";

#define AssertEqualErrors(error, expectedError)              \
  XCTAssertNotNil(error);                                    \
  XCTAssertEqualObjects(error.domain, expectedError.domain); \
  XCTAssertEqual(error.code, expectedError.code);            \
  XCTAssertEqualObjects(error.localizedDescription, expectedError.localizedDescription)

NS_ASSUME_NONNULL_BEGIN

@interface MPPAudioRecordTests : XCTestCase
@end

@implementation MPPAudioRecordTests

- (void)testInitAudioRecordFailsWithInvalidChannelCount {
  const NSInteger channelCount = 3;
  const NSInteger sampleRate = 8000;
  MPPAudioDataFormat *audioDataFormat =
      [[MPPAudioDataFormat alloc] initWithChannelCount:channelCount sampleRate:sampleRate];

  NSError *error = nil;
  const NSInteger bufferLength = 100;
  MPPAudioRecord *audioRecord = [[MPPAudioRecord alloc] initWithAudioDataFormat:audioDataFormat
                                                                   bufferLength:bufferLength
                                                                          error:&error];
  XCTAssertNil(audioRecord);

  NSError *expectedError = [NSError
      errorWithDomain:kExpectedErrorDomain
                 code:MPPTasksErrorCodeInvalidArgumentError
             userInfo:@{
               NSLocalizedDescriptionKey : @"The channel count provided does not match the "
                                           @"supported channel count. Only channels counts "
                                           @"in the range [1 : 2] are supported"
             }];

  AssertEqualErrors(error, expectedError);
}

- (void)testInitAudioRecordFailsWithInvalidBufferLength {
  const NSInteger channelCount = 2;
  const NSInteger sampleRate = 8000;
  MPPAudioDataFormat *audioDataFormat =
      [[MPPAudioDataFormat alloc] initWithChannelCount:channelCount sampleRate:sampleRate];

  NSError *error = nil;
  const NSInteger bufferLength = 101;
  MPPAudioRecord *audioRecord = [[MPPAudioRecord alloc] initWithAudioDataFormat:audioDataFormat
                                                                   bufferLength:bufferLength
                                                                          error:&error];
  XCTAssertNil(audioRecord);

  NSError *expectedError =
      [NSError errorWithDomain:kExpectedErrorDomain
                          code:MPPTasksErrorCodeInvalidArgumentError
                      userInfo:@{
                        NSLocalizedDescriptionKey :
                            [NSString stringWithFormat:@"The buffer length provided (%lu) is not a "
                                                       @"multiple of channel count(%lu).",
                                                       bufferLength, audioDataFormat.channelCount]
                      }];

  AssertEqualErrors(error, expectedError);
}

+ (MPPAudioRecord *)createAudioRecordWithChannelCount:(const NSInteger)channelCount
                                           sampleRate:(const NSInteger)sampleRate
                                         bufferLength:(const NSInteger)bufferLength {
  MPPAudioDataFormat *audioDataFormat =
      [[MPPAudioDataFormat alloc] initWithChannelCount:channelCount sampleRate:sampleRate];

  MPPAudioRecord *audioRecord = [[MPPAudioRecord alloc] initWithAudioDataFormat:audioDataFormat
                                                                   bufferLength:bufferLength
                                                                          error:nil];

  XCTAssertNotNil(audioRecord);

  return audioRecord;
}

@end

NS_ASSUME_NONNULL_END
