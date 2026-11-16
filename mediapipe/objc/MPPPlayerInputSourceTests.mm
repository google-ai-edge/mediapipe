// Copyright 2026 The MediaPipe Authors.
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

#include <atomic>

#import "mediapipe/objc/MPPInputSource.h"
#import "mediapipe/objc/MPPPlayerInputSource.h"

namespace {

// Pulls every audio sample of `asset` through `audioMix`, which runs its processing tap inline.
BOOL ReadAudioWithMix(AVAsset* asset, AVAudioMix* audioMix) {
  AVAssetReader* reader = [AVAssetReader assetReaderWithAsset:asset error:nil];
  AVAssetReaderAudioMixOutput* output = [AVAssetReaderAudioMixOutput
      assetReaderAudioMixOutputWithAudioTracks:[asset tracksWithMediaType:AVMediaTypeAudio]
                                 audioSettings:nil];
  output.audioMix = audioMix;
  [reader addOutput:output];
  [reader startReading];
  while (CMSampleBufferRef buffer = [output copyNextSampleBuffer]) {
    CFRelease(buffer);
  }
  return reader.status == AVAssetReaderStatusCompleted;
}

}  // namespace

@interface MPPPlayerInputSourceTestDelegate : NSObject <MPPInputSourceDelegate>
@property(nonatomic, readonly) int audioPacketCallbacks;
@end

@implementation MPPPlayerInputSourceTestDelegate {
  std::atomic<int> _audioPacketCallbacks;
}
- (int)audioPacketCallbacks {
  return _audioPacketCallbacks.load(std::memory_order_relaxed);
}
- (void)processAudioPacket:(const AudioBufferList*)audioPacket
                 numFrames:(CMItemCount)numFrames
                 timestamp:(CMTime)timestamp
                fromSource:(MPPInputSource*)source {
  _audioPacketCallbacks.fetch_add(1, std::memory_order_relaxed);
}
@end

@interface MPPPlayerInputSourceTests : XCTestCase
@end

@implementation MPPPlayerInputSourceTests

- (void)testStopAndDeallocDetachAudioMixAndSilenceRetainedTap {
  NSURL* url = [[NSBundle bundleForClass:[self class]] URLForResource:@"format_MP4_AVC720P_AAC"
                                                        withExtension:@"video"];
  XCTAssertNotNil(url);
  AVAsset* asset = [AVURLAsset URLAssetWithURL:url
                                       options:@{@"AVURLAssetOutOfBandMIMETypeKey" : @"video/mp4"}];
  MPPPlayerInputSourceTestDelegate* delegate =
      [[MPPPlayerInputSourceTestDelegate alloc] init];
  dispatch_queue_t queue = dispatch_queue_create("delegate", DISPATCH_QUEUE_SERIAL);
  AVPlayerItem* playerItem = nil;
  AVAudioMix* retainedMix = nil;
  __weak MPPPlayerInputSource* weakSource = nil;

  @autoreleasepool {
    MPPPlayerInputSource* inputSource = [[MPPPlayerInputSource alloc] initWithAVAsset:asset
                                                                       audioProcessingEnabled:YES];
    weakSource = inputSource;
    [inputSource setDelegate:delegate queue:queue];
    playerItem = [inputSource valueForKey:@"_videoItem"];
    XCTAssertTrue(ReadAudioWithMix(asset, playerItem.audioMix));
    XCTAssertGreaterThan(delegate.audioPacketCallbacks, 0);

    // `stop` detaches and silences the tap, and `start` attaches a fresh one.
    AVAudioMix* stoppedMix = playerItem.audioMix;
    [inputSource stop];
    XCTAssertNil(playerItem.audioMix);
    const int callbacksAfterStop = delegate.audioPacketCallbacks;
    XCTAssertTrue(ReadAudioWithMix(asset, stoppedMix));
    XCTAssertEqual(delegate.audioPacketCallbacks, callbacksAfterStop);
    [inputSource start];
    XCTAssertTrue(ReadAudioWithMix(asset, playerItem.audioMix));
    XCTAssertGreaterThan(delegate.audioPacketCallbacks, callbacksAfterStop);
    retainedMix = playerItem.audioMix;
    inputSource = nil;
  }
  // The player's audio thread can briefly hold the source while a tap callback is in flight.
  NSDate* deadline = [NSDate dateWithTimeIntervalSinceNow:2.0];
  while ((weakSource != nil || playerItem.audioMix != nil) && deadline.timeIntervalSinceNow > 0) {
    [[NSRunLoop currentRunLoop] runUntilDate:[NSDate dateWithTimeIntervalSinceNow:0.005]];
  }
  // An empty block drains `queue`, so no delegate callback is still in flight.
  dispatch_sync(queue, ^{
                });
  XCTAssertNil(weakSource);
  XCTAssertNil(playerItem.audioMix);

  // A tap that outlives its source must not call back into it.
  const int callbacksBeforeRead = delegate.audioPacketCallbacks;
  XCTAssertTrue(ReadAudioWithMix(asset, retainedMix));
  XCTAssertEqual(delegate.audioPacketCallbacks, callbacksBeforeRead);
}

@end
