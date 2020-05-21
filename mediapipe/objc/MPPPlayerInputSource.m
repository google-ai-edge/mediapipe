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

#import "MPPPlayerInputSource.h"
#import "mediapipe/objc/MPPDisplayLinkWeakTarget.h"

@implementation MPPPlayerInputSource {
  AVAsset* _video;
  AVPlayerItem* _videoItem;
  AVPlayer* _videoPlayer;
  AVPlayerItemVideoOutput* _videoOutput;
  CADisplayLink* _videoDisplayLink;
  MPPDisplayLinkWeakTarget* _displayLinkWeakTarget;
  id _videoEndObserver;
}

- (instancetype)initWithAVAsset:(AVAsset*)video {
  self = [super init];
  if (self) {
    _video = video;
    _videoItem = [AVPlayerItem playerItemWithAsset:_video];
    // Necessary to ensure the video's preferred transform is respected.
    _videoItem.videoComposition = [AVVideoComposition videoCompositionWithPropertiesOfAsset:_video];

    _videoOutput = [[AVPlayerItemVideoOutput alloc] initWithPixelBufferAttributes:@{
      (id)kCVPixelBufferPixelFormatTypeKey : @(kCVPixelFormatType_32BGRA),
      (id)kCVPixelBufferIOSurfacePropertiesKey : [NSDictionary dictionary]
    }];
    _videoOutput.suppressesPlayerRendering = YES;
    [_videoItem addOutput:_videoOutput];

    _displayLinkWeakTarget =
        [[MPPDisplayLinkWeakTarget alloc] initWithTarget:self selector:@selector(videoUpdate:)];

    _videoDisplayLink = [CADisplayLink displayLinkWithTarget:_displayLinkWeakTarget
                                                    selector:@selector(displayLinkCallback:)];
    _videoDisplayLink.paused = YES;
    [_videoDisplayLink addToRunLoop:[NSRunLoop mainRunLoop] forMode:NSRunLoopCommonModes];

    _videoPlayer = [AVPlayer playerWithPlayerItem:_videoItem];
    _videoPlayer.actionAtItemEnd = AVPlayerActionAtItemEndNone;
    NSNotificationCenter* center = [NSNotificationCenter defaultCenter];

    __weak typeof(self) weakSelf = self;
    _videoEndObserver = [center addObserverForName:AVPlayerItemDidPlayToEndTimeNotification
                                            object:_videoItem
                                             queue:nil
                                        usingBlock:^(NSNotification* note) {
                                          [weakSelf playerItemDidPlayToEnd:note];
                                        }];
  }
  return self;
}

- (void)start {
  [_videoPlayer play];
  _videoDisplayLink.paused = NO;
}

- (void)stop {
  _videoDisplayLink.paused = YES;
  [_videoPlayer pause];
}

- (BOOL)isRunning {
  return _videoPlayer.rate != 0.0;
}

- (void)videoUpdate:(CADisplayLink*)sender {
  CMTime timestamp = [_videoItem currentTime];
  if ([_videoOutput hasNewPixelBufferForItemTime:timestamp]) {
    CVPixelBufferRef pixelBuffer =
        [_videoOutput copyPixelBufferForItemTime:timestamp itemTimeForDisplay:nil];
    if (pixelBuffer)
      dispatch_async(self.delegateQueue, ^{
        if ([self.delegate respondsToSelector:@selector(processVideoFrame:timestamp:fromSource:)]) {
          [self.delegate processVideoFrame:pixelBuffer timestamp:timestamp fromSource:self];
        } else if ([self.delegate respondsToSelector:@selector(processVideoFrame:fromSource:)]) {
          [self.delegate processVideoFrame:pixelBuffer fromSource:self];
        }
        CFRelease(pixelBuffer);
      });
  }
}

- (void)dealloc {
  [[NSNotificationCenter defaultCenter] removeObserver:self];
  [_videoDisplayLink invalidate];
  _videoPlayer = nil;
}

#pragma mark - NSNotificationCenter / observer

- (void)playerItemDidPlayToEnd:(NSNotification*)notification {
  CMTime timestamp = [_videoItem currentTime];
  dispatch_async(self.delegateQueue, ^{
    if ([self.delegate respondsToSelector:@selector(videoDidPlayToEnd:)]) {
      [self.delegate videoDidPlayToEnd:timestamp];
    } else {
      // Default to loop if no delegate handler set.
      [_videoPlayer seekToTime:kCMTimeZero];
    }
  });
}

- (void)seekToTime:(CMTime)time tolerance:(CMTime)tolerance {
  [_videoPlayer seekToTime:time toleranceBefore:tolerance toleranceAfter:tolerance];
}

- (void)setPlaybackEndTime:(CMTime)time {
  _videoPlayer.currentItem.forwardPlaybackEndTime = time;
}

- (CMTime)currentPlayerTime {
  return _videoPlayer.currentTime;
}

@end
