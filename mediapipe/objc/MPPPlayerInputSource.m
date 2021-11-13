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

#import <CoreVideo/CoreVideo.h>

#import "MPPPlayerInputSource.h"
#if !TARGET_OS_OSX
#import "mediapipe/objc/MPPDisplayLinkWeakTarget.h"
#endif

@implementation MPPPlayerInputSource {
  AVAsset* _video;
  AVPlayerItem* _videoItem;
  AVPlayer* _videoPlayer;
  AVPlayerItemVideoOutput* _videoOutput;
#if !TARGET_OS_OSX
  CADisplayLink* _videoDisplayLink;
  MPPDisplayLinkWeakTarget* _displayLinkWeakTarget;
#else
  CVDisplayLinkRef _videoDisplayLink;
#endif  // TARGET_OS_OSX
  id _videoEndObserver;
  id _audioInterruptedObserver;
  BOOL _playing;
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

#if !TARGET_OS_OSX
    _displayLinkWeakTarget =
        [[MPPDisplayLinkWeakTarget alloc] initWithTarget:self selector:@selector(videoUpdate:)];

    _videoDisplayLink = [CADisplayLink displayLinkWithTarget:_displayLinkWeakTarget
                                                    selector:@selector(displayLinkCallback:)];
    _videoDisplayLink.paused = YES;
    [_videoDisplayLink addToRunLoop:[NSRunLoop mainRunLoop] forMode:NSRunLoopCommonModes];
#else
    CGDirectDisplayID displayID = CGMainDisplayID();
    CVReturn error = CVDisplayLinkCreateWithCGDisplay(displayID, &_videoDisplayLink);
    if (error) {
      _videoDisplayLink = NULL;
    }
    CVDisplayLinkStop(_videoDisplayLink);
    CVDisplayLinkSetOutputCallback(_videoDisplayLink, renderCallback, (__bridge void*)self);
#endif  // TARGET_OS_OSX
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
    _audioInterruptedObserver = [center addObserverForName:AVAudioSessionInterruptionNotification
                                                    object:nil
                                                     queue:nil
                                                usingBlock:^(NSNotification* note) {
                                                  [weakSelf audioSessionInterruption:note];
                                                }];
  }
  return self;
}

- (void)start {
  [_videoPlayer play];
  _playing = YES;
#if !TARGET_OS_OSX
  _videoDisplayLink.paused = NO;
#else
  CVDisplayLinkStart(_videoDisplayLink);
#endif
}

- (void)stop {
#if !TARGET_OS_OSX
  _videoDisplayLink.paused = YES;
#else
  CVDisplayLinkStop(_videoDisplayLink);
#endif
  [_videoPlayer pause];
  _playing = NO;
}

- (BOOL)isRunning {
  return _videoPlayer.rate != 0.0;
}

#if !TARGET_OS_OSX
- (void)videoUpdate:(CADisplayLink*)sender {
  [self videoUpdateIfNeeded];
}
#else
static CVReturn renderCallback(CVDisplayLinkRef displayLink, const CVTimeStamp* inNow,
                               const CVTimeStamp* inOutputTime, CVOptionFlags flagsIn,
                               CVOptionFlags* flagsOut, void* displayLinkContext) {
  [(__bridge MPPPlayerInputSource*)displayLinkContext videoUpdateIfNeeded];
  return kCVReturnSuccess;
}
#endif  // TARGET_OS_OSX

- (void)videoUpdateIfNeeded {
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
  } else if (
#if !TARGET_OS_OSX
             !_videoDisplayLink.paused &&
#endif
             _videoPlayer.rate == 0) {
    // The video might be paused by the operating system fo other reasons not catched by the context
    // of an interruption. If this condition happens the @c _videoDisplayLink will not have a
    // paused state, while the _videoPlayer will have rate 0 AKA paused. In this scenario we restart
    // the video playback.
    [_videoPlayer play];
  }
}

- (void)dealloc {
  [[NSNotificationCenter defaultCenter] removeObserver:self];
#if !TARGET_OS_OSX
  [_videoDisplayLink invalidate];
#else
  CVDisplayLinkRelease(_videoDisplayLink);
#endif
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

- (void)audioSessionInterruption:(NSNotification*)notification {
  if ([notification.userInfo[AVAudioSessionInterruptionTypeKey] intValue] ==
      AVAudioSessionInterruptionTypeEnded) {
    if ([notification.userInfo[AVAudioSessionInterruptionOptionKey] intValue] ==
        AVAudioSessionInterruptionOptionShouldResume && _playing) {
      // AVVideoPlayer does not automatically resume on this notification.
      [_videoPlayer play];
    }
  }
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
