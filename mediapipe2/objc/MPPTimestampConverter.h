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

#import <CoreMedia/CoreMedia.h>
#import <Foundation/Foundation.h>

#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/objc/util.h"

/// Helps convert a CMTime to a MediaPipe timestamp.
@interface MPPTimestampConverter : NSObject

/// The last timestamp returned by timestampForMediaTime:.
@property(nonatomic, readonly) mediapipe::Timestamp lastTimestamp;

/// Initializer.
- (instancetype)init NS_DESIGNATED_INITIALIZER;

/// Resets the object. After this method is called, we can return timestamps
/// that are lower than previously returned timestamps.
- (void)reset;

/// Converts a CMTime to a MediaPipe timestamp. This ensures that MediaPipe
/// timestamps
/// are always increasing: if the provided CMTime has gone backwards (e.g. if
/// it's from a
/// looping video), we shift all timestamps from that point on to keep the
/// output increasing.
/// This state is erased when reset is called.
- (mediapipe::Timestamp)timestampForMediaTime:(CMTime)mediaTime;

@end
