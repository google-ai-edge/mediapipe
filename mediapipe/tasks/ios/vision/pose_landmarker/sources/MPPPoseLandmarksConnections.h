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

#import <Foundation/Foundation.h>
#import "mediapipe/tasks/ios/components/containers/sources/MPPConnection.h"

NS_ASSUME_NONNULL_BEGIN

NSArray<MPPConnection *> *const MPPPoseLandmarksConnections = @[
  [[MPPConnection alloc] initWithStart:0 end:1],   [[MPPConnection alloc] initWithStart:1 end:2],
  [[MPPConnection alloc] initWithStart:2 end:3],   [[MPPConnection alloc] initWithStart:3 end:7],
  [[MPPConnection alloc] initWithStart:0 end:4],   [[MPPConnection alloc] initWithStart:4 end:5],
  [[MPPConnection alloc] initWithStart:5 end:6],   [[MPPConnection alloc] initWithStart:6 end:8],
  [[MPPConnection alloc] initWithStart:9 end:10],  [[MPPConnection alloc] initWithStart:11 end:12],
  [[MPPConnection alloc] initWithStart:11 end:13], [[MPPConnection alloc] initWithStart:13 end:15],
  [[MPPConnection alloc] initWithStart:15 end:17], [[MPPConnection alloc] initWithStart:15 end:19],
  [[MPPConnection alloc] initWithStart:15 end:21], [[MPPConnection alloc] initWithStart:17 end:19],
  [[MPPConnection alloc] initWithStart:12 end:14], [[MPPConnection alloc] initWithStart:14 end:16],
  [[MPPConnection alloc] initWithStart:16 end:18], [[MPPConnection alloc] initWithStart:16 end:20],
  [[MPPConnection alloc] initWithStart:16 end:22], [[MPPConnection alloc] initWithStart:18 end:20],
  [[MPPConnection alloc] initWithStart:11 end:23], [[MPPConnection alloc] initWithStart:12 end:24],
  [[MPPConnection alloc] initWithStart:23 end:24], [[MPPConnection alloc] initWithStart:23 end:25],
  [[MPPConnection alloc] initWithStart:24 end:26], [[MPPConnection alloc] initWithStart:25 end:27],
  [[MPPConnection alloc] initWithStart:26 end:28], [[MPPConnection alloc] initWithStart:27 end:29],
  [[MPPConnection alloc] initWithStart:28 end:30], [[MPPConnection alloc] initWithStart:29 end:31],
  [[MPPConnection alloc] initWithStart:30 end:32], [[MPPConnection alloc] initWithStart:27 end:31],
  [[MPPConnection alloc] initWithStart:28 end:32]
];

NS_ASSUME_NONNULL_END
