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

#import "HandTrackingViewController.h"

#include "mediapipe/framework/formats/landmark.pb.h"

static const char* kLandmarksOutputStream = "hand_landmarks";

@implementation HandTrackingViewController

#pragma mark - UIViewController methods

- (void)viewDidLoad {
  [super viewDidLoad];

  [self.mediapipeGraph addFrameOutputStream:kLandmarksOutputStream
                           outputPacketType:MPPPacketTypeRaw];
}

#pragma mark - MPPGraphDelegate methods

// Receives a raw packet from the MediaPipe graph. Invoked on a MediaPipe worker thread.
- (void)mediapipeGraph:(MPPGraph*)graph
     didOutputPacket:(const ::mediapipe::Packet&)packet
          fromStream:(const std::string&)streamName {
  if (streamName == kLandmarksOutputStream) {
    if (packet.IsEmpty()) {
      NSLog(@"[TS:%lld] No hand landmarks", packet.Timestamp().Value());
      return;
    }
    const auto& landmarks = packet.Get<::mediapipe::NormalizedLandmarkList>();
    NSLog(@"[TS:%lld] Number of landmarks on hand: %d", packet.Timestamp().Value(),
          landmarks.landmark_size());
    for (int i = 0; i < landmarks.landmark_size(); ++i) {
      NSLog(@"\tLandmark[%d]: (%f, %f, %f)", i, landmarks.landmark(i).x(),
            landmarks.landmark(i).y(), landmarks.landmark(i).z());
    }
  }
}

@end
