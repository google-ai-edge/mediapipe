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

#import "IrisTrackingViewController.h"

#include "mediapipe/framework/formats/landmark.pb.h"

static const char* kLandmarksOutputStream = "iris_landmarks";

@implementation IrisTrackingViewController {
  /// Input side packet for focal length parameter.
  std::map<std::string, mediapipe::Packet> _input_side_packets;
  mediapipe::Packet _focal_length_side_packet;
}

#pragma mark - UIViewController methods

- (void)viewDidLoad {
  [super viewDidLoad];

  [self.mediapipeGraph addFrameOutputStream:kLandmarksOutputStream
                           outputPacketType:MPPPacketTypeRaw];
  _focal_length_side_packet =
      mediapipe::MakePacket<std::unique_ptr<float>>(absl::make_unique<float>(0.0));
  _input_side_packets = {
      {"focal_length_pixel", _focal_length_side_packet},
  };
  [self.mediapipeGraph addSidePackets:_input_side_packets];
}

#pragma mark - MPPGraphDelegate methods

// Receives a raw packet from the MediaPipe graph. Invoked on a MediaPipe worker thread.
- (void)mediapipeGraph:(MPPGraph*)graph
     didOutputPacket:(const ::mediapipe::Packet&)packet
          fromStream:(const std::string&)streamName {
  if (streamName == kLandmarksOutputStream) {
    if (packet.IsEmpty()) {
      NSLog(@"[TS:%lld] No iris landmarks", packet.Timestamp().Value());
      return;
    }
    const auto& landmarks = packet.Get<::mediapipe::NormalizedLandmarkList>();
    NSLog(@"[TS:%lld] Number of landmarks on iris: %d", packet.Timestamp().Value(),
          landmarks.landmark_size());
    for (int i = 0; i < landmarks.landmark_size(); ++i) {
      NSLog(@"\tLandmark[%d]: (%f, %f, %f)", i, landmarks.landmark(i).x(),
            landmarks.landmark(i).y(), landmarks.landmark(i).z());
    }
  }
}

#pragma mark - MPPInputSourceDelegate methods

// Must be invoked on _videoQueue.
- (void)processVideoFrame:(CVPixelBufferRef)imageBuffer
                timestamp:(CMTime)timestamp
               fromSource:(MPPInputSource*)source {
  if (source != self.cameraSource) {
    NSLog(@"Unknown source: %@", source);
    return;
  }

  // TODO: This is a temporary solution. Need to verify whether the focal length is
  // constant. In that case, we need to use input stream instead of using side packet.
  *(_input_side_packets["focal_length_pixel"].Get<std::unique_ptr<float>>()) =
      self.cameraSource.cameraIntrinsicMatrix.columns[0][0];
  [self.mediapipeGraph sendPixelBuffer:imageBuffer
                            intoStream:self.graphInputStream
                            packetType:MPPPacketTypePixelBuffer];
}

@end
