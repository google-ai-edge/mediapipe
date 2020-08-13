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

#import "FaceMeshGpuViewController.h"

#include "mediapipe/framework/formats/landmark.pb.h"

static NSString* const kGraphName = @"face_mesh_mobile_gpu";

static const char* kNumFacesInputSidePacket = "num_faces";
static const char* kLandmarksOutputStream = "multi_face_landmarks";

// Max number of faces to detect/process.
static const int kNumFaces = 1;

@implementation FaceMeshGpuViewController

#pragma mark - UIViewController methods

- (void)viewDidLoad {
  [super viewDidLoad];

  [self.mediapipeGraph setSidePacket:(mediapipe::MakePacket<int>(kNumFaces))
                               named:kNumFacesInputSidePacket];
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
      NSLog(@"[TS:%lld] No face landmarks", packet.Timestamp().Value());
      return;
    }
    const auto& multi_face_landmarks = packet.Get<std::vector<::mediapipe::NormalizedLandmarkList>>();
    NSLog(@"[TS:%lld] Number of face instances with landmarks: %lu", packet.Timestamp().Value(),
          multi_face_landmarks.size());
    for (int face_index = 0; face_index < multi_face_landmarks.size(); ++face_index) {
      const auto& landmarks = multi_face_landmarks[face_index];
      NSLog(@"\tNumber of landmarks for face[%d]: %d", face_index, landmarks.landmark_size());
      for (int i = 0; i < landmarks.landmark_size(); ++i) {
        NSLog(@"\t\tLandmark[%d]: (%f, %f, %f)", i, landmarks.landmark(i).x(),
              landmarks.landmark(i).y(), landmarks.landmark(i).z());
      }
    }
  }
}

@end
