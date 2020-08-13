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

#import <UIKit/UIKit.h>

#import "mediapipe/objc/MPPCameraInputSource.h"
#import "mediapipe/objc/MPPGraph.h"
#import "mediapipe/objc/MPPLayerRenderer.h"
#import "mediapipe/objc/MPPPlayerInputSource.h"

typedef NS_ENUM(NSInteger, MediaPipeDemoSourceMode) {
  MediaPipeDemoSourceCamera,
  MediaPipeDemoSourceVideo
};

@interface CommonViewController : UIViewController <MPPGraphDelegate, MPPInputSourceDelegate>

// The MediaPipe graph currently in use. Initialized in viewDidLoad, started in
// viewWillAppear: and sent video frames on videoQueue.
@property(nonatomic) MPPGraph* mediapipeGraph;

// Handles camera access via AVCaptureSession library.
@property(nonatomic) MPPCameraInputSource* cameraSource;

// Provides data from a video.
@property(nonatomic) MPPPlayerInputSource* videoSource;

// The data source for the demo.
@property(nonatomic) MediaPipeDemoSourceMode sourceMode;

// Inform the user when camera is unavailable.
@property(nonatomic) IBOutlet UILabel* noCameraLabel;

// Display the camera preview frames.
@property(strong, nonatomic) IBOutlet UIView* liveView;

// Render frames in a layer.
@property(nonatomic) MPPLayerRenderer* renderer;

// Process camera frames on this queue.
@property(nonatomic) dispatch_queue_t videoQueue;

// Graph name.
@property(nonatomic) NSString* graphName;

// Graph input stream.
@property(nonatomic) const char* graphInputStream;

// Graph output stream.
@property(nonatomic) const char* graphOutputStream;

@end
