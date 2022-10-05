// Copyright 2020 The MediaPipe Authors.
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

#import "PoseTrackingViewController.h"

#include "mediapipe/framework/formats/landmark.pb.h"
#include "mediapipe/objc/solutions/posetracking_gpu/PoseTrackingOptions.h"
#include "mediapipe/objc/solutions/posetracking_gpu/PoseTracking.h"

static const char* kLandmarksOutputStream = "pose_landmarks";

@implementation PoseTrackingViewController

#pragma mark - UIViewController methods


- (void)viewDidLoad {
    
    
    [super viewDidLoad];

    // create pose tracking options
    PoseTrackingOptions* options =   [ [PoseTrackingOptions alloc] initWithShowLandmarks:true cameraRotation:0];
    // create pose tracking from options
    self.poseTracking = [[PoseTracking alloc] initWithPoseTrackingOptions:options];
    // render pose tracking to a UIView (self.liveView)
    self.poseTracking.renderer.layer.frame = self.liveView.layer.bounds;
    [self.liveView.layer addSublayer:self.poseTracking.renderer.layer];
    
    // create a block to run when PoseTrackingResults are available
    self.poseTracking.poseTrackingResultsListener = ^(PoseTrackingResults* results){
              NSLog(@"\tLandmark[%d]: (%f, %f, %f)", 0, results.landmarks[0].x,results.landmarks[0].y,results.landmarks[0].z);
        
    };
}


// In this application, there is only one ViewController which has no navigation to other view
// controllers, and there is only one View with live display showing the result of running the
// MediaPipe graph on the live video feed. If more view controllers are needed later, the graph
// setup/teardown and camera start/stop logic should be updated appropriately in response to the
// appearance/disappearance of this ViewController, as viewWillAppear: can be invoked multiple times
// depending on the application navigation flow in that case.
- (void)viewWillAppear:(BOOL)animated {
    [super viewWillAppear:animated];

    // create and set camera options
    self.cameraSource = [[MPPCameraInputSource alloc] init];
    self.cameraSource.sessionPreset = AVCaptureSessionPresetHigh;
    self.cameraSource.cameraPosition = AVCaptureDevicePositionBack;
    // When using the front camera, mirror the input for a more natural look.
    //self.cameraSource.videoMirrored = YES;

    // The frame's native format is rotated with respect to the portrait orientation.
    self.cameraSource.orientation = AVCaptureVideoOrientationPortrait;

    // request camera access permission
    [self.cameraSource requestCameraAccessWithCompletionHandler:^void(BOOL granted) {
      if (granted) {
          //start pose tracking
          [self.poseTracking startWithCamera:self.cameraSource];
      }
    }];
    
}

//#pragma mark - MPPGraphDelegate methods
//




@end
