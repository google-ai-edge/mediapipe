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

#import <AVFoundation/AVFoundation.h>

#import "mediapipe/objc/MPPInputSource.h"

/// A source that obtains video frames from the camera.
@interface MPPCameraInputSource : MPPInputSource

/// Whether we are allowed to use the camera.
@property(nonatomic, getter=isAuthorized, readonly) BOOL authorized;

/// Session preset to use for capturing.
@property(nonatomic) NSString *sessionPreset;

/// Which camera on an iOS device to use, assuming iOS device with more than one camera.
@property(nonatomic) AVCaptureDevicePosition cameraPosition;

// Whether to use depth data or not
@property(nonatomic) BOOL useDepth;

/// Whether to rotate video buffers with device rotation.
@property(nonatomic) BOOL autoRotateBuffers;

/// Whether to mirror the video or not.
@property(nonatomic) BOOL videoMirrored;

/// The camera intrinsic matrix.
@property(nonatomic, readonly) matrix_float3x3 cameraIntrinsicMatrix;

/// The capture session.
@property(nonatomic, readonly) AVCaptureSession *session;

/// The capture video preview layer.
@property(nonatomic, readonly) AVCaptureVideoPreviewLayer *videoPreviewLayer;

/// The orientation of camera frame buffers.
@property(nonatomic) AVCaptureVideoOrientation orientation;

/// Prompts the user to grant camera access and provides the result as a BOOL to a completion
/// handler. Should be called after [MPPCameraInputSource init] and before
/// [MPPCameraInputSource start]. If the user has previously granted or denied permission, this
/// method simply returns the saved response to the permission request.
- (void)requestCameraAccessWithCompletionHandler:(void (^_Nullable)(BOOL granted))handler;

@end
