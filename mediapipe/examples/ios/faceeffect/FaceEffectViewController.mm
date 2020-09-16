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

#import "FaceEffectViewController.h"

#import "mediapipe/objc/MPPCameraInputSource.h"
#import "mediapipe/objc/MPPGraph.h"
#import "mediapipe/objc/MPPLayerRenderer.h"

#include <utility>

#include "mediapipe/framework/formats/matrix_data.pb.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/modules/face_geometry/protos/face_geometry.pb.h"

static NSString* const kGraphName = @"face_effect_gpu";

static const char* kInputStream = "input_video";
static const char* kIsFacepaintEffectSelectedInputStream = "is_facepaint_effect_selected";
static const char* kOutputStream = "output_video";
static const char* kMultiFaceGeometryStream = "multi_face_geometry";
static const char* kVideoQueueLabel = "com.google.mediapipe.example.videoQueue";

static const int kMatrixTranslationZIndex = 14;

@interface FaceEffectViewController () <MPPGraphDelegate, MPPInputSourceDelegate>

// The MediaPipe graph currently in use. Initialized in viewDidLoad, started in viewWillAppear: and
// sent video frames on _videoQueue.
@property(nonatomic) MPPGraph* graph;

@end

@implementation FaceEffectViewController {
  /// Handle tap gestures.
  UITapGestureRecognizer* _tapGestureRecognizer;
  BOOL _isFacepaintEffectSelected;

  /// Handles camera access via AVCaptureSession library.
  MPPCameraInputSource* _cameraSource;

  /// Inform the user when camera is unavailable.
  IBOutlet UILabel* _noCameraLabel;
  /// Inform the user about how to switch between effects.
  UILabel* _effectSwitchingHintLabel;
  /// Display the camera preview frames.
  IBOutlet UIView* _liveView;
  /// Render frames in a layer.
  MPPLayerRenderer* _renderer;

  /// Process camera frames on this queue.
  dispatch_queue_t _videoQueue;
}

#pragma mark - Cleanup methods

- (void)dealloc {
  self.graph.delegate = nil;
  [self.graph cancel];
  // Ignore errors since we're cleaning up.
  [self.graph closeAllInputStreamsWithError:nil];
  [self.graph waitUntilDoneWithError:nil];
}

#pragma mark - MediaPipe graph methods

+ (MPPGraph*)loadGraphFromResource:(NSString*)resource {
  // Load the graph config resource.
  NSError* configLoadError = nil;
  NSBundle* bundle = [NSBundle bundleForClass:[self class]];
  if (!resource || resource.length == 0) {
    return nil;
  }
  NSURL* graphURL = [bundle URLForResource:resource withExtension:@"binarypb"];
  NSData* data = [NSData dataWithContentsOfURL:graphURL options:0 error:&configLoadError];
  if (!data) {
    NSLog(@"Failed to load MediaPipe graph config: %@", configLoadError);
    return nil;
  }

  // Parse the graph config resource into mediapipe::CalculatorGraphConfig proto object.
  mediapipe::CalculatorGraphConfig config;
  config.ParseFromArray(data.bytes, data.length);

  // Create MediaPipe graph with mediapipe::CalculatorGraphConfig proto object.
  MPPGraph* newGraph = [[MPPGraph alloc] initWithGraphConfig:config];
  [newGraph addFrameOutputStream:kOutputStream outputPacketType:MPPPacketTypePixelBuffer];
  [newGraph addFrameOutputStream:kMultiFaceGeometryStream outputPacketType:MPPPacketTypeRaw];
  return newGraph;
}

#pragma mark - UIViewController methods

- (void)viewDidLoad {
  [super viewDidLoad];

  _effectSwitchingHintLabel.hidden = YES;
  _tapGestureRecognizer = [[UITapGestureRecognizer alloc] initWithTarget:self
                                                                  action:@selector(handleTap)];
  [self.view addGestureRecognizer:_tapGestureRecognizer];

  // By default, render the glasses effect.
  _isFacepaintEffectSelected = NO;

  _renderer = [[MPPLayerRenderer alloc] init];
  _renderer.layer.frame = _liveView.layer.bounds;
  [_liveView.layer insertSublayer:_renderer.layer atIndex:0];
  _renderer.frameScaleMode = MPPFrameScaleModeFillAndCrop;
  _renderer.mirrored = NO;

  dispatch_queue_attr_t qosAttribute = dispatch_queue_attr_make_with_qos_class(
      DISPATCH_QUEUE_SERIAL, QOS_CLASS_USER_INTERACTIVE, /*relative_priority=*/0);
  _videoQueue = dispatch_queue_create(kVideoQueueLabel, qosAttribute);

  _cameraSource = [[MPPCameraInputSource alloc] init];
  [_cameraSource setDelegate:self queue:_videoQueue];
  _cameraSource.sessionPreset = AVCaptureSessionPresetHigh;
  _cameraSource.cameraPosition = AVCaptureDevicePositionFront;
  // The frame's native format is rotated with respect to the portrait orientation.
  _cameraSource.orientation = AVCaptureVideoOrientationPortrait;
  _cameraSource.videoMirrored = YES;

  self.graph = [[self class] loadGraphFromResource:kGraphName];
  self.graph.delegate = self;
  // Set maxFramesInFlight to a small value to avoid memory contention for real-time processing.
  self.graph.maxFramesInFlight = 2;
}

// In this application, there is only one ViewController which has no navigation to other view
// controllers, and there is only one View with live display showing the result of running the
// MediaPipe graph on the live video feed. If more view controllers are needed later, the graph
// setup/teardown and camera start/stop logic should be updated appropriately in response to the
// appearance/disappearance of this ViewController, as viewWillAppear: can be invoked multiple times
// depending on the application navigation flow in that case.
- (void)viewWillAppear:(BOOL)animated {
  [super viewWillAppear:animated];

  [_cameraSource requestCameraAccessWithCompletionHandler:^void(BOOL granted) {
    if (granted) {
      [self startGraphAndCamera];
      dispatch_async(dispatch_get_main_queue(), ^{
        _noCameraLabel.hidden = YES;
      });
    }
  }];
}

- (void)startGraphAndCamera {
  // Start running self.graph.
  NSError* error;
  if (![self.graph startWithError:&error]) {
    NSLog(@"Failed to start graph: %@", error);
  }

  // Start fetching frames from the camera.
  dispatch_async(_videoQueue, ^{
    [_cameraSource start];
  });
}

#pragma mark - UITapGestureRecognizer methods

// We use the tap gesture recognizer to switch between face effects. This allows users to try
// multiple pre-bundled face effects without a need to recompile the app.
- (void)handleTap {
  dispatch_async(_videoQueue, ^{
    _isFacepaintEffectSelected = !_isFacepaintEffectSelected;
  });
}

#pragma mark - MPPGraphDelegate methods

// Receives CVPixelBufferRef from the MediaPipe graph. Invoked on a MediaPipe worker thread.
- (void)mediapipeGraph:(MPPGraph*)graph
    didOutputPixelBuffer:(CVPixelBufferRef)pixelBuffer
              fromStream:(const std::string&)streamName {
  if (streamName == kOutputStream) {
    // Display the captured image on the screen.
    CVPixelBufferRetain(pixelBuffer);
    dispatch_async(dispatch_get_main_queue(), ^{
      _effectSwitchingHintLabel.hidden = NO;
      [_renderer renderPixelBuffer:pixelBuffer];
      CVPixelBufferRelease(pixelBuffer);
    });
  }
}

// Receives a raw packet from the MediaPipe graph. Invoked on a MediaPipe worker thread.
//
// This callback demonstrates how the output face geometry packet can be obtained and used in an
// iOS app. As an example, the Z-translation component of the face pose transform matrix is logged
// for each face being equal to the approximate distance away from the camera in centimeters.
- (void)mediapipeGraph:(MPPGraph*)graph
     didOutputPacket:(const ::mediapipe::Packet&)packet
          fromStream:(const std::string&)streamName {
  if (streamName == kMultiFaceGeometryStream) {
    if (packet.IsEmpty()) {
      NSLog(@"[TS:%lld] No face geometry", packet.Timestamp().Value());
      return;
    }

    const auto& multiFaceGeometry =
        packet.Get<std::vector<::mediapipe::face_geometry::FaceGeometry>>();
    NSLog(@"[TS:%lld] Number of face instances with geometry: %lu ", packet.Timestamp().Value(),
          multiFaceGeometry.size());
    for (int faceIndex = 0; faceIndex < multiFaceGeometry.size(); ++faceIndex) {
      const auto& faceGeometry = multiFaceGeometry[faceIndex];
      NSLog(@"\tApprox. distance away from camera for face[%d]: %.6f cm", faceIndex,
            -faceGeometry.pose_transform_matrix().packed_data(kMatrixTranslationZIndex));
    }
  }
}

#pragma mark - MPPInputSourceDelegate methods

// Must be invoked on _videoQueue.
- (void)processVideoFrame:(CVPixelBufferRef)imageBuffer
                timestamp:(CMTime)timestamp
               fromSource:(MPPInputSource*)source {
  if (source != _cameraSource) {
    NSLog(@"Unknown source: %@", source);
    return;
  }

  mediapipe::Timestamp graphTimestamp(static_cast<mediapipe::TimestampBaseType>(
      mediapipe::Timestamp::kTimestampUnitsPerSecond * CMTimeGetSeconds(timestamp)));

  mediapipe::Packet isFacepaintEffectSelectedPacket =
      mediapipe::MakePacket<bool>(_isFacepaintEffectSelected).At(graphTimestamp);

  [self.graph sendPixelBuffer:imageBuffer
                   intoStream:kInputStream
                   packetType:MPPPacketTypePixelBuffer
                    timestamp:graphTimestamp];

  // Alongside the input camera frame, we also send the `is_facepaint_effect_selected` boolean
  // packet to indicate which effect should be rendered on this frame.
  [self.graph movePacket:std::move(isFacepaintEffectSelectedPacket)
              intoStream:kIsFacepaintEffectSelectedInputStream
                   error:nil];
}

@end
