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

#import "MPPCameraInputSource.h"

#import <UIKit/UIKit.h>

@interface MPPCameraInputSource () <AVCaptureVideoDataOutputSampleBufferDelegate,
                                        AVCaptureDepthDataOutputDelegate>
@end

@implementation MPPCameraInputSource {
  AVCaptureSession* _session;
  AVCaptureDeviceInput* _videoDeviceInput;
  AVCaptureVideoDataOutput* _videoDataOutput;
  AVCaptureDepthDataOutput* _depthDataOutput API_AVAILABLE(ios(11.0));
  AVCaptureDevice* _currentDevice;

  matrix_float3x3 _cameraIntrinsicMatrix;
  OSType _pixelFormatType;
  BOOL _autoRotateBuffers;
  BOOL _didReadCameraIntrinsicMatrix;
  BOOL _setupDone;
  BOOL _useDepth;
  BOOL _useCustomOrientation;
  BOOL _videoMirrored;
}

- (instancetype)init {
  self = [super init];
  if (self) {
    _cameraPosition = AVCaptureDevicePositionBack;
    _session = [[AVCaptureSession alloc] init];
    _pixelFormatType = kCVPixelFormatType_32BGRA;

    AVAuthorizationStatus status =
        [AVCaptureDevice authorizationStatusForMediaType:AVMediaTypeVideo];
    _authorized = status == AVAuthorizationStatusAuthorized;
  }
  return self;
}

- (void)setDelegate:(id<MPPInputSourceDelegate>)delegate queue:(dispatch_queue_t)queue {
  [super setDelegate:delegate queue:queue];
  // Note that _depthDataOutput and _videoDataOutput may not have been created yet. In that case,
  // this message to nil is ignored, and the delegate will be set later by setupCamera.
  [_videoDataOutput setSampleBufferDelegate:self queue:queue];
  [_depthDataOutput setDelegate:self callbackQueue:queue];
}

- (void)start {
  if (!_setupDone) [self setupCamera];
  if (_autoRotateBuffers) {
    [self enableAutoRotateBufferObserver:YES];
  }
  [_session startRunning];
}

- (void)stop {
  if (_autoRotateBuffers) {
    [self enableAutoRotateBufferObserver:NO];
  }
  [_session stopRunning];
}

- (BOOL)isRunning {
  return _session.isRunning;
}

- (void)setCameraPosition:(AVCaptureDevicePosition)cameraPosition {
  BOOL wasRunning = [self isRunning];
  if (wasRunning) {
    [self stop];
  }
  _cameraPosition = cameraPosition;
  _setupDone = NO;
  if (wasRunning) {
    [self start];
  }
}

- (void)setUseDepth:(BOOL)useDepth {
  if (useDepth == _useDepth) {
    return;
  }

  BOOL wasRunning = [self isRunning];
  if (wasRunning) {
    [self stop];
  }
  _useDepth = useDepth;
  _setupDone = NO;
  if (wasRunning) {
    [self start];
  }
}

- (void)setOrientation:(AVCaptureVideoOrientation)orientation {
  if (orientation == _orientation) {
    return;
  }

  BOOL wasRunning = [self isRunning];
  if (wasRunning) {
    [self stop];
  }

  _orientation = orientation;
  _useCustomOrientation = YES;
  _setupDone = NO;
  if (wasRunning) {
    [self start];
  }
}

- (void)setVideoMirrored:(BOOL)videoMirrored {
  if (videoMirrored == _videoMirrored) {
    return;
  }

  BOOL wasRunning = [self isRunning];
  if (wasRunning) {
    [self stop];
  }
  _videoMirrored = videoMirrored;
  _setupDone = NO;
  if (wasRunning) {
    [self start];
  }
}

- (void)setAutoRotateBuffers:(BOOL)autoRotateBuffers {
  if (autoRotateBuffers == _autoRotateBuffers) {
    return;  // State has not changed.
  }
  _autoRotateBuffers = autoRotateBuffers;
  if ([self isRunning]) {
    // Enable or disable observer this settings changes while this input source is running.
    [self enableAutoRotateBufferObserver:_autoRotateBuffers];
  }
}

- (void)enableAutoRotateBufferObserver:(BOOL)enable {
  if (enable) {
    [[NSNotificationCenter defaultCenter] addObserver:self
                                             selector:@selector(deviceOrientationChanged)
                                                 name:UIDeviceOrientationDidChangeNotification
                                               object:nil];
    // Trigger a device orientation change instead of waiting for the first change.
    [self deviceOrientationChanged];
  } else {
    [[NSNotificationCenter defaultCenter] removeObserver:self
                                                    name:UIDeviceOrientationDidChangeNotification
                                                  object:nil];
  }
}

- (OSType)pixelFormatType {
  return _pixelFormatType;
}

- (void)setPixelFormatType:(OSType)pixelFormatType {
  _pixelFormatType = pixelFormatType;
  if ([self isRunning]) {
    _videoDataOutput.videoSettings = @{(id)kCVPixelBufferPixelFormatTypeKey : @(_pixelFormatType)};
  }
}

#pragma mark - Camera-specific methods

- (NSString*)sessionPreset {
  return _session.sessionPreset;
}

- (void)setSessionPreset:(NSString*)sessionPreset {
  _session.sessionPreset = sessionPreset;
}

- (void)setupCamera {
  NSError* error = nil;

  if (_videoDeviceInput) {
    [_session removeInput:_videoDeviceInput];
  }

  AVCaptureDeviceType deviceType = AVCaptureDeviceTypeBuiltInWideAngleCamera;
  if (@available(iOS 11.1, *)) {
    if (_cameraPosition == AVCaptureDevicePositionFront && _useDepth) {
      deviceType = AVCaptureDeviceTypeBuiltInTrueDepthCamera;
    }
  }
  AVCaptureDeviceDiscoverySession* deviceDiscoverySession = [AVCaptureDeviceDiscoverySession
      discoverySessionWithDeviceTypes:@[ deviceType ]
                            mediaType:AVMediaTypeVideo
                             position:_cameraPosition];
  AVCaptureDevice* videoDevice =
      [deviceDiscoverySession devices]
          ? [deviceDiscoverySession devices].firstObject
          : [AVCaptureDevice defaultDeviceWithMediaType:AVMediaTypeVideo];
  _videoDeviceInput = [AVCaptureDeviceInput deviceInputWithDevice:videoDevice error:&error];
  if (error) {
    NSLog(@"%@", error);
    return;
  }
  [_session addInput:_videoDeviceInput];

  if (!_videoDataOutput) {
    _videoDataOutput = [[AVCaptureVideoDataOutput alloc] init];
    [_session addOutput:_videoDataOutput];

    // Set this when we have a handler.
    if (self.delegateQueue)
      [_videoDataOutput setSampleBufferDelegate:self queue:self.delegateQueue];
    _videoDataOutput.alwaysDiscardsLateVideoFrames = YES;

    // Only a few pixel formats are available for capture output:
    //   kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange,
    //   kCVPixelFormatType_420YpCbCr8BiPlanarFullRange,
    //   kCVPixelFormatType_32BGRA.
    _videoDataOutput.videoSettings = @{(id)kCVPixelBufferPixelFormatTypeKey : @(_pixelFormatType)};
  }

  // Remove Old Depth Depth
  if (_depthDataOutput) {
    [_session removeOutput:_depthDataOutput];
  }

  if (@available(iOS 11.1, *)) {
    if (_useDepth) {
      // Add Depth Output
      _depthDataOutput = [[AVCaptureDepthDataOutput alloc] init];
      _depthDataOutput.alwaysDiscardsLateDepthData = YES;
      if ([_session canAddOutput:_depthDataOutput]) {
        [_session addOutput:_depthDataOutput];

        AVCaptureConnection* connection =
            [_depthDataOutput connectionWithMediaType:AVMediaTypeDepthData];

        // Set this when we have a handler.
        if (self.delegateQueue) {
          [_depthDataOutput setDelegate:self callbackQueue:self.delegateQueue];
        }
      } else {
        _depthDataOutput = nil;
      }
    }
  }

  if (_useCustomOrientation) {
    AVCaptureConnection* connection = [_videoDataOutput connectionWithMediaType:AVMediaTypeVideo];
    connection.videoOrientation = _orientation;
  }

  if (@available(iOS 11.0, *)) {
    AVCaptureConnection* connection = [_videoDataOutput connectionWithMediaType:AVMediaTypeVideo];
    if ([connection isCameraIntrinsicMatrixDeliverySupported]) {
      [connection setCameraIntrinsicMatrixDeliveryEnabled:YES];
    }
  }

  if (_videoMirrored) {
    AVCaptureConnection* connection = [_videoDataOutput connectionWithMediaType:AVMediaTypeVideo];
    connection.videoMirrored = _videoMirrored;
  }

  _setupDone = YES;
}

- (void)requestCameraAccessWithCompletionHandler:(void (^)(BOOL))handler {
  [AVCaptureDevice requestAccessForMediaType:AVMediaTypeVideo
                           completionHandler:^(BOOL granted) {
                             _authorized = granted;
                             if (handler) {
                               handler(granted);
                             }
                           }];
}

#pragma mark - AVCaptureVideoDataOutputSampleBufferDelegate methods

// Receives frames from the camera. Invoked on self.frameHandlerQueue.
- (void)captureOutput:(AVCaptureOutput*)captureOutput
    didOutputSampleBuffer:(CMSampleBufferRef)sampleBuffer
           fromConnection:(AVCaptureConnection*)connection {
  if (@available(iOS 11.0, *)) {
    if (!_didReadCameraIntrinsicMatrix) {
      // Get camera intrinsic matrix.
      CFTypeRef cameraIntrinsicData =
          CMGetAttachment(sampleBuffer, kCMSampleBufferAttachmentKey_CameraIntrinsicMatrix, nil);
      if (cameraIntrinsicData != nil) {
        CFDataRef cfdr = (CFDataRef)cameraIntrinsicData;
        matrix_float3x3* intrinsicMatrix = (matrix_float3x3*)(CFDataGetBytePtr(cfdr));
        if (intrinsicMatrix != nil) {
          _cameraIntrinsicMatrix = *intrinsicMatrix;
        }
      }
      _didReadCameraIntrinsicMatrix = YES;
    }
  }
  CVPixelBufferRef imageBuffer = CMSampleBufferGetImageBuffer(sampleBuffer);
  CMTime timestamp = CMSampleBufferGetPresentationTimeStamp(sampleBuffer);
  if ([self.delegate respondsToSelector:@selector(processVideoFrame:timestamp:fromSource:)]) {
    [self.delegate processVideoFrame:imageBuffer timestamp:timestamp fromSource:self];
  } else if ([self.delegate respondsToSelector:@selector(processVideoFrame:fromSource:)]) {
    [self.delegate processVideoFrame:imageBuffer fromSource:self];
  }
}

#pragma mark - AVCaptureDepthDataOutputDelegate methods

// Receives depth frames from the camera. Invoked on self.frameHandlerQueue.
- (void)depthDataOutput:(AVCaptureDepthDataOutput*)output
     didOutputDepthData:(AVDepthData*)depthData
              timestamp:(CMTime)timestamp
             connection:(AVCaptureConnection*)connection API_AVAILABLE(ios(11.0)) {
  if (depthData.depthDataType != kCVPixelFormatType_DepthFloat32) {
    depthData = [depthData depthDataByConvertingToDepthDataType:kCVPixelFormatType_DepthFloat32];
  }
  CVPixelBufferRef depthBuffer = depthData.depthDataMap;
  [self.delegate processDepthData:depthData timestamp:timestamp fromSource:self];
}

#pragma mark - NSNotificationCenter event handlers

- (void)deviceOrientationChanged {
  AVCaptureConnection* connection = [_videoDataOutput connectionWithMediaType:AVMediaTypeVideo];
  connection.videoOrientation = (AVCaptureVideoOrientation)[UIDevice currentDevice].orientation;
}

@end
