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

#import <Foundation/Foundation.h>
#import <GLKit/GLKit.h>

#import "mediapipe/objc/MPPGLViewRenderer.h"

/// Renders frames in a Core Animation layer.
@interface MPPLayerRenderer : NSObject

@property(nonatomic, readonly) CAEAGLLayer *layer;

/// Updates the layer with a new pixel buffer.
- (void)renderPixelBuffer:(CVPixelBufferRef)pixelBuffer;

/// Sets which way to rotate input frames before rendering them.
/// Default value is MPPFrameRotationNone.
@property(nonatomic) MPPFrameRotation frameRotationMode;

/// Sets how to scale the frame within the layer.
/// Default value is MediaPipeFrameScaleScaleToFit.
@property(nonatomic) MPPFrameScaleMode frameScaleMode;

/// If YES, swap left and right. Useful for the front camera.
@property(nonatomic) BOOL mirrored;

@end
