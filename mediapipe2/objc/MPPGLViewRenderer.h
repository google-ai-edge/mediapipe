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

/// Modes of rotation (clockwise) for input frames.
typedef NS_ENUM(int, MPPFrameRotation) {
  MPPFrameRotationNone,
  MPPFrameRotationCw90,
  MPPFrameRotationCw180,
  MPPFrameRotationCw270,
};

typedef NS_ENUM(int, MPPFrameScaleMode) {
  // Scale the frame up to fit the drawing area, preserving aspect ratio; may letterbox.
  MPPFrameScaleModeFit,
  // Scale the frame up to fill the drawing area, preserving aspect ratio; may crop.
  MPPFrameScaleModeFillAndCrop,
};

/// Renders frames in a GLKView.
@interface MPPGLViewRenderer : NSObject <GLKViewDelegate>

/// Rendering context for display.
@property(nonatomic) EAGLContext *glContext;

/// The frame to be rendered next. This pixel buffer must be unlocked, and
/// should not be modified after handing it to the renderer.
@property(atomic, retain) __attribute__((NSObject)) CVPixelBufferRef nextPixelBufferToRender;

/// When YES, the last drawn pixel buffer is retained by this object after it is drawn in the GLView
/// for which it is a delegate. Otherwise it is released after it has been rendered.
/// Set this property to YES when your GLView can be redrawn with the same pixel buffer, such as
/// during an animation.
@property(nonatomic, assign) BOOL retainsLastPixelBuffer;

/// Sets which way to rotate input frames before rendering them.
/// Default value is MPPFrameRotationNone.
/// Note that changing the transform property of a GLKView once rendering has
/// started causes problems inside GLKView. Instead, we perform the rotation
/// in our rendering code.
@property(nonatomic) MPPFrameRotation frameRotationMode;

/// Sets how to scale the frame within the view.
/// Default value is MPPFrameScaleModeFit.
@property(nonatomic) MPPFrameScaleMode frameScaleMode;

/// If YES, swap left and right. Useful for the front camera.
@property(nonatomic) BOOL mirrored;

/// Draws a pixel buffer to its context with the specified view size.
- (void)drawPixelBuffer:(CVPixelBufferRef)pixelBuffer
                  width:(GLfloat)viewWidth
                 height:(GLfloat)viewHeight;

@end
