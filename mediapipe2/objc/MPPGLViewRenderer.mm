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

#import "MPPGLViewRenderer.h"

#include <OpenGLES/ES2/gl.h>
#include <OpenGLES/ES2/glext.h>
#include <algorithm>
#include <libkern/OSAtomic.h>

#include "absl/memory/memory.h"
#include "mediapipe/gpu/gl_quad_renderer.h"
#include "mediapipe/gpu/gl_simple_shaders.h"
#include "mediapipe/gpu/shader_util.h"

#import "mediapipe/objc/NSError+util_status.h"
#import "GTMDefines.h"

@implementation MPPGLViewRenderer {
  /// Used to create textures for the pixel buffers to be rendered.
  /// The use of this class allows the GPU to access the buffer's memory directly,
  /// without a memory copy.
  CVOpenGLESTextureCacheRef _textureCache;

  /// Internal renderer.
  std::unique_ptr<mediapipe::QuadRenderer> renderer_;

  /// Used to synchronize access to _nextPixelBufferToRender.
  OSSpinLock _bufferLock;
  volatile CVPixelBufferRef _nextPixelBufferToRender;
}

- (instancetype)init {
  self = [super init];
  if (self) {
    _glContext = [[EAGLContext alloc] initWithAPI:kEAGLRenderingAPIOpenGLES2];
    _bufferLock = OS_SPINLOCK_INIT;
    _frameRotationMode = MPPFrameRotationNone;
    _frameScaleMode = MPPFrameScaleModeFit;
  }
  return self;
}

- (void)dealloc {
  if (_textureCache) CFRelease(_textureCache);
  CVPixelBufferRelease(_nextPixelBufferToRender);
  // Fixes crash during dealloc that only happens in iOS 9. More info at b/67095363.
  if ([EAGLContext currentContext] == _glContext) {
    [EAGLContext setCurrentContext:nil];
  }
}

- (CVPixelBufferRef)nextPixelBufferToRender {
  OSSpinLockLock(&_bufferLock);
  CVPixelBufferRef buffer = _nextPixelBufferToRender;
  OSSpinLockUnlock(&_bufferLock);
  return buffer;
}

- (void)setNextPixelBufferToRender:(CVPixelBufferRef)buffer {
  OSSpinLockLock(&_bufferLock);
  if (_nextPixelBufferToRender != buffer) {
    CVPixelBufferRelease(_nextPixelBufferToRender);
    _nextPixelBufferToRender = buffer;
    CVPixelBufferRetain(_nextPixelBufferToRender);
  }
  OSSpinLockUnlock(&_bufferLock);
}

- (void)setupGL {
  CVReturn err;
  err = CVOpenGLESTextureCacheCreate(kCFAllocatorDefault, NULL, _glContext, NULL, &_textureCache);
  _GTMDevAssert(err == kCVReturnSuccess,
                @"CVOpenGLESTextureCacheCreate failed: %d", err);

  renderer_ = absl::make_unique<mediapipe::QuadRenderer>();
  auto status = renderer_->GlSetup();
  _GTMDevAssert(status.ok(),
                @"renderer setup failed: %@", [NSError gus_errorWithStatus:status]);
}

mediapipe::FrameScaleMode InternalScaleMode(MPPFrameScaleMode mode) {
  switch (mode) {
    case MPPFrameScaleModeFit:
      return mediapipe::FrameScaleMode::kFit;
    case MPPFrameScaleModeFillAndCrop:
      return mediapipe::FrameScaleMode::kFillAndCrop;
  }
}

mediapipe::FrameRotation InternalRotationMode(MPPFrameRotation rot) {
  switch (rot) {
    case MPPFrameRotationNone:
      return mediapipe::FrameRotation::kNone;
    case MPPFrameRotationCw90:
      return mediapipe::FrameRotation::k90;
    case MPPFrameRotationCw180:
      return mediapipe::FrameRotation::k180;
    case MPPFrameRotationCw270:
      return mediapipe::FrameRotation::k270;
  }
}

- (void)drawPixelBuffer:(CVPixelBufferRef)pixelBuffer
                  width:(GLfloat)viewWidth
                 height:(GLfloat)viewHeight {
  if (!_textureCache) [self setupGL];

  size_t frameWidth = CVPixelBufferGetWidth(pixelBuffer);
  size_t frameHeight = CVPixelBufferGetHeight(pixelBuffer);
  CVOpenGLESTextureRef texture = NULL;
  OSType pixelFormat = CVPixelBufferGetPixelFormatType(pixelBuffer);
  CVReturn error;
  if (pixelFormat == kCVPixelFormatType_OneComponent8) {
    error = CVOpenGLESTextureCacheCreateTextureFromImage(
        kCFAllocatorDefault, _textureCache, pixelBuffer, NULL,
        GL_TEXTURE_2D, GL_LUMINANCE, (GLsizei)frameWidth, (GLsizei)frameHeight,
        GL_LUMINANCE, GL_UNSIGNED_BYTE, 0, &texture);
  } else if (pixelFormat == kCVPixelFormatType_OneComponent32Float) {
    error = CVOpenGLESTextureCacheCreateTextureFromImage(
        kCFAllocatorDefault, _textureCache, pixelBuffer, NULL,
        GL_TEXTURE_2D, GL_LUMINANCE, (GLsizei)frameWidth, (GLsizei)frameHeight,
        GL_LUMINANCE, GL_FLOAT, 0, &texture);
  } else {
    error = CVOpenGLESTextureCacheCreateTextureFromImage(
        kCFAllocatorDefault, _textureCache, pixelBuffer, NULL,
        GL_TEXTURE_2D, GL_RGBA, (GLsizei)frameWidth, (GLsizei)frameHeight,
        GL_BGRA, GL_UNSIGNED_BYTE, 0, &texture);
  }
  _GTMDevAssert(error == kCVReturnSuccess,
                @"CVOpenGLESTextureCacheCreateTextureFromImage failed: %d", error);

  glClear(GL_COLOR_BUFFER_BIT);

  glActiveTexture(GL_TEXTURE1);
  glBindTexture(CVOpenGLESTextureGetTarget(texture), CVOpenGLESTextureGetName(texture));
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

  // Note: we perform a vertical flip by swapping the top and bottom coordinates.
  // CVPixelBuffers have a top left origin and OpenGL has a bottom left origin.
  auto status = renderer_->GlRender(
      frameWidth, frameHeight, viewWidth, viewHeight,
      InternalScaleMode(_frameScaleMode),
      InternalRotationMode(_frameRotationMode),
      _mirrored, /*flip_vertical=*/false, /*flip_texture=*/true);
  _GTMDevAssert(status.ok(),
                @"render failed: %@", [NSError gus_errorWithStatus:status]);

  glBindTexture(CVOpenGLESTextureGetTarget(texture), 0);
  glBindTexture(GL_TEXTURE_2D, 0 );
  CFRelease(texture);
  CVOpenGLESTextureCacheFlush(_textureCache, 0);
}

#pragma mark - GLKViewDelegate

- (void)glkView:(GLKView *)view drawInRect:(CGRect)rect {
  CVPixelBufferRef pixelBuffer = NULL;

  OSSpinLockLock(&_bufferLock);
  pixelBuffer = _nextPixelBufferToRender;
  if (_retainsLastPixelBuffer) {
    CVPixelBufferRetain(pixelBuffer);
  } else {
    _nextPixelBufferToRender = NULL;
  }
  OSSpinLockUnlock(&_bufferLock);

  if (!pixelBuffer) return;

  [self drawPixelBuffer:pixelBuffer width:view.drawableWidth height:view.drawableHeight];
  CVPixelBufferRelease(pixelBuffer);
}

@end
