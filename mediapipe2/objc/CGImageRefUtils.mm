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

#import "CGImageRefUtils.h"

#import <Foundation/Foundation.h>
#import "mediapipe/objc/CFHolder.h"
#import "mediapipe/objc/NSError+util_status.h"
#import "mediapipe/objc/util.h"

#include "mediapipe/framework/port/status.h"

CGImageRef CreateCGImageFromCVPixelBuffer(CVPixelBufferRef imageBuffer, NSError **error) {
  CFHolder<CGImageRef> cg_image_holder;
  absl::Status status = CreateCGImageFromCVPixelBuffer(imageBuffer, &cg_image_holder);
  if (!status.ok()) {
    *error = [NSError gus_errorWithStatus:status];
    return nil;
  }
  CGImageRef cg_image = *cg_image_holder;
  CGImageRetain(cg_image);
  return cg_image;
}

CVPixelBufferRef CreateCVPixelBufferFromCGImage(CGImageRef image, NSError **error) {
  CFHolder<CVPixelBufferRef> pixel_buffer_holder;
  absl::Status status = CreateCVPixelBufferFromCGImage(image, &pixel_buffer_holder);
  if (!status.ok()) {
    *error = [NSError gus_errorWithStatus:status];
    return nil;
  }
  CVPixelBufferRef pixel_buffer = *pixel_buffer_holder;
  CVPixelBufferRetain(pixel_buffer);
  return pixel_buffer;
}
