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

#ifndef MEDIAPIPE_OBJC_CGIMAGEREFUTILS_H_
#define MEDIAPIPE_OBJC_CGIMAGEREFUTILS_H_

#import <CoreVideo/CoreVideo.h>

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

extern NSString *const kCGImageRefUtilsErrorDomain;

// TODO: Get rid of this library or make it a wrapper around util.h
// versions so that it can be used in pure Objective-C code.

/// Creates a CGImage with a copy of the contents of the CVPixelBuffer. Returns nil on error, if
/// the |error| argument is not nil, *error is set to an NSError describing the failure. Caller
/// is responsible for releasing the CGImage by calling CGImageRelease().
CGImageRef CreateCGImageFromCVPixelBuffer(CVPixelBufferRef imageBuffer, NSError **error);

/// Creates a CVPixelBuffer with a copy of the contents of the CGImage. Returns nil on error, if
/// the |error| argument is not nil, *error is set to an NSError describing the failure. Caller
/// is responsible for releasing the CVPixelBuffer by calling CVPixelBufferRelease.
CVPixelBufferRef CreateCVPixelBufferFromCGImage(CGImageRef image, NSError **error);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // MEDIAPIPE_OBJC_CGIMAGEREFUTILS_H_
