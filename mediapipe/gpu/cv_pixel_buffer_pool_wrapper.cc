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

#include "mediapipe/gpu/cv_pixel_buffer_pool_wrapper.h"

#include <tuple>

#include "CoreFoundation/CFBase.h"
#include "absl/log/absl_check.h"
#include "absl/status/statusor.h"
#include "mediapipe/framework/port/logging.h"
#include "mediapipe/objc/CFHolder.h"
#include "mediapipe/objc/util.h"

namespace mediapipe {

CvPixelBufferPoolWrapper::CvPixelBufferPoolWrapper(
    int width, int height, GpuBufferFormat format, CFTimeInterval maxAge,
    CvTextureCacheManager* texture_caches) {
  OSType cv_format = CVPixelFormatForGpuBufferFormat(format);
  ABSL_CHECK_NE(cv_format, -1) << "unsupported pixel format";
  pool_ = MakeCFHolderAdopting(
      /* keep count is 0 because the age param keeps buffers around anyway */
      CreateCVPixelBufferPool(width, height, cv_format, 0, maxAge));
  texture_caches_ = texture_caches;
}

absl::StatusOr<CFHolder<CVPixelBufferRef>>
CvPixelBufferPoolWrapper::GetBuffer() {
  CVPixelBufferRef buffer;
  int threshold = 1;
  NSMutableDictionary* auxAttributes =
      [NSMutableDictionary dictionaryWithCapacity:1];
  CVReturn err;
  bool tried_flushing = false;
  while (1) {
    auxAttributes[(id)kCVPixelBufferPoolAllocationThresholdKey] = @(threshold);
    err = CVPixelBufferPoolCreatePixelBufferWithAuxAttributes(
        kCFAllocatorDefault, *pool_, (__bridge CFDictionaryRef)auxAttributes,
        &buffer);
    if (err != kCVReturnWouldExceedAllocationThreshold) break;
    if (texture_caches_ && !tried_flushing) {
      // Call the flush function to potentially release old holds on buffers
      // and try again to create a pixel buffer.
      // This is used to flush CV texture caches, which may retain buffers until
      // flushed.
      texture_caches_->FlushTextureCaches();
      tried_flushing = true;
    } else {
      ++threshold;
    }
  }
  ABSL_CHECK(!err) << "Error creating pixel buffer: " << err;
  count_ = threshold;
  return MakeCFHolderAdopting(buffer);
}

std::string CvPixelBufferPoolWrapper::GetDebugString() const {
  auto description = MakeCFHolderAdopting(CFCopyDescription(*pool_));
  return [(__bridge NSString*)*description UTF8String];
}

void CvPixelBufferPoolWrapper::Flush() { CVPixelBufferPoolFlush(*pool_, 0); }

absl::StatusOr<CFHolder<CVPixelBufferRef>>
CvPixelBufferPoolWrapper::CreateBufferWithoutPool(
    const internal::GpuBufferSpec& spec) {
  OSType cv_format = CVPixelFormatForGpuBufferFormat(spec.format);
  ABSL_CHECK_NE(cv_format, -1) << "unsupported pixel format";
  CVPixelBufferRef buffer;
  CVReturn err = CreateCVPixelBufferWithoutPool(spec.width, spec.height,
                                                cv_format, &buffer);
  ABSL_CHECK(!err) << "Error creating pixel buffer: " << err;
  return MakeCFHolderAdopting(buffer);
}

}  // namespace mediapipe
