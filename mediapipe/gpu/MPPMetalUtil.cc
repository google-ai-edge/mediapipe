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

#import "mediapipe/gpu/MPPMetalUtil.h"

#include "absl/time/clock.h"
#include "absl/time/time.h"

@implementation MPPMetalUtil

+ (void)blitMetalBufferTo:(id<MTLBuffer>)destination
                     from:(id<MTLBuffer>)source
                 blocking:(bool)blocking
            commandBuffer:(id<MTLCommandBuffer>)commandBuffer {
  size_t bytes = MIN(destination.length, source.length);
  [self blitMetalBufferTo:destination
        destinationOffset:0
                     from:source
             sourceOffset:0
                    bytes:bytes
                 blocking:blocking
            commandBuffer:commandBuffer];
}

+ (void)blitMetalBufferTo:(id<MTLBuffer>)destination
        destinationOffset:(int)destinationOffset
                     from:(id<MTLBuffer>)source
             sourceOffset:(int)sourceOffset
                    bytes:(size_t)bytes
                 blocking:(bool)blocking
            commandBuffer:(id<MTLCommandBuffer>)commandBuffer {
  id<MTLBlitCommandEncoder> blit_command = [commandBuffer blitCommandEncoder];
  [blit_command copyFromBuffer:source
                  sourceOffset:sourceOffset
                      toBuffer:destination
             destinationOffset:destinationOffset
                          size:bytes];
  [blit_command endEncoding];
  if (blocking) {
    [MPPMetalUtil commitCommandBufferAndWait:commandBuffer];
  } else {
    [commandBuffer commit];
  }
}

+ (void)commitCommandBufferAndWait:(id<MTLCommandBuffer>)commandBuffer {
#if !defined(MEDIAPIPE_DISABLE_ACTIVE_WAIT)
  // The bufferCompleted variable doesn't require atomic access.
  // std::atomic<> can't be used here because the variable must be captured
  // with the block. Also std::atomic<> orders changes of the variable but
  // in this case any kind of out-of-order execution will be serialized.
  __block volatile bool bufferCompleted = false;
  [commandBuffer addCompletedHandler:^(id<MTLCommandBuffer>) {
    bufferCompleted = true;
  }];
  [commandBuffer commit];
  absl::Time start_time = absl::Now();
  while (!bufferCompleted) {
    auto duration = absl::Now() - start_time;
    // If the spin-lock takes more than 5 ms then go to blocking wait:
    // - it frees the CPU core for another threads: increase the
    // performance/decrease power consumption.
    // - if a driver thread that notifies that the GPU buffer is completed has
    // lower priority then the CPU core is allocated for the thread.
    if (duration >= absl::Milliseconds(5)) {
      [commandBuffer waitUntilCompleted];
      break;
    }
  }
#else
  [commandBuffer commit];
  [commandBuffer waitUntilCompleted];
#endif
}

@end
