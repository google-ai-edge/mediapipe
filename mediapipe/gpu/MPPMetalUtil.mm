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
  [commandBuffer commit];
  if (blocking) [commandBuffer waitUntilCompleted];
}

@end
