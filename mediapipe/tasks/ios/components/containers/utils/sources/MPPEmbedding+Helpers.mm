// Copyright 2023 The MediaPipe Authors.
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

#import "mediapipe/tasks/ios/components/containers/utils/sources/MPPEmbedding+Helpers.h"

#include "mediapipe/tasks/c/components/containers/embedding_result.h"

@implementation MPPEmbedding (Helpers)

+ (MPPEmbedding *)embeddingWithCEmbedding:(const MpEmbedding &)cEmbedding {
  NSMutableArray<NSNumber *> *floatEmbedding = nil;
  NSMutableArray<NSNumber *> *quantizedEmbedding = nil;
  NSString *headName =
      cEmbedding.head_name ? [NSString stringWithUTF8String:cEmbedding.head_name] : nil;

  if (cEmbedding.float_embedding && cEmbedding.values_count > 0) {
    floatEmbedding = [NSMutableArray arrayWithCapacity:cEmbedding.values_count];
    for (uint32_t i = 0; i < cEmbedding.values_count; ++i) {
      [floatEmbedding addObject:@(cEmbedding.float_embedding[i])];
    }
  }

  if (cEmbedding.quantized_embedding && cEmbedding.values_count > 0) {
    quantizedEmbedding = [NSMutableArray arrayWithCapacity:cEmbedding.values_count];
    for (uint32_t i = 0; i < cEmbedding.values_count; ++i) {
      [quantizedEmbedding addObject:@(cEmbedding.quantized_embedding[i])];
    }
  }

  return [[MPPEmbedding alloc] initWithFloatEmbedding:floatEmbedding
                                   quantizedEmbedding:quantizedEmbedding
                                            headIndex:cEmbedding.head_index
                                             headName:headName];
}

@end
