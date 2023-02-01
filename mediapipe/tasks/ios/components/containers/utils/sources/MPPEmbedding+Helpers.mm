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

#import "mediapipe/tasks/ios/common/utils/sources/NSString+Helpers.h"
#import "mediapipe/tasks/ios/components/containers/utils/sources/MPPEmbedding+Helpers.h"

#include <memory>

namespace {
using EmbeddingProto = ::mediapipe::tasks::components::containers::proto::Embedding;
}

@implementation MPPEmbedding (Helpers)

+ (MPPEmbedding *)embeddingWithProto:(const EmbeddingProto &)embeddingProto {

  NSMutableArray<NSNumber *> *floatEmbedding;
  NSMutableArray<NSNumber *> *quantizedEmbedding;
  NSString *headName;

  if (embeddingProto.has_float_embedding()) {
    floatEmbedding =
        [NSMutableArray arrayWithCapacity:embeddingProto.float_embedding().values_size()];

    for (const auto value : embeddingProto.float_embedding().values()) {
      [floatEmbedding addObject:[NSNumber numberWithFloat:value]];
    }
  }

  if (embeddingProto.has_quantized_embedding()) {
    const std::string &cppQuantizedEmbedding = embeddingProto.quantized_embedding().values();
    quantizedEmbedding = [NSMutableArray arrayWithCapacity:cppQuantizedEmbedding.length()];

    for (char ch : cppQuantizedEmbedding) {
      [quantizedEmbedding addObject:[NSNumber numberWithChar:ch]];
    }
  }

  if (embeddingProto.has_head_name()) {
    headName = [NSString stringWithCppString:embeddingProto.head_name()];
  }

  return [[MPPEmbedding alloc] initWithFloatEmbedding:floatEmbedding
                                   quantizedEmbedding:quantizedEmbedding
                                            headIndex:embeddingProto.head_index()
                                             headName:headName];
}

@end
