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

#import "mediapipe/tasks/ios/text/text_embedder/utils/sources/MPPTextEmbedderOptions+Helpers.h"

#import "mediapipe/tasks/ios/common/utils/sources/NSString+Helpers.h"
#import "mediapipe/tasks/ios/core/utils/sources/MPPBaseOptions+Helpers.h"

#include "mediapipe/tasks/cc/components/processors/proto/embedder_options.pb.h"
#include "mediapipe/tasks/cc/text/text_embedder/proto/text_embedder_graph_options.pb.h"

namespace {
using CalculatorOptionsProto = ::mediapipe::CalculatorOptions;
using TextEmbedderGraphOptionsProto =
    ::mediapipe::tasks::text::text_embedder::proto::TextEmbedderGraphOptions;
using EmbedderOptionsProto = ::mediapipe::tasks::components::processors::proto::EmbedderOptions;
}  // namespace

@implementation MPPTextEmbedderOptions (Helpers)

- (void)copyToProto:(CalculatorOptionsProto *)optionsProto {
  TextEmbedderGraphOptionsProto *graphOptions =
      optionsProto->MutableExtension(TextEmbedderGraphOptionsProto::ext);
  [self.baseOptions copyToProto:graphOptions->mutable_base_options()];

  EmbedderOptionsProto *embedderOptionsProto = graphOptions->mutable_embedder_options();
  embedderOptionsProto->Clear();

  embedderOptionsProto->set_l2_normalize(self.l2Normalize ? true : false);
  embedderOptionsProto->set_quantize(self.quantize ? true : false);
}

@end
