// Copyright 2022 The MediaPipe Authors.
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

#include "mediapipe/tasks/cc/text/text_classifier/proto/text_classifier_graph_options.pb.h"
#import "mediapipe/tasks/ios/components/processors/utils/sources/MPPClassifierOptions+Helpers.h"
#import "mediapipe/tasks/ios/core/utils/sources/MPPBaseOptions+Helpers.h"
#import "mediapipe/tasks/ios/text/text_classifier/utils/sources/MPPTextClassifierOptions+Helpers.h"

namespace {
using CalculatorOptionsProto = ::mediapipe::CalculatorOptions;
using TextClassifierGraphOptionsProto =
    ::mediapipe::tasks::text::text_classifier::proto::TextClassifierGraphOptions;

}  // namespace

@implementation MPPTextClassifierOptions (Helpers)

- (void)copyToProto:(CalculatorOptionsProto *)optionsProto {
  TextClassifierGraphOptionsProto *graph_options =
      optionsProto->MutableExtension(TextClassifierGraphOptionsProto::ext);
  [self.baseOptions copyToProto:graph_options->mutable_base_options()];
  [self.classifierOptions copyToProto:graph_options->mutable_classifier_options()];
}

@end
