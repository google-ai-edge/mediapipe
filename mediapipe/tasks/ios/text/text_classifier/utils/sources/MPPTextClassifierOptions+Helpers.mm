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

#import "mediapipe/tasks/ios/text/text_classifier/utils/sources/MPPTextClassifierOptions+Helpers.h"

#import "mediapipe/tasks/ios/common/utils/sources/NSString+Helpers.h"
#import "mediapipe/tasks/ios/core/utils/sources/MPPBaseOptions+Helpers.h"

#include "mediapipe/tasks/cc/text/text_classifier/proto/text_classifier_graph_options.pb.h"
#include "mediapipe/tasks/cc/components/processors/proto/classifier_options.pb.h"

namespace {
using CalculatorOptionsProto = ::mediapipe::CalculatorOptions;
using TextClassifierGraphOptionsProto =
    ::mediapipe::tasks::text::text_classifier::proto::TextClassifierGraphOptions;
using ClassifierOptionsProto = ::mediapipe::tasks::components::processors::proto::ClassifierOptions;
}  // namespace

@implementation MPPTextClassifierOptions (Helpers)

- (void)copyToProto:(CalculatorOptionsProto *)optionsProto {
  TextClassifierGraphOptionsProto *graphOptions =
      optionsProto->MutableExtension(TextClassifierGraphOptionsProto::ext);
  [self.baseOptions copyToProto:graphOptions->mutable_base_options()];

  ClassifierOptionsProto *classifierOptionsProto = graphOptions->mutable_classifier_options();
  classifierOptionsProto->Clear();

  if (self.displayNamesLocale) {
    classifierOptionsProto->set_display_names_locale(self.displayNamesLocale.cppString);
  }

  classifierOptionsProto->set_max_results((int)self.maxResults);
  classifierOptionsProto->set_score_threshold(self.scoreThreshold);

  for (NSString *category in self.categoryAllowlist) {
    classifierOptionsProto->add_category_allowlist(category.cppString);
  }

  for (NSString *category in self.categoryDenylist) {
    classifierOptionsProto->add_category_denylist(category.cppString);
  }
}

@end
