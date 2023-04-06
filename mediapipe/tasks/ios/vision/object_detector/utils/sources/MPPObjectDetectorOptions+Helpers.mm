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

#import "mediapipe/tasks/ios/vision/object_detector/utils/sources/MPPObjectDetectorOptions+Helpers.h"

#import "mediapipe/tasks/ios/common/utils/sources/NSString+Helpers.h"
#import "mediapipe/tasks/ios/core/utils/sources/MPPBaseOptions+Helpers.h"

#include "mediapipe/tasks/cc/vision/object_detector/proto/object_detector_options.pb.h"

namespace {
using CalculatorOptionsProto = ::mediapipe::CalculatorOptions;
using ObjectDetectorOptionsProto =
    ::mediapipe::tasks::vision::object_detector::proto::ObjectDetectorOptions;
}  // namespace

@implementation MPPObjectDetectorOptions (Helpers)

- (void)copyToProto:(CalculatorOptionsProto *)optionsProto {
  ObjectDetectorOptionsProto *graphOptions =
      optionsProto->MutableExtension(ObjectDetectorOptionsProto::ext);

  graphOptions->Clear();

  [self.baseOptions copyToProto:graphOptions->mutable_base_options()];

  if (self.displayNamesLocale) {
    graphOptions->set_display_names_locale(self.displayNamesLocale.cppString);
  }

  graphOptions->set_max_results((int)self.maxResults);
  graphOptions->set_score_threshold(self.scoreThreshold);

  for (NSString *category in self.categoryAllowlist) {
    graphOptions->add_category_allowlist(category.cppString);
  }

  for (NSString *category in self.categoryDenylist) {
    graphOptions->add_category_denylist(category.cppString);
  }
}

@end
