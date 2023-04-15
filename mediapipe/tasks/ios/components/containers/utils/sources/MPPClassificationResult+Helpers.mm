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
#import "mediapipe/tasks/ios/components/containers/utils/sources/MPPCategory+Helpers.h"
#import "mediapipe/tasks/ios/components/containers/utils/sources/MPPClassificationResult+Helpers.h"

namespace {
using ClassificationsProto = ::mediapipe::tasks::components::containers::proto::Classifications;
using ClassificationResultProto =
    ::mediapipe::tasks::components::containers::proto::ClassificationResult;
}  // namespace

@implementation MPPClassifications (Helpers)

+ (MPPClassifications *)classificationsWithProto:
    (const ClassificationsProto &)classificationsProto {
  NSMutableArray<MPPCategory *> *categories =
      [NSMutableArray arrayWithCapacity:(NSUInteger)classificationsProto.classification_list()
                                            .classification_size()];
  for (const auto &classification : classificationsProto.classification_list().classification()) {
    [categories addObject:[MPPCategory categoryWithProto:classification]];
  }

  NSString *headName;
  if (classificationsProto.has_head_name()) {
    headName = [NSString stringWithCppString:classificationsProto.head_name()];
  }

  return [[MPPClassifications alloc] initWithHeadIndex:(NSInteger)classificationsProto.head_index()
                                              headName:headName
                                            categories:categories];
}

@end

@implementation MPPClassificationResult (Helpers)

+ (MPPClassificationResult *)classificationResultWithProto:
    (const ClassificationResultProto &)classificationResultProto {
  NSMutableArray *classifications = [NSMutableArray
      arrayWithCapacity:(NSUInteger)classificationResultProto.classifications_size()];
  for (const auto &classificationsProto : classificationResultProto.classifications()) {
    [classifications addObject:[MPPClassifications classificationsWithProto:classificationsProto]];
  }

  NSInteger timestampInMilliseconds = 0;
  if (classificationResultProto.has_timestamp_ms()) {
    timestampInMilliseconds = (NSInteger)classificationResultProto.timestamp_ms();
  }

  return [[MPPClassificationResult alloc] initWithClassifications:classifications
                                          timestampInMilliseconds:timestampInMilliseconds];
  ;
}

@end
