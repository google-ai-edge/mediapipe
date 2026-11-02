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

#include "mediapipe/tasks/c/components/containers/category.h"

@implementation MPPCategory (Helpers)

+ (MPPCategory *)categoryWithProto:(const ::mediapipe::Classification &)classificationProto {
  return [MPPCategory categoryWithProto:classificationProto index:classificationProto.index()];
}

+ (MPPCategory *)categoryWithProto:(const ::mediapipe::Classification &)classificationProto
                             index:(NSInteger)index {
  NSString *categoryName;
  if (classificationProto.has_label()) {
    categoryName = [NSString stringWithCppString:classificationProto.label()];
  }

  NSString *displayName;
  if (classificationProto.has_display_name()) {
    displayName = [NSString stringWithCppString:classificationProto.display_name()];
  }

  return [[MPPCategory alloc] initWithIndex:index
                                      score:classificationProto.score()
                               categoryName:categoryName
                                displayName:displayName];
}

+ (MPPCategory *)categoryWithCCategory:(const struct MpCategory &)cCategory {
  NSString *categoryName =
      cCategory.category_name ? [NSString stringWithUTF8String:cCategory.category_name] : nil;
  NSString *displayName =
      cCategory.display_name ? [NSString stringWithUTF8String:cCategory.display_name] : nil;

  return [[MPPCategory alloc] initWithIndex:cCategory.index
                                      score:cCategory.score
                               categoryName:categoryName
                                displayName:displayName];
}

@end
