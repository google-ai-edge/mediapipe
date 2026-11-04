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

#import "mediapipe/tasks/ios/text/text_classifier/sources/MPPTextClassifier.h"

#import "mediapipe/tasks/c/text/text_classifier/text_classifier.h"
#import "mediapipe/tasks/ios/common/utils/sources/MPPCommonUtils.h"
#import "mediapipe/tasks/ios/text/text_classifier/utils/sources/MPPTextClassifierOptions+Helpers.h"
#import "mediapipe/tasks/ios/text/text_classifier/utils/sources/MPPTextClassifierResult+Helpers.h"

@interface MPPTextClassifier () {
  MpTextClassifierPtr _classifier;
}
@end

@implementation MPPTextClassifier

- (void)dealloc {
  if (_classifier) {
    MpTextClassifierClose(_classifier, nullptr);
    _classifier = nullptr;
  }
}

- (instancetype)initWithOptions:(MPPTextClassifierOptions *)options error:(NSError **)error {
  self = [super init];
  if (self) {
    MpTextClassifierOptions cOptions = {};
    [options copyToCOptions:&cOptions];
    char *cError = nullptr;
    MpStatus status = MpTextClassifierCreate(&cOptions, &_classifier, &cError);
    [MPPTextClassifierOptions freeCOptions:&cOptions];
    if (![MPPCommonUtils checkCError:status errorMessage:cError toError:error]) {
      return nil;
    }
  }
  return self;
}

- (instancetype)initWithModelPath:(NSString *)modelPath error:(NSError **)error {
  MPPTextClassifierOptions *options = [[MPPTextClassifierOptions alloc] init];
  options.baseOptions.modelAssetPath = modelPath;
  return [self initWithOptions:options error:error];
}

- (nullable MPPTextClassifierResult *)classifyText:(NSString *)text error:(NSError **)error {
  MpTextClassifierResult cResult = {};
  char *cError = nullptr;
  MpStatus status =
      MpTextClassifierClassify(_classifier, text ? text.UTF8String : "", &cResult, &cError);
  if (![MPPCommonUtils checkCError:status errorMessage:cError toError:error]) {
    return nil;
  }
  MPPTextClassifierResult *result =
      [MPPTextClassifierResult textClassifierResultWithCResult:cResult];
  MpTextClassifierCloseResult(&cResult);
  return result;
}

@end
