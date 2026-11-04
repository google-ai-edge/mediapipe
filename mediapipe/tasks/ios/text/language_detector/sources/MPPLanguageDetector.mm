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

#import "mediapipe/tasks/ios/text/language_detector/sources/MPPLanguageDetector.h"

#import "mediapipe/tasks/c/text/language_detector/language_detector.h"
#import "mediapipe/tasks/ios/common/utils/sources/MPPCommonUtils.h"
#import "mediapipe/tasks/ios/text/language_detector/utils/sources/MPPLanguageDetectorOptions+Helpers.h"
#import "mediapipe/tasks/ios/text/language_detector/utils/sources/MPPLanguageDetectorResult+Helpers.h"

@interface MPPLanguageDetector () {
  MpLanguageDetectorPtr _detector;
}
@end

@implementation MPPLanguageDetector

- (void)dealloc {
  if (_detector) {
    MpLanguageDetectorClose(_detector, nullptr);
    _detector = nullptr;
  }
}

- (instancetype)initWithOptions:(MPPLanguageDetectorOptions *)options error:(NSError **)error {
  self = [super init];
  if (self) {
    MpLanguageDetectorOptions cOptions = {};
    [options copyToCOptions:&cOptions];
    char *cError = nullptr;
    MpStatus status = MpLanguageDetectorCreate(&cOptions, &_detector, &cError);
    [MPPLanguageDetectorOptions freeCOptions:&cOptions];
    if (![MPPCommonUtils checkCError:status errorMessage:cError toError:error]) {
      return nil;
    }
  }
  return self;
}

- (instancetype)initWithModelPath:(NSString *)modelPath error:(NSError **)error {
  MPPLanguageDetectorOptions *options = [[MPPLanguageDetectorOptions alloc] init];
  options.baseOptions.modelAssetPath = modelPath;
  return [self initWithOptions:options error:error];
}

- (nullable MPPLanguageDetectorResult *)detectText:(NSString *)text error:(NSError **)error {
  MpLanguageDetectorResult cResult = {};
  char *cError = nullptr;
  MpStatus status =
      MpLanguageDetectorDetect(_detector, text ? text.UTF8String : "", &cResult, &cError);
  if (![MPPCommonUtils checkCError:status errorMessage:cError toError:error]) {
    return nil;
  }
  MPPLanguageDetectorResult *result =
      [MPPLanguageDetectorResult languageDetectorResultWithCResult:cResult];
  MpLanguageDetectorCloseResult(&cResult);
  return result;
}

@end
