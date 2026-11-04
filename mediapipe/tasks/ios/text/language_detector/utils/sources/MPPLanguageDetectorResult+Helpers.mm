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

#import "mediapipe/tasks/ios/text/language_detector/utils/sources/MPPLanguageDetectorResult+Helpers.h"

@implementation MPPLanguageDetectorResult (Helpers)

+ (MPPLanguageDetectorResult *)languageDetectorResultWithCResult:
    (const MpLanguageDetectorResult &)cResult {
  NSMutableArray<MPPLanguagePrediction *> *languagePredictions =
      [NSMutableArray arrayWithCapacity:cResult.predictions_count];

  for (uint32_t i = 0; i < cResult.predictions_count; ++i) {
    const auto &cPrediction = cResult.predictions[i];
    NSString *languageCode =
        cPrediction.language_code ? [NSString stringWithUTF8String:cPrediction.language_code] : @"";
    MPPLanguagePrediction *languagePrediction =
        [[MPPLanguagePrediction alloc] initWithLanguageCode:languageCode
                                                probability:cPrediction.probability];
    [languagePredictions addObject:languagePrediction];
  }

  return [[MPPLanguageDetectorResult alloc] initWithLanguagePredictions:languagePredictions
                                                timestampInMilliseconds:0];
}

@end
