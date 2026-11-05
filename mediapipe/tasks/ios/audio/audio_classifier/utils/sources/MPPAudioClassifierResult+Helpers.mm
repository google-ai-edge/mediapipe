// Copyright 2024 The MediaPipe Authors.
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

#import "mediapipe/tasks/ios/audio/audio_classifier/utils/sources/MPPAudioClassifierResult+Helpers.h"

#import "mediapipe/tasks/ios/components/containers/utils/sources/MPPClassificationResult+Helpers.h"

#include "mediapipe/tasks/c/audio/audio_classifier/audio_classifier.h"

@implementation MPPAudioClassifierResult (Helpers)

+ (nullable MPPAudioClassifierResult *)
    audioClassifierResultWithCResult:(const MpAudioClassifierResult &)cResult
             timestampInMilliseconds:(NSInteger)timestampInMilliseconds {
  NSMutableArray<MPPClassificationResult *> *classificationResults =
      [NSMutableArray arrayWithCapacity:cResult.results_count];

  for (int i = 0; i < cResult.results_count; ++i) {
    MPPClassificationResult *classificationResult =
        [MPPClassificationResult classificationResultWithCResult:cResult.results[i]];
    if (classificationResult) {
      [classificationResults addObject:classificationResult];
    }
  }

  return [[MPPAudioClassifierResult alloc] initWithClassificationResults:classificationResults
                                                 timestampInMilliseconds:timestampInMilliseconds];
}

@end
