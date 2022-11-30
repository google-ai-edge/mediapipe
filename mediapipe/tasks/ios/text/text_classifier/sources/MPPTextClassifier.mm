/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

 http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
 ==============================================================================*/
#import "mediapipe/tasks/ios/text/text_classifier/sources/MPPTextClassifier.h"

#import "mediapipe/tasks/ios/common/utils/sources/MPPCommonUtils.h"
#import "mediapipe/tasks/ios/core/sources/MPPTaskInfo.h"
#import "mediapipe/tasks/ios/text/text_classifier/utils/sources/MPPTextClassifierOptions+Helpers.h"

static NSString *const kClassificationsStreamName = @"classifications_out";
static NSString *const kClassificationsTag = @"classifications";
static NSString *const kTextInStreamName = @"text_in";
static NSString *const kTextTag = @"TEXT";
static NSString *const kTaskGraphName = @"mediapipe.tasks.text.text_classifier.TextClassifierGraph";

@implementation MPPTextClassifier

- (instancetype)initWithOptions:(MPPTextClassifierOptions *)options error:(NSError **)error {
  MPPTaskInfo *taskInfo = [[MPPTaskInfo alloc]
      initWithTaskGraphName:kTaskGraphName
               inputStreams:@[ [NSString stringWithFormat:@"@:@", kTextTag, kTextInStreamName] ]
              outputStreams:@[ [NSString stringWithFormat:@"@:@", kClassificationsTag,
                                                          kClassificationsStreamName] ]
                taskOptions:options
         enableFlowLimiting:NO
                      error:error];

  if (!taskInfo) {
    return nil;
  }

  return [super initWithCalculatorGraphConfig:[taskInfo generateGraphConfig] error:error];
}

- (instancetype)initWithModelPath:(NSString *)modelPath error:(NSError **)error {
  MPPTextClassifierOptions *options =
      [[MPPTextClassifierOptions alloc] initWithModelPath:modelPath];

  return [self initWithOptions:options error:error];
}

@end
