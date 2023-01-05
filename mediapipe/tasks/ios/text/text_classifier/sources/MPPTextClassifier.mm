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
#import "mediapipe/tasks/ios/common/utils/sources/MPPCommonUtils.h"
#import "mediapipe/tasks/ios/common/utils/sources/NSString+Helpers.h"
#import "mediapipe/tasks/ios/core/sources/MPPTaskInfo.h"
#import "mediapipe/tasks/ios/core/sources/MPPTextPacketCreator.h"
#import "mediapipe/tasks/ios/text/core/sources/MPPTextTaskRunner.h"
#import "mediapipe/tasks/ios/text/text_classifier/utils/sources/MPPTextClassifierOptions+Helpers.h"
#import "mediapipe/tasks/ios/text/text_classifier/utils/sources/MPPTextClassifierResult+Helpers.h"

#include "mediapipe/tasks/cc/components/containers/proto/classifications.pb.h"

#include "absl/status/statusor.h"

namespace {
using ::mediapipe::Packet;
using ::mediapipe::tasks::components::containers::proto::ClassificationResult;
using ::mediapipe::tasks::core::PacketMap;
}  // namespace

static NSString *const kClassificationsStreamName = @"classifications_out";
static NSString *const kClassificationsTag = @"CLASSIFICATIONS";
static NSString *const kTextInStreamName = @"text_in";
static NSString *const kTextTag = @"TEXT";
static NSString *const kTaskGraphName = @"mediapipe.tasks.text.text_classifier.TextClassifierGraph";

@interface MPPTextClassifier () {
  /** TextSearcher backed by C++ API */
  MPPTextTaskRunner *_taskRunner;
}
@end

@implementation MPPTextClassifier

- (instancetype)initWithOptions:(MPPTextClassifierOptions *)options error:(NSError **)error {
  MPPTaskInfo *taskInfo = [[MPPTaskInfo alloc]
      initWithTaskGraphName:kTaskGraphName
               inputStreams:@[ [NSString stringWithFormat:@"%@:%@", kTextTag, kTextInStreamName] ]
              outputStreams:@[ [NSString stringWithFormat:@"%@:%@", kClassificationsTag,
                                                          kClassificationsStreamName] ]
                taskOptions:options
         enableFlowLimiting:NO
                      error:error];

  if (!taskInfo) {
    return nil;
  }

  _taskRunner =
      [[MPPTextTaskRunner alloc] initWithCalculatorGraphConfig:[taskInfo generateGraphConfig]
                                                         error:error];
  self = [super init];

  return self;
}

- (instancetype)initWithModelPath:(NSString *)modelPath error:(NSError **)error {
  MPPTextClassifierOptions *options = [[MPPTextClassifierOptions alloc] init];

  options.baseOptions.modelAssetPath = modelPath;

  return [self initWithOptions:options error:error];
}

- (nullable MPPTextClassifierResult *)classifyWithText:(NSString *)text error:(NSError **)error {
  Packet packet = [MPPTextPacketCreator createWithText:text];

  std::map<std::string, Packet> packet_map = {{kTextInStreamName.cppString, packet}};
  absl::StatusOr<PacketMap> status_or_output_packet_map = [_taskRunner process:packet_map];

  if (![MPPCommonUtils checkCppError:status_or_output_packet_map.status() toError:error]) {
    return nil;
  }

  Packet classifications_packet =
      status_or_output_packet_map.value()[kClassificationsStreamName.cppString];

  return [MPPTextClassifierResult
      textClassifierResultWithClassificationsPacket:status_or_output_packet_map.value()
                                                        [kClassificationsStreamName.cppString]];

  // return [MPPTextClassifierResult
  //     textClassifierResultWithClassificationsPacket:output_packet_map.value()[kClassificationsStreamName.cppString]
  //                                       .Get<ClassificationResult>()];
}

@end
