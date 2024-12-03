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

#import "mediapipe/tasks/ios/common/utils/sources/MPPCommonUtils.h"
#import "mediapipe/tasks/ios/common/utils/sources/NSString+Helpers.h"
#import "mediapipe/tasks/ios/core/sources/MPPTaskInfo.h"
#import "mediapipe/tasks/ios/core/sources/MPPTextPacketCreator.h"
#import "mediapipe/tasks/ios/text/core/sources/MPPTextTaskRunner.h"
#import "mediapipe/tasks/ios/text/language_detector/utils/sources/MPPLanguageDetectorOptions+Helpers.h"
#import "mediapipe/tasks/ios/text/language_detector/utils/sources/MPPLanguageDetectorResult+Helpers.h"

namespace {
using ::mediapipe::Packet;
using ::mediapipe::tasks::core::PacketMap;
}  // namespace

static NSString *const kClassificationsStreamName = @"classifications_out";
static NSString *const kClassificationsTag = @"CLASSIFICATIONS";
static NSString *const kTextInStreamName = @"text_in";
static NSString *const kTextTag = @"TEXT";
static NSString *const kTaskGraphName = @"mediapipe.tasks.text.text_classifier.TextClassifierGraph";

@interface MPPLanguageDetector () {
  /** iOS Text Task Runner */
  MPPTextTaskRunner *_textTaskRunner;
}
@end

@implementation MPPLanguageDetector

- (instancetype)initWithOptions:(MPPLanguageDetectorOptions *)options error:(NSError **)error {
  self = [super init];
  if (self) {
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

    _textTaskRunner = [[MPPTextTaskRunner alloc] initWithTaskInfo:taskInfo error:error];

    if (!_textTaskRunner) {
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
  Packet packet = [MPPTextPacketCreator createWithText:text];

  std::map<std::string, Packet> packetMap = {{kTextInStreamName.cppString, packet}};
  std::optional<PacketMap> outputPacketMap = [_textTaskRunner processPacketMap:packetMap
                                                                         error:error];

  if (!outputPacketMap.has_value()) {
    return nil;
  }

  return
      [MPPLanguageDetectorResult languageDetectorResultWithClassificationsPacket:
                                     outputPacketMap.value()[kClassificationsStreamName.cppString]];
}

@end
