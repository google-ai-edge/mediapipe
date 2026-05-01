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

#import "mediapipe/tasks/ios/text/text_embedder/sources/MPPTextEmbedder.h"

#import "mediapipe/tasks/ios/common/utils/sources/MPPCommonUtils.h"
#import "mediapipe/tasks/ios/common/utils/sources/NSString+Helpers.h"
#import "mediapipe/tasks/ios/components/utils/sources/MPPCosineSimilarity.h"
#import "mediapipe/tasks/ios/core/sources/MPPTaskInfo.h"
#import "mediapipe/tasks/ios/core/sources/MPPTextPacketCreator.h"
#import "mediapipe/tasks/ios/text/core/sources/MPPTextTaskRunner.h"
#import "mediapipe/tasks/ios/text/text_embedder/utils/sources/MPPTextEmbedderOptions+Helpers.h"
#import "mediapipe/tasks/ios/text/text_embedder/utils/sources/MPPTextEmbedderResult+Helpers.h"

namespace {
using ::mediapipe::Packet;
using ::mediapipe::tasks::core::PacketMap;
}  // namespace

static NSString *const kEmbeddingsOutStreamName = @"embeddings_out";
static NSString *const kEmbeddingsTag = @"EMBEDDINGS";
static NSString *const kTextInStreamName = @"text_in";
static NSString *const kTextTag = @"TEXT";
static NSString *const kTaskGraphName = @"mediapipe.tasks.text.text_embedder.TextEmbedderGraph";
static NSString *const kTaskName = @"textEmbedder";

static NSString *const kQueryTemplate = @"task: %@ | query: %@";
static NSString *const kDocumentTemplate = @"title: %@ | text: %@";

@implementation MPPTextFormatContext

- (instancetype)init {
  self = [super init];
  if (self) {
    _textRole = MPPTextRoleQuery;
  }
  return self;
}

@end

@interface MPPTextEmbedder () {
  /** iOS Text Task Runner */
  MPPTextTaskRunner *_textTaskRunner;
}
@end

@implementation MPPTextEmbedder

- (instancetype)initWithOptions:(MPPTextEmbedderOptions *)options error:(NSError **)error {
  self = [super init];
  if (self) {
    MPPTaskInfo *taskInfo = [[MPPTaskInfo alloc]
          initWithTaskName:kTaskName
             taskGraphName:kTaskGraphName
              inputStreams:@[ [NSString stringWithFormat:@"%@:%@", kTextTag, kTextInStreamName] ]
             outputStreams:@[ [NSString stringWithFormat:@"%@:%@", kEmbeddingsTag,
                                                         kEmbeddingsOutStreamName] ]
               taskOptions:options
        enableFlowLimiting:NO
               runningMode:MPPCoreRunningModeUnspecified
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
  MPPTextEmbedderOptions *options = [[MPPTextEmbedderOptions alloc] init];

  options.baseOptions.modelAssetPath = modelPath;

  return [self initWithOptions:options error:error];
}

- (nullable MPPTextEmbedderResult *)embedText:(NSString *)text error:(NSError **)error {
  return [self embedText:text textFormatContext:nil error:error];
}

- (NSString *)taskStringWithEmbeddingType:(MPPEmbeddingType)embeddingType {
  switch (embeddingType) {
    case MPPEmbeddingTypeRetrievalQuery:
      return @"search result";
    case MPPEmbeddingTypeSemanticSimilarity:
      return @"sentence similarity";
    case MPPEmbeddingTypeClassification:
      return @"classification";
    case MPPEmbeddingTypeClustering:
      return @"clustering";
    case MPPEmbeddingTypeQuestionAnswering:
      return @"question answering";
    case MPPEmbeddingTypeFactChecking:
      return @"fact checking";
    case MPPEmbeddingTypeCodeRetrieval:
      return @"code retrieval";
    default:
      return @"search result";
  }
}

- (NSString *)formatText:(NSString *)text
       textFormatContext:(MPPTextFormatContext *)textFormatContext {
  MPPEmbeddingType embeddingType = textFormatContext.embeddingType;
  BOOL isQuery = textFormatContext.textRole != MPPTextRoleDocument;
  NSString *title = textFormatContext.title != nil && textFormatContext.title.length > 0
                        ? textFormatContext.title
                        : @"none";

  switch (embeddingType) {
    case MPPEmbeddingTypeRetrievalDocument:
      return [NSString stringWithFormat:kDocumentTemplate, title, text];
    case MPPEmbeddingTypeRetrievalQuery:
      return [NSString
          stringWithFormat:kQueryTemplate, [self taskStringWithEmbeddingType:embeddingType], text];
    case MPPEmbeddingTypeQuestionAnswering:
    case MPPEmbeddingTypeFactChecking:
    case MPPEmbeddingTypeCodeRetrieval:
      return isQuery ? [NSString stringWithFormat:kQueryTemplate,
                                                  [self taskStringWithEmbeddingType:embeddingType],
                                                  text]
                     : [NSString stringWithFormat:kDocumentTemplate, title, text];
    default:
      return [NSString
          stringWithFormat:kQueryTemplate, [self taskStringWithEmbeddingType:embeddingType], text];
  }
}

- (nullable MPPTextEmbedderResult *)embedText:(NSString *)text
                            textFormatContext:(nullable MPPTextFormatContext *)textFormatContext
                                        error:(NSError **)error {
  NSString *inputText = text;
  if (textFormatContext != nil) {
    inputText = [self formatText:text textFormatContext:textFormatContext];
  }

  Packet packet = [MPPTextPacketCreator createWithText:inputText];

  std::map<std::string, Packet> packetMap = {{kTextInStreamName.cppString, packet}};

  std::optional<PacketMap> outputPacketMap = [_textTaskRunner processPacketMap:packetMap
                                                                         error:error];

  if (!outputPacketMap.has_value()) {
    return nil;
  }
  return [MPPTextEmbedderResult
      textEmbedderResultWithOutputPacket:outputPacketMap
                                             .value()[kEmbeddingsOutStreamName.cppString]];
}

+ (nullable NSNumber *)cosineSimilarityBetweenEmbedding1:(MPPEmbedding *)embedding1
                                           andEmbedding2:(MPPEmbedding *)embedding2
                                                   error:(NSError **)error {
  return [MPPCosineSimilarity computeBetweenEmbedding1:embedding1
                                         andEmbedding2:embedding2
                                                 error:error];
}

@end
