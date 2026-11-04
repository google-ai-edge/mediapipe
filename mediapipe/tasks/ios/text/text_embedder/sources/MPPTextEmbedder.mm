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

#import "mediapipe/tasks/c/text/text_embedder/text_embedder.h"
#import "mediapipe/tasks/ios/common/sources/MPPCommon.h"
#import "mediapipe/tasks/ios/common/utils/sources/MPPCommonUtils.h"
#import "mediapipe/tasks/ios/components/utils/sources/MPPCosineSimilarity.h"
#import "mediapipe/tasks/ios/text/text_embedder/utils/sources/MPPTextEmbedderOptions+Helpers.h"
#import "mediapipe/tasks/ios/text/text_embedder/utils/sources/MPPTextEmbedderResult+Helpers.h"

@implementation MPPTextFormatContext

- (instancetype)init {
  self = [super init];
  if (self) {
    _embeddingType = MPPEmbeddingTypeRetrievalQuery;
    _textRole = MPPTextRoleQuery;
  }
  return self;
}

@end

@interface MPPTextEmbedder () {
  MpTextEmbedderPtr _embedder;
}
@end

@implementation MPPTextEmbedder

- (void)dealloc {
  if (_embedder) {
    MpTextEmbedderClose(_embedder, nullptr);
    _embedder = nullptr;
  }
}

- (instancetype)initWithOptions:(MPPTextEmbedderOptions *)options error:(NSError **)error {
  self = [super init];
  if (self) {
    MpTextEmbedderOptions cOptions = {};
    [options copyToCOptions:&cOptions];
    char *cError = nullptr;
    MpStatus status = MpTextEmbedderCreate(&cOptions, &_embedder, &cError);
    [MPPTextEmbedderOptions freeCOptions:&cOptions];
    if (![MPPCommonUtils checkCError:status errorMessage:cError toError:error]) {
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

- (nullable MPPTextEmbedderResult *)embedText:(NSString *)text
                            textFormatContext:(nullable MPPTextFormatContext *)textFormatContext
                                        error:(NSError **)error {
  MpTextEmbedderResult cResult = {};
  char *cError = nullptr;
  MpStatus status;
  if (textFormatContext) {
    MpTextEmbedderFormatContext cFormatContext = {};
    cFormatContext.task_type =
        textFormatContext.embeddingType != 0
            ? static_cast<enum MpTextEmbedderEmbeddingType>(textFormatContext.embeddingType)
            : MP_TEXT_EMBEDDER_EMBEDDING_TYPE_RETRIEVAL_QUERY;
    cFormatContext.role = (textFormatContext.textRole == MPPTextRoleDocument)
                              ? MP_TEXT_EMBEDDER_ROLE_DOCUMENT
                              : MP_TEXT_EMBEDDER_ROLE_QUERY;
    cFormatContext.title = textFormatContext.title ? textFormatContext.title.UTF8String : nullptr;
    status = MpTextEmbedderEmbed(_embedder, text ? text.UTF8String : "", &cFormatContext, &cResult,
                                 &cError);
  } else {
    status =
        MpTextEmbedderEmbed(_embedder, text ? text.UTF8String : "", nullptr, &cResult, &cError);
  }
  if (![MPPCommonUtils checkCError:status errorMessage:cError toError:error]) {
    return nil;
  }
  MPPTextEmbedderResult *result = [MPPTextEmbedderResult textEmbedderResultWithCResult:cResult];
  MpTextEmbedderCloseResult(&cResult);
  return result;
}

+ (nullable NSNumber *)cosineSimilarityBetweenEmbedding1:(MPPEmbedding *)embedding1
                                           andEmbedding2:(MPPEmbedding *)embedding2
                                                   error:(NSError **)error {
  return [MPPCosineSimilarity computeBetweenEmbedding1:embedding1
                                         andEmbedding2:embedding2
                                                 error:error];
}

- (nullable MPPEmbeddingResult *)embedContent:(NSArray<id> *)content error:(NSError **)error {
  for (id part in content) {
    if ([part isKindOfClass:[NSString class]]) {
      MPPTextEmbedderResult *result = [self embedText:(NSString *)part error:error];
      if (result == nil) {
        return nil;
      }
      return result.embeddingResult;
    } else if ([part isKindOfClass:[MPPTextPart class]]) {
      MPPTextEmbedderResult *result = [self embedText:((MPPTextPart *)part).text error:error];
      if (result == nil) {
        return nil;
      }
      return result.embeddingResult;
    }
  }
  [MPPCommonUtils
      createCustomError:error
               withCode:MPPTasksErrorCodeInvalidArgumentError
            description:@"Content is empty or does not contain any supported text elements."];
  return nil;
}

@end
