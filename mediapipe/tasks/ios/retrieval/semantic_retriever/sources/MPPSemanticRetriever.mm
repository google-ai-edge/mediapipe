// Copyright 2026 The MediaPipe Authors. All Rights Reserved.
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

#import "mediapipe/tasks/ios/retrieval/semantic_retriever/sources/MPPSemanticRetriever.h"

const NSInteger MPPSemanticRetrieverDefaultEmbeddingDimension = 768;

#import <UIKit/UIKit.h>

#import "mediapipe/tasks/ios/audio/core/sources/MPPAudioData.h"
#import "mediapipe/tasks/ios/common/sources/MPPCommon.h"
#import "mediapipe/tasks/ios/common/utils/sources/MPPCommonUtils.h"
#import "mediapipe/tasks/ios/components/containers/sources/MPPEmbedding.h"
#import "mediapipe/tasks/ios/components/containers/sources/MPPEmbeddingResult.h"
#import "mediapipe/tasks/ios/components/utils/sources/MPPCosineSimilarity.h"
#import "mediapipe/tasks/ios/retrieval/semantic_retriever/sources/MPPAudioLoader.h"
#import "mediapipe/tasks/ios/retrieval/semantic_retriever/sources/MPPRetrievalResult.h"
#import "mediapipe/tasks/ios/vision/core/sources/MPPImage.h"

#include <atomic>
#include <memory>
#include <string>
#include <vector>

#include "mediapipe/framework/timestamp.h"
#include "mediapipe/tasks/cc/core/logging/factory/logging_factory.h"
#include "mediapipe/tasks/cc/core/logging/tasks_logger.h"
#include "mediapipe/tasks/cc/text/utils/genai_utils.h"

@interface MPPDatabaseRecord : NSObject

@property(nonatomic, readonly, copy) NSString *recordId;
@property(nonatomic, readonly, copy) NSArray<MPPTaskPart *> *parts;
@property(nonatomic, readonly, copy, nullable) NSString *parentId;
@property(nonatomic, readonly, copy, nullable) NSArray<NSString *> *childIds;

- (instancetype)initWithRecordId:(NSString *)recordId
                           parts:(NSArray<MPPTaskPart *> *)parts
                        parentId:(nullable NSString *)parentId
                        childIds:(nullable NSArray<NSString *> *)childIds;

@end

@implementation MPPDatabaseRecord

- (instancetype)initWithRecordId:(NSString *)recordId
                           parts:(NSArray<MPPTaskPart *> *)parts
                        parentId:(nullable NSString *)parentId
                        childIds:(nullable NSArray<NSString *> *)childIds {
  self = [super init];
  if (self) {
    _recordId = [recordId copy];
    _parts = [parts copy];
    _parentId = [parentId copy];
    _childIds = [childIds copy];
  }
  return self;
}

@end

@interface MPPSemanticRetriever () {
  MPPSemanticRetrieverComponents *_components;
  id<MPPVectorStore> _vectorStore;
  id<MPPTextChunker> _chunker;
  std::unique_ptr<mediapipe::tasks::core::logging::TasksLogger> _statsLogger;
  std::atomic<int64_t> _syntheticTimestamp;
}

- (nullable NSArray<NSNumber *> *)embedContent:(NSArray<MPPTaskPart *> *)content
                                         error:(NSError **)error;

- (BOOL)persistRecordWithId:(NSString *)recordId
                      parts:(NSArray<MPPTaskPart *> *)parts
                 embeddings:(NSArray<NSNumber *> *)embeddings
                   metadata:(NSDictionary<NSString *, NSString *> *)metadata
                   parentId:(nullable NSString *)parentId
                   childIds:(nullable NSArray<NSString *> *)childIds
                      error:(NSError **)error;

@end

@implementation MPPSemanticRetriever

+ (nullable instancetype)createFromComponents:(MPPSemanticRetrieverComponents *)components
                                        error:(NSError **)error {
  return [[MPPSemanticRetriever alloc] initWithComponents:components error:error];
}

- (nullable instancetype)initWithComponents:(MPPSemanticRetrieverComponents *)components
                                      error:(NSError **)error {
  self = [super init];
  if (self) {
    _components = components;
    _vectorStore = components.vectorStore;
    _chunker = components.chunker;

    NSString *appId = [MPPCommonUtils appID];
    NSString *appVersion = [MPPCommonUtils appVersion];
    NSString *iosVersion = [MPPCommonUtils osVersion];

    mediapipe::tasks::core::logging::LoggingOptions loggingOptions = {
        .task_name = "SemanticRetriever",
        .task_running_mode = mediapipe::tasks::core::RunningMode::kUnspecified,
        .host_environment = mediapipe::tasks::core::HostEnvironment::HOST_ENVIRONMENT_IOS,
        .host_system = mediapipe::tasks::core::HostSystem::HOST_SYSTEM_IOS,
        .host_version = iosVersion ? iosVersion.UTF8String : "",
        .app_id = appId ? appId.UTF8String : "",
        .app_version = appVersion ? appVersion.UTF8String : ""};

    _statsLogger = mediapipe::tasks::core::logging::CreateTasksLogger(loggingOptions);
    _statsLogger->LogSessionStart();
    _syntheticTimestamp = 0;
  }
  return self;
}

- (void)dealloc {
  if (_statsLogger) {
    _statsLogger->LogSessionEnd();
  }
}

- (nullable NSArray<NSNumber *> *)embedContent:(NSArray<MPPTaskPart *> *)content
                                         error:(NSError **)error {
  if (!content || content.count == 0) {
    [MPPCommonUtils createCustomError:error
                             withCode:MPPTasksErrorCodeInvalidArgumentError
                          description:@"Content parts array cannot be nil or empty."];
    return nil;
  }

  // Fast path: attempt to pass MPPTaskPart objects directly to providers that support them.
  NSError *lastError = nil;
  for (id<MPPEmbeddingProvider> provider in _components.providers) {
    NSError *providerError = nil;
    MPPEmbeddingResult *result = [provider embedContent:content error:&providerError];
    if (result && result.embeddings.count > 0) {
      return result.embeddings[0].floatEmbedding;
    }
    if (providerError) {
      lastError = providerError;
    }
  }

  // Fallback path: Unpack MPPTaskPart into legacy raw objects (NSString, MPPImage, MPPAudioData)
  // for providers that expect legacy media representations.
  NSMutableArray<id> *rawContent = [NSMutableArray arrayWithCapacity:content.count];
  for (MPPTaskPart *part in content) {
    if ([part isKindOfClass:[MPPTextPart class]]) {
      [rawContent addObject:((MPPTextPart *)part).text];
    } else if ([part isKindOfClass:[MPPImagePart class]]) {
      MPPImagePart *imagePart = (MPPImagePart *)part;
      UIImage *uiImage = nil;
      if (imagePart.data) {
        uiImage = [UIImage imageWithData:imagePart.data];
        if (!uiImage) {
          [MPPCommonUtils createCustomError:error
                                   withCode:MPPTasksErrorCodeInvalidArgumentError
                                description:@"Failed to decode image from data."];
          return nil;
        }
      } else {
        NSString *filePath = imagePart.filePath;
        NSData *imageBytes = [NSData dataWithContentsOfFile:filePath];
        if (!imageBytes) {
          imageBytes = [NSData dataWithContentsOfURL:[NSURL URLWithString:filePath]];
        }
        if (!imageBytes) {
          [MPPCommonUtils
              createCustomError:error
                       withCode:MPPTasksErrorCodeInvalidArgumentError
                    description:[NSString stringWithFormat:@"Failed to load image from path: %@",
                                                           filePath]];
          return nil;
        }
        uiImage = [UIImage imageWithData:imageBytes];
        if (!uiImage) {
          [MPPCommonUtils
              createCustomError:error
                       withCode:MPPTasksErrorCodeInvalidArgumentError
                    description:[NSString stringWithFormat:@"Failed to decode image from path: %@",
                                                           filePath]];
          return nil;
        }
      }
      MPPImage *mppImage = [[MPPImage alloc] initWithUIImage:uiImage error:error];
      if (!mppImage) {
        return nil;
      }
      [rawContent addObject:mppImage];
    } else if ([part isKindOfClass:[MPPAudioPart class]]) {
      MPPAudioPart *audioPart = (MPPAudioPart *)part;
      MPPAudioData *audioData = nil;
      if (audioPart.data != nil) {
        audioData = [MPPAudioLoader loadAudioFromData:audioPart.data error:error];
      } else {
        audioData = [MPPAudioLoader loadAudioFromFilePath:audioPart.filePath error:error];
      }
      if (!audioData) {
        return nil;
      }
      [rawContent addObject:audioData];
    } else {
      [MPPCommonUtils createCustomError:error
                               withCode:MPPTasksErrorCodeInvalidArgumentError
                            description:@"Unsupported part type in content list."];
      return nil;
    }
  }

  for (id<MPPEmbeddingProvider> provider in _components.providers) {
    NSError *providerError = nil;
    MPPEmbeddingResult *result = [provider embedContent:rawContent error:&providerError];
    if (result && result.embeddings.count > 0) {
      return result.embeddings[0].floatEmbedding;
    }
    if (providerError) {
      lastError = providerError;
    }
  }
  if (lastError) {
    if (error) {
      *error = lastError;
    }
  } else {
    [MPPCommonUtils createCustomError:error
                             withCode:MPPTasksErrorCodeFailedPreconditionError
                          description:@"No embedding provider found that can embed this content."];
  }
  return nil;
}

- (BOOL)insertDocumentWithId:(NSString *)recordId text:(NSString *)text error:(NSError **)error {
  return [self insertDocumentWithId:recordId text:text metadata:@{} error:error];
}

- (BOOL)insertDocumentWithId:(NSString *)recordId
                        text:(NSString *)text
                    metadata:(NSDictionary<NSString *, NSString *> *)metadata
                       error:(NSError **)error {
  MPPTextPart *textPart = [[MPPTextPart alloc] initWithText:text];
  return [self insertContentWithId:recordId parts:@[ textPart ] metadata:metadata error:error];
}

- (BOOL)insertImageWithId:(NSString *)recordId
                 filePath:(NSString *)filePath
                    error:(NSError **)error {
  return [self insertImageWithId:recordId filePath:filePath metadata:@{} error:error];
}

- (BOOL)insertImageWithId:(NSString *)recordId
                 filePath:(NSString *)filePath
                 metadata:(NSDictionary<NSString *, NSString *> *)metadata
                    error:(NSError **)error {
  MPPImagePart *imagePart = [[MPPImagePart alloc] initWithFilePath:filePath];
  return [self insertContentWithId:recordId parts:@[ imagePart ] metadata:metadata error:error];
}

- (BOOL)insertAudioWithId:(NSString *)recordId
                 filePath:(NSString *)filePath
                    error:(NSError **)error {
  return [self insertAudioWithId:recordId filePath:filePath metadata:@{} error:error];
}

- (BOOL)insertAudioWithId:(NSString *)recordId
                 filePath:(NSString *)filePath
                 metadata:(NSDictionary<NSString *, NSString *> *)metadata
                    error:(NSError **)error {
  MPPAudioPart *audioPart = [[MPPAudioPart alloc] initWithFilePath:filePath];
  return [self insertContentWithId:recordId parts:@[ audioPart ] metadata:metadata error:error];
}

- (BOOL)insertContentWithId:(NSString *)recordId
                      parts:(NSArray<MPPTaskPart *> *)parts
                      error:(NSError **)error {
  return [self insertContentWithId:recordId parts:parts metadata:@{} error:error];
}

- (BOOL)insertContentWithId:(NSString *)recordId
                      parts:(NSArray<MPPTaskPart *> *)parts
                   metadata:(NSDictionary<NSString *, NSString *> *)metadata
                      error:(NSError **)error {
  if (!parts || parts.count == 0) {
    [MPPCommonUtils createCustomError:error
                             withCode:MPPTasksErrorCodeInvalidArgumentError
                          description:@"Content parts array cannot be nil or empty."];
    return NO;
  }
  if (![self deleteWithIds:@[ recordId ] error:error]) {
    return NO;
  }

  NSMutableArray<MPPDatabaseRecord *> *databaseRecords = [NSMutableArray array];
  size_t childCount = 0;

  BOOL isSingleTextPart =
      (parts.count == 1 && [parts.firstObject isKindOfClass:[MPPTextPart class]]);
  if (isSingleTextPart) {
    NSString *text = ((MPPTextPart *)parts.firstObject).text;
    NSArray<NSString *> *chunks = [_chunker chunkText:text error:error];
    if (!chunks) {
      return NO;
    }
    if (chunks.count > 1) {
      NSMutableArray<NSString *> *childIds = [NSMutableArray arrayWithCapacity:chunks.count];
      for (NSString *chunkText in chunks) {
        NSString *childId = [NSString stringWithFormat:@"%@_chunk_%zu", recordId, childCount++];
        [childIds addObject:childId];
        MPPTextPart *chunkPart = [[MPPTextPart alloc] initWithText:chunkText];
        MPPDatabaseRecord *childRecord = [[MPPDatabaseRecord alloc] initWithRecordId:childId
                                                                               parts:@[ chunkPart ]
                                                                            parentId:recordId
                                                                            childIds:nil];
        [databaseRecords addObject:childRecord];
      }
      MPPDatabaseRecord *parentRecord = [[MPPDatabaseRecord alloc] initWithRecordId:recordId
                                                                              parts:parts
                                                                           parentId:nil
                                                                           childIds:childIds];
      [databaseRecords addObject:parentRecord];
    } else {
      MPPDatabaseRecord *defaultRecord = [[MPPDatabaseRecord alloc] initWithRecordId:recordId
                                                                               parts:parts
                                                                            parentId:nil
                                                                            childIds:nil];
      [databaseRecords addObject:defaultRecord];
    }
  } else {
    MPPDatabaseRecord *defaultRecord = [[MPPDatabaseRecord alloc] initWithRecordId:recordId
                                                                             parts:parts
                                                                          parentId:nil
                                                                          childIds:nil];
    [databaseRecords addObject:defaultRecord];
  }

  for (MPPDatabaseRecord *record in databaseRecords) {
    NSArray<NSNumber *> *embedding = [self embedContent:record.parts error:error];
    if (!embedding) {
      return NO;
    }
    if (![self persistRecordWithId:record.recordId
                             parts:record.parts
                        embeddings:embedding
                          metadata:metadata
                          parentId:record.parentId
                          childIds:record.childIds
                             error:error]) {
      return NO;
    }
  }

  return YES;
}

- (BOOL)persistRecordWithId:(NSString *)recordId
                      parts:(NSArray<MPPTaskPart *> *)parts
                 embeddings:(NSArray<NSNumber *> *)embeddings
                   metadata:(NSDictionary<NSString *, NSString *> *)metadata
                   parentId:(nullable NSString *)parentId
                   childIds:(nullable NSArray<NSString *> *)childIds
                      error:(NSError **)error {
  MPPRetrievalRecord *record = [[MPPRetrievalRecord alloc] initWithRecordId:recordId
                                                                    content:parts
                                                                 embeddings:embeddings
                                                                   metadata:metadata
                                                                   parentId:parentId
                                                                   childIds:childIds];
  return [_vectorStore upsertRecords:@[ record ] error:error];
}

- (nullable NSArray<MPPRetrievalResult *> *)retrieveWithText:(NSString *)queryText
                                                        topK:(NSInteger)topK
                                                       error:(NSError **)error {
  return [self retrieveWithText:queryText topK:topK metadataFilter:nil error:error];
}

- (nullable NSArray<MPPRetrievalResult *> *)
    retrieveWithText:(NSString *)queryText
                topK:(NSInteger)topK
      metadataFilter:(nullable NSDictionary<NSString *, NSString *> *)metadataFilter
               error:(NSError **)error {
  MPPTextPart *textPart = [[MPPTextPart alloc] initWithText:queryText];
  return [self retrieveWithParts:@[ textPart ] topK:topK metadataFilter:metadataFilter error:error];
}

- (nullable NSArray<MPPRetrievalResult *> *)retrieveWithParts:(NSArray<MPPTaskPart *> *)queryParts
                                                         topK:(NSInteger)topK
                                                        error:(NSError **)error {
  return [self retrieveWithParts:queryParts topK:topK metadataFilter:nil error:error];
}

- (nullable NSArray<MPPRetrievalResult *> *)
    retrieveWithParts:(NSArray<MPPTaskPart *> *)queryParts
                 topK:(NSInteger)topK
       metadataFilter:(nullable NSDictionary<NSString *, NSString *> *)metadataFilter
                error:(NSError **)error {
  int64_t timestampVal = _syntheticTimestamp++;
  mediapipe::Timestamp timestamp(timestampVal);
  _statsLogger->RecordCpuInputArrival(timestamp);

  NSArray<NSNumber *> *queryEmbedding = [self embedContent:queryParts error:error];
  if (!queryEmbedding) {
    _statsLogger->RecordInvocationEnd(timestamp);
    return nil;
  }

  return [self retrieveRecordsForEmbedding:queryEmbedding
                                      topK:topK
                            metadataFilter:metadataFilter
                                 timestamp:timestamp
                                     error:error];
}

- (nullable NSArray<MPPRetrievalResult *> *)retrieveWithEmbedding:
                                                (NSArray<NSNumber *> *)queryEmbedding
                                                             topK:(NSInteger)topK
                                                            error:(NSError **)error {
  return [self retrieveWithEmbedding:queryEmbedding topK:topK metadataFilter:nil error:error];
}

- (nullable NSArray<MPPRetrievalResult *> *)
    retrieveWithEmbedding:(NSArray<NSNumber *> *)queryEmbedding
                     topK:(NSInteger)topK
           metadataFilter:(nullable NSDictionary<NSString *, NSString *> *)metadataFilter
                    error:(NSError **)error {
  int64_t timestampVal = _syntheticTimestamp++;
  mediapipe::Timestamp timestamp(timestampVal);
  _statsLogger->RecordCpuInputArrival(timestamp);

  return [self retrieveRecordsForEmbedding:queryEmbedding
                                      topK:topK
                            metadataFilter:metadataFilter
                                 timestamp:timestamp
                                     error:error];
}

- (nullable NSArray<MPPRetrievalResult *> *)
    retrieveRecordsForEmbedding:(NSArray<NSNumber *> *)queryEmbedding
                           topK:(NSInteger)topK
                 metadataFilter:(nullable NSDictionary<NSString *, NSString *> *)metadataFilter
                      timestamp:(mediapipe::Timestamp)timestamp
                          error:(NSError **)error {
  NSArray<MPPRetrievalRecord *> *searchResults = [_vectorStore searchWithEmbedding:queryEmbedding
                                                                              topK:(int)topK
                                                                    metadataFilter:metadataFilter
                                                                             error:error];
  if (!searchResults) {
    _statsLogger->RecordInvocationEnd(timestamp);
    return nil;
  }

  NSMutableArray<MPPRetrievalRecord *> *results = [NSMutableArray array];
  NSMutableOrderedSet<NSString *> *uniqueParentIds = [NSMutableOrderedSet orderedSet];

  for (MPPRetrievalRecord *record in searchResults) {
    if (record.parentId.length > 0) {
      [uniqueParentIds addObject:record.parentId];
    }
  }

  NSMutableDictionary<NSString *, MPPRetrievalRecord *> *parentRecordsMap =
      [NSMutableDictionary dictionary];
  if (uniqueParentIds.count > 0) {
    NSArray<MPPRetrievalRecord *> *parentRecords =
        [_vectorStore getRecordsWithIds:uniqueParentIds.array error:error];
    if (!parentRecords) {
      _statsLogger->RecordInvocationEnd(timestamp);
      return nil;
    }
    for (MPPRetrievalRecord *parent in parentRecords) {
      parentRecordsMap[parent.recordId] = parent;
    }
  }

  NSMutableSet<NSString *> *seenParents = [NSMutableSet set];
  for (MPPRetrievalRecord *record in searchResults) {
    if (record.parentId.length > 0) {
      if (![seenParents containsObject:record.parentId]) {
        [seenParents addObject:record.parentId];
        MPPRetrievalRecord *parent = parentRecordsMap[record.parentId];
        if (parent) {
          // Populate the parent record with the highest-scoring child record's embedding
          // so that the cosine similarity is based on the matching child chunk.
          MPPRetrievalRecord *parentWithChildEmbedding =
              [[MPPRetrievalRecord alloc] initWithRecordId:parent.recordId
                                                   content:parent.content
                                                embeddings:record.embeddings
                                                  metadata:parent.metadata
                                                  parentId:parent.parentId
                                                  childIds:parent.childIds];
          [results addObject:parentWithChildEmbedding];
        }
      }
    } else {
      if (![seenParents containsObject:record.recordId]) {
        [seenParents addObject:record.recordId];
        [results addObject:record];
      }
    }
  }

  NSMutableArray<MPPRetrievalResult *> *clientResults =
      [NSMutableArray arrayWithCapacity:results.count];
  MPPEmbedding *query = [[MPPEmbedding alloc] initWithFloatEmbedding:queryEmbedding
                                                  quantizedEmbedding:nil
                                                           headIndex:0
                                                            headName:nil];
  for (MPPRetrievalRecord *record in results) {
    MPPEmbedding *stored = [[MPPEmbedding alloc] initWithFloatEmbedding:record.embeddings
                                                     quantizedEmbedding:nil
                                                              headIndex:0
                                                               headName:nil];
    NSNumber *scoreNum = [MPPCosineSimilarity computeBetweenEmbedding1:query
                                                         andEmbedding2:stored
                                                                 error:error];
    double score = scoreNum ? scoreNum.doubleValue : 0.0;
    MPPRetrievalResult *resultObj = [[MPPRetrievalResult alloc] initWithRecordId:record.recordId
                                                                         content:record.content
                                                                      embeddings:record.embeddings
                                                                        metadata:record.metadata
                                                                           score:score];
    [clientResults addObject:resultObj];
  }

  _statsLogger->RecordInvocationEnd(timestamp);

  [clientResults
      sortUsingComparator:^NSComparisonResult(MPPRetrievalResult *obj1, MPPRetrievalResult *obj2) {
        if (obj1.score > obj2.score) {
          return NSOrderedAscending;
        } else if (obj1.score < obj2.score) {
          return NSOrderedDescending;
        }
        return NSOrderedSame;
      }];
  return clientResults;
}

- (BOOL)deleteWithIds:(NSArray<NSString *> *)ids error:(NSError **)error {
  return [_vectorStore deleteWithIds:ids error:error];
}

- (BOOL)deleteWithMetadataFilter:(NSDictionary<NSString *, NSString *> *)metadataFilter
                           error:(NSError **)error {
  if (metadataFilter == nil || metadataFilter.count == 0) {
    [MPPCommonUtils createCustomError:error
                             withCode:MPPTasksErrorCodeInvalidArgumentError
                          description:@"metadataFilter cannot be nil or empty."];
    return NO;
  }
  return [_vectorStore deleteWithMetadataFilter:metadataFilter error:error];
}

- (BOOL)deleteAllRecordsWithError:(NSError **)error {
  return [_vectorStore deleteAllRecords:error];
}

- (nullable NSArray<NSString *> *)getAllRecordIdsWithError:(NSError **)error {
  return [_vectorStore recordIdentifiers:error];
}

@end
