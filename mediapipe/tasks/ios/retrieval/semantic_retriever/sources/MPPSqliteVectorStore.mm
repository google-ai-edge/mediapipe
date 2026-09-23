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

#import "mediapipe/tasks/ios/retrieval/semantic_retriever/sources/MPPSqliteVectorStore.h"

#import "mediapipe/tasks/ios/common/sources/MPPCommon.h"
#import "mediapipe/tasks/ios/common/utils/sources/MPPCommonUtils.h"

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "mediapipe/tasks/cc/retrieval/semantic_retriever/proto/memory.pb.h"
#include "mediapipe/tasks/cc/retrieval/semantic_retriever/proto/vector_stores.pb.h"
#include "mediapipe/tasks/cc/retrieval/semantic_retriever/sqlite_memory_store.h"
#include "mediapipe/tasks/cc/retrieval/semantic_retriever/sqlite_vector_store.h"

@interface MPPSqliteVectorStore () {
  std::unique_ptr<mediapipe::tasks::retrieval::SqliteMemoryStore> _sqliteMemoryStore;
  int _dimension;
}

- (BOOL)ensureDelegateWithError:(NSError **)error;
- (MPPRetrievalRecord *)recordFromCppRecord:
    (const mediapipe::tasks::retrieval::MemoryRecord &)cppRecord;

@end

@implementation MPPSqliteVectorStore

- (instancetype)initWithDatabasePath:(NSString *)databasePath
                  embeddingDimension:(NSInteger)embeddingDimension {
  if (embeddingDimension <= 0) {
    @throw [NSException exceptionWithName:NSInvalidArgumentException
                                   reason:@"Embedding dimension must be greater than 0."
                                 userInfo:nil];
  }
  self = [super init];
  if (self) {
    _databasePath = [databasePath copy];
    _dimension = (int)embeddingDimension;
  }
  return self;
}

- (BOOL)ensureDelegateWithError:(NSError **)error {
  if (_sqliteMemoryStore) {
    return YES;
  }

  auto sqliteVectorStore = std::make_unique<mediapipe::tasks::retrieval::SqliteVectorStore>();
  auto statusInit = sqliteVectorStore->Initialize(_databasePath.UTF8String);
  if (![MPPCommonUtils checkCppError:statusInit toError:error]) {
    return NO;
  }

  auto memoryStore = std::make_unique<mediapipe::tasks::retrieval::SqliteMemoryStore>(
      std::move(sqliteVectorStore));
  auto memory_config = mediapipe::tasks::retrieval::SqliteMemoryStore::GetDefaultConfig(_dimension);
  auto statusMemoryInit = memoryStore->Initialize(memory_config);
  if (![MPPCommonUtils checkCppError:statusMemoryInit toError:error]) {
    return NO;
  }

  _sqliteMemoryStore = std::move(memoryStore);
  return YES;
}

// Required by searchWithEmbedding:topK:error: because vector similarity search delegates to
// SqliteMemoryStore::GetNearestRecords(), which returns mediapipe::tasks::retrieval::MemoryRecord
// rather than raw database column values.
- (MPPRetrievalRecord *)recordFromCppRecord:
    (const mediapipe::tasks::retrieval::MemoryRecord &)cppRecord {
  NSString *recordId = [NSString stringWithUTF8String:cppRecord.record_id().c_str()];
  NSString *contentTypeStr = [NSString stringWithUTF8String:cppRecord.content_type().c_str()];
  NSString *parentId = cppRecord.has_parent_id()
      ? [NSString stringWithUTF8String:cppRecord.parent_id().c_str()]
      : nil;
  NSArray<NSString *> *childIds = nil;

  NSMutableDictionary<NSString *, NSString *> *userMetadata = [NSMutableDictionary dictionary];
  for (const auto &pair : cppRecord.metadata().key_value_pairs()) {
    userMetadata[[NSString stringWithUTF8String:pair.key().c_str()]] =
        [NSString stringWithUTF8String:pair.value().c_str()];
  }

  if (cppRecord.child_ids_size() > 0) {
    NSMutableArray<NSString *> *cIds =
        [NSMutableArray arrayWithCapacity:cppRecord.child_ids_size()];
    for (int i = 0; i < cppRecord.child_ids_size(); ++i) {
      [cIds addObject:[NSString stringWithUTF8String:cppRecord.child_ids(i).c_str()]];
    }
    childIds = cIds;
  }

  NSMutableArray<NSNumber *> *embeddings =
      [NSMutableArray arrayWithCapacity:cppRecord.embeddings_size()];
  for (int i = 0; i < cppRecord.embeddings_size(); ++i) {
    [embeddings addObject:@(cppRecord.embeddings(i))];
  }

  NSMutableArray<MPPTaskPart *> *content = [NSMutableArray array];
  for (const auto &cppPart : cppRecord.parts()) {
    if (cppPart.has_data()) {
      @throw
          [NSException exceptionWithName:NSInternalInconsistencyException
                                  reason:@"Vector store records must not persist raw media bytes."
                                userInfo:nil];
    }
    NSString *text =
        cppPart.has_text() ? [NSString stringWithUTF8String:cppPart.text().c_str()] : @"";
    switch (cppPart.kind()) {
      case mediapipe::tasks::retrieval::Part::IMAGE:
        [content addObject:[[MPPImagePart alloc] initWithFilePath:text]];
        break;
      case mediapipe::tasks::retrieval::Part::AUDIO:
        [content addObject:[[MPPAudioPart alloc] initWithFilePath:text]];
        break;
      case mediapipe::tasks::retrieval::Part::TEXT:
      default:
        [content addObject:[[MPPTextPart alloc] initWithText:text]];
        break;
    }
  }

  // Fallback for single-part records where parts were not explicitly populated.
  if (content.count == 0) {
    NSString *payload = [NSString stringWithUTF8String:cppRecord.text().c_str()];
    MPPTaskPart *part = nil;
    if ([contentTypeStr isEqualToString:@"IMAGE"]) {
      part = [[MPPImagePart alloc] initWithFilePath:payload];
    } else if ([contentTypeStr isEqualToString:@"AUDIO"]) {
      part = [[MPPAudioPart alloc] initWithFilePath:payload];
    } else {
      part = [[MPPTextPart alloc] initWithText:payload];
    }
    [content addObject:part];
  }

  return [[MPPRetrievalRecord alloc] initWithRecordId:recordId
                                              content:content
                                           embeddings:embeddings
                                             metadata:userMetadata
                                             parentId:parentId
                                             childIds:childIds];
}

- (BOOL)upsertRecords:(NSArray<MPPRetrievalRecord *> *)records error:(NSError **)error {
  if (records.count == 0) {
    return YES;
  }

  if (![self ensureDelegateWithError:error]) {
    return NO;
  }

  for (MPPRetrievalRecord *record in records) {
    if ((int)record.embeddings.count != _dimension) {
      [MPPCommonUtils createCustomError:error
                               withCode:MPPTasksErrorCodeInvalidArgumentError
                            description:@"Embedding dimension does not match configured dimension."];
      return NO;
    }
    if (record.content.count == 0) {
      [MPPCommonUtils createCustomError:error
                               withCode:MPPTasksErrorCodeInvalidArgumentError
                            description:@"Record content cannot be empty."];
      return NO;
    }

    mediapipe::tasks::retrieval::MemoryRecord cppRecord;

    NSString *contentType = nil;
    NSString *payload = @"";
    if (record.content.count == 1) {
      MPPTaskPart *part = record.content.firstObject;
      if ([part isKindOfClass:[MPPTextPart class]]) {
        contentType = @"TEXT";
        payload = ((MPPTextPart *)part).text ?: @"";
      } else if ([part isKindOfClass:[MPPImagePart class]]) {
        contentType = @"IMAGE";
        payload = ((MPPImagePart *)part).filePath ?: @"";
      } else if ([part isKindOfClass:[MPPAudioPart class]]) {
        contentType = @"AUDIO";
        payload = ((MPPAudioPart *)part).filePath ?: @"";
      }
      cppRecord.set_text(payload.UTF8String);
    }

    for (NSNumber *val in record.embeddings) {
      cppRecord.add_embeddings(val.floatValue);
    }

    auto *cppMetadata = cppRecord.mutable_metadata();
    if (record.metadata) {
      for (NSString *key in record.metadata) {
        NSString *val = [NSString stringWithFormat:@"%@", record.metadata[key]];
        auto *pair = cppMetadata->add_key_value_pairs();
        pair->set_key(key.UTF8String);
        pair->set_value(val.UTF8String);
      }
    }

    for (MPPTaskPart *part in record.content) {
      auto *cppPart = cppRecord.add_parts();
      if ([part isKindOfClass:[MPPTextPart class]]) {
        cppPart->set_kind(mediapipe::tasks::retrieval::Part::TEXT);
        cppPart->set_text((((MPPTextPart *)part).text ?: @"").UTF8String);
      } else if ([part isKindOfClass:[MPPImagePart class]]) {
        MPPImagePart *imagePart = (MPPImagePart *)part;
        cppPart->set_kind(mediapipe::tasks::retrieval::Part::IMAGE);
        if (imagePart.filePath.length > 0) {
          cppPart->set_text(imagePart.filePath.UTF8String);
        }
      } else if ([part isKindOfClass:[MPPAudioPart class]]) {
        MPPAudioPart *audioPart = (MPPAudioPart *)part;
        cppPart->set_kind(mediapipe::tasks::retrieval::Part::AUDIO);
        if (audioPart.filePath.length > 0) {
          cppPart->set_text(audioPart.filePath.UTF8String);
        }
      }
    }

    if (contentType != nil) {
      cppRecord.set_content_type(contentType.UTF8String);
    }

    if (record.recordId) {
      cppRecord.set_record_id(record.recordId.UTF8String);
    }

    if (record.parentId) {
      cppRecord.set_parent_id(record.parentId.UTF8String);
    }

    for (NSString *childId in record.childIds) {
      cppRecord.add_child_ids(childId.UTF8String);
    }

    auto status = _sqliteMemoryStore->Insert(cppRecord);
    if (![MPPCommonUtils checkCppError:status toError:error]) {
      return NO;
    }
  }

  return YES;
}

- (nullable NSArray<MPPRetrievalRecord *> *)searchWithEmbedding:
                                                (NSArray<NSNumber *> *)queryEmbedding
                                                           topK:(NSInteger)topK
                                                          error:(NSError **)error {
  return [self searchWithEmbedding:queryEmbedding topK:topK metadataFilter:nil error:error];
}

- (nullable NSArray<MPPRetrievalRecord *> *)
    searchWithEmbedding:(NSArray<NSNumber *> *)queryEmbedding
                   topK:(NSInteger)topK
         metadataFilter:(nullable NSDictionary<NSString *, NSString *> *)metadataFilter
                  error:(NSError **)error {
  if (![self ensureDelegateWithError:error]) {
    return nil;
  }

  if (queryEmbedding.count == 0 || topK <= 0) {
    [MPPCommonUtils
        createCustomError:error
                 withCode:MPPTasksErrorCodeInvalidArgumentError
              description:@"queryEmbedding must be non-empty and topK must be positive."];
    return nil;
  }

  if ((int)queryEmbedding.count != _dimension) {
    [MPPCommonUtils createCustomError:error
                             withCode:MPPTasksErrorCodeInvalidArgumentError
                          description:@"Embedding dimension does not match configured dimension."];
    return nil;
  }

  std::vector<float> cppEmbeddings;
  cppEmbeddings.reserve(queryEmbedding.count);
  for (NSNumber *num in queryEmbedding) {
    cppEmbeddings.push_back(num.floatValue);
  }

  absl::flat_hash_map<std::string, std::string> cppFilter;
  if (metadataFilter) {
    for (NSString *key in metadataFilter) {
      NSString *val = metadataFilter[key];
      cppFilter[key.UTF8String] = val.UTF8String;
    }
  }

  auto cppRecordsStatus =
      _sqliteMemoryStore->GetNearestRecords(cppEmbeddings, (int)topK, 0.0f, cppFilter);
  if (![MPPCommonUtils checkCppError:cppRecordsStatus.status() toError:error]) {
    return nil;
  }

  NSMutableArray<MPPRetrievalRecord *> *records =
      [NSMutableArray arrayWithCapacity:cppRecordsStatus->size()];
  for (const auto &cppRecord : *cppRecordsStatus) {
    [records addObject:[self recordFromCppRecord:cppRecord]];
  }

  return records;
}

- (BOOL)deleteWithIds:(NSArray<NSString *> *)ids error:(NSError **)error {
  if (ids.count == 0) {
    return YES;
  }
  if (![self ensureDelegateWithError:error]) {
    return NO;
  }
  for (NSString *idStr in ids) {
    auto status = _sqliteMemoryStore->DeleteById(idStr.UTF8String ?: "");
    if (![MPPCommonUtils checkCppError:status toError:error]) return NO;
  }
  return YES;
}

- (BOOL)deleteWithMetadataFilter:(NSDictionary<NSString *, NSString *> *)metadataFilter
                           error:(NSError **)error {
  if (metadataFilter.count == 0) {
    return YES;
  }
  if (![self ensureDelegateWithError:error]) {
    return NO;
  }

  absl::flat_hash_map<std::string, std::string> cppFilter;
  for (NSString *key in metadataFilter) {
    NSString *val = metadataFilter[key];
    cppFilter[key.UTF8String] = val.UTF8String;
  }

  auto status = _sqliteMemoryStore->DeleteByMetadata(cppFilter);
  return [MPPCommonUtils checkCppError:status toError:error];
}

- (BOOL)deleteAllRecords:(NSError **)error {
  if (![self ensureDelegateWithError:error]) {
    return NO;
  }
  auto status = _sqliteMemoryStore->DeleteAll();
  return [MPPCommonUtils checkCppError:status toError:error];
}

- (nullable NSArray<MPPRetrievalRecord *> *)getRecordsWithIds:(NSArray<NSString *> *)ids
                                                        error:(NSError **)error {
  if (ids.count == 0) {
    return @[];
  }

  if (![self ensureDelegateWithError:error]) {
    return nil;
  }

  std::vector<std::string> cpp_ids;
  cpp_ids.reserve(ids.count);
  for (NSString *idStr in ids) {
    cpp_ids.push_back(idStr.UTF8String ?: "");
  }

  auto statusOrResults = _sqliteMemoryStore->GetRecords(cpp_ids);
  if (![MPPCommonUtils checkCppError:statusOrResults.status() toError:error]) {
    return nil;
  }

  NSMutableArray<MPPRetrievalRecord *> *records =
      [NSMutableArray arrayWithCapacity:statusOrResults->size()];
  for (const auto &rec : *statusOrResults) {
    [records addObject:[self recordFromCppRecord:rec]];
  }

  return records;
}

- (nullable MPPRetrievalRecord *)getRecordWithId:(NSString *)recordId error:(NSError **)error {
  if (!recordId) {
    return nil;
  }
  NSArray<MPPRetrievalRecord *> *records = [self getRecordsWithIds:@[ recordId ] error:error];
  return records.firstObject;
}

- (nullable NSArray<NSString *> *)recordIdentifiers:(NSError **)error {
  if (![self ensureDelegateWithError:error]) {
    return nil;
  }

  auto statusOrIds = _sqliteMemoryStore->GetAllRecordIds();
  if (![MPPCommonUtils checkCppError:statusOrIds.status() toError:error]) {
    return nil;
  }

  NSMutableArray<NSString *> *identifiers = [NSMutableArray arrayWithCapacity:statusOrIds->size()];
  for (const std::string &id : *statusOrIds) {
    [identifiers addObject:[NSString stringWithUTF8String:id.c_str()]];
  }
  return identifiers;
}

@end
