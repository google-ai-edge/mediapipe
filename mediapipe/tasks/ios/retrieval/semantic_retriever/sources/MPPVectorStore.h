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

#import <Foundation/Foundation.h>
#import "mediapipe/tasks/ios/retrieval/semantic_retriever/sources/MPPRetrievalRecord.h"

NS_ASSUME_NONNULL_BEGIN

/**
 * A protocol that defines the storage backend for semantic retriever.
 */
NS_SWIFT_NAME(VectorStore)
@protocol MPPVectorStore <NSObject>

/**
 * Upserts a list of records in the vector store.
 */
- (BOOL)upsertRecords:(NSArray<MPPRetrievalRecord *> *)records
                error:(NSError **)error NS_SWIFT_NAME(upsertRecords(_:));

/**
 * Deletes the records matching the specified IDs.
 */
- (BOOL)deleteWithIds:(NSArray<NSString *> *)ids
                error:(NSError **)error NS_SWIFT_NAME(delete(withIds:));

/**
 * Deletes all records from the vector store.
 */
- (BOOL)deleteAllRecords:(NSError **)error NS_SWIFT_NAME(deleteAllRecords());

/**
 * Searches the top-K nearest records to the query embedding.
 */
- (nullable NSArray<MPPRetrievalRecord *> *)searchWithEmbedding:
                                                (NSArray<NSNumber *> *)queryEmbedding
                                                           topK:(NSInteger)topK
                                                          error:(NSError **)error
    NS_SWIFT_NAME(search(withEmbedding:topK:));

/**
 * Searches the top-K nearest records to the query embedding with optional metadata filtering.
 */
- (nullable NSArray<MPPRetrievalRecord *> *)
    searchWithEmbedding:(NSArray<NSNumber *> *)queryEmbedding
                   topK:(NSInteger)topK
         metadataFilter:(nullable NSDictionary<NSString *, NSString *> *)metadataFilter
                  error:(NSError **)error NS_SWIFT_NAME(search(withEmbedding:topK:metadataFilter:));

/**
 * Deletes records matching the specified metadata filter.
 */
- (BOOL)deleteWithMetadataFilter:(NSDictionary<NSString *, NSString *> *)metadataFilter
                           error:(NSError **)error NS_SWIFT_NAME(delete(withMetadataFilter:));

/**
 * Retrieves the records matching the specified IDs.
 */
- (nullable NSArray<MPPRetrievalRecord *> *)getRecordsWithIds:(NSArray<NSString *> *)ids
                                                        error:(NSError **)error
    NS_SWIFT_NAME(getRecordsWithIds(_:));

/**
 * Retrieves a single record matching the specified ID.
 */
- (nullable MPPRetrievalRecord *)getRecordWithId:(NSString *)recordId
                                           error:(NSError **)error
    NS_SWIFT_NAME(getRecord(withId:));

/**
 * Retrieves all unique top-level record identifiers in the vector store.
 */
- (nullable NSArray<NSString *> *)recordIdentifiers:(NSError **)error
    NS_SWIFT_NAME(recordIdentifiers());

@end

NS_ASSUME_NONNULL_END
