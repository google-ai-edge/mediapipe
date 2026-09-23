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
#import "mediapipe/tasks/ios/retrieval/semantic_retriever/sources/MPPVectorStore.h"

NS_ASSUME_NONNULL_BEGIN

/**
 * A vector store implementation backed by SQLite database using native SQLite vector engine.
 */
NS_SWIFT_NAME(SqliteVectorStore)
@interface MPPSqliteVectorStore : NSObject <MPPVectorStore>

@property(nonatomic, readonly, copy) NSString *databasePath;

/**
 * Initializes the SqliteVectorStore with a local SQLite database file path and the embedding
 * dimension.
 */
- (instancetype)initWithDatabasePath:(NSString *)databasePath
                  embeddingDimension:(NSInteger)embeddingDimension NS_DESIGNATED_INITIALIZER;

/**
 * Inserts or updates records in the vector store.
 */
- (BOOL)upsertRecords:(NSArray<MPPRetrievalRecord *> *)records
                error:(NSError **)error NS_SWIFT_NAME(upsertRecords(_:));

/**
 * Retrieves the topK nearest records from the vector store matching the query embedding.
 */
- (nullable NSArray<MPPRetrievalRecord *> *)searchWithEmbedding:
                                                (NSArray<NSNumber *> *)queryEmbedding
                                                           topK:(NSInteger)topK
                                                          error:(NSError **)error
    NS_SWIFT_NAME(search(withEmbedding:topK:));

/**
 * Retrieves the topK nearest records from the vector store matching the query embedding with
 * metadata filtering.
 */
- (nullable NSArray<MPPRetrievalRecord *> *)
    searchWithEmbedding:(NSArray<NSNumber *> *)queryEmbedding
                   topK:(NSInteger)topK
         metadataFilter:(nullable NSDictionary<NSString *, NSString *> *)metadataFilter
                  error:(NSError **)error NS_SWIFT_NAME(search(withEmbedding:topK:metadataFilter:));

/**
 * Deletes records from the vector store matching the metadata filter.
 */
- (BOOL)deleteWithMetadataFilter:(NSDictionary<NSString *, NSString *> *)metadataFilter
                           error:(NSError **)error NS_SWIFT_NAME(delete(withMetadataFilter:));

/**
 * Deletes records from the vector store for the given list of IDs.
 */
- (BOOL)deleteWithIds:(NSArray<NSString *> *)ids
                error:(NSError **)error NS_SWIFT_NAME(delete(withIds:));

/**
 * Deletes all records from the vector store.
 */
- (BOOL)deleteAllRecords:(NSError **)error NS_SWIFT_NAME(deleteAllRecords());

/**
 * Retrieves records from the vector store for the given list of IDs.
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

- (instancetype)init NS_UNAVAILABLE;
+ (instancetype)new NS_UNAVAILABLE;

@end

NS_ASSUME_NONNULL_END
