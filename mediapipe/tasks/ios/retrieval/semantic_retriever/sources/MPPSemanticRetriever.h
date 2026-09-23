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
#import "mediapipe/tasks/ios/core/sources/MPPTaskPart.h"
#import "mediapipe/tasks/ios/retrieval/semantic_retriever/sources/MPPRetrievalResult.h"
#import "mediapipe/tasks/ios/retrieval/semantic_retriever/sources/MPPSemanticRetrieverComponents.h"

NS_ASSUME_NONNULL_BEGIN

extern const NSInteger MPPSemanticRetrieverDefaultEmbeddingDimension;

/**
 * Performs on-device semantic search, vector storage, and query retrieval on text, image, and audio
 * files.
 */
NS_SWIFT_NAME(SemanticRetriever)
@interface MPPSemanticRetriever : NSObject

/**
 * Creates a new instance of `MPPSemanticRetriever` from the given `MPPSemanticRetrieverComponents`.
 *
 * @param components The components to configure the semantic retriever.
 * @param error An optional error parameter populated when there is an error in initializing.
 * @return A new instance of `MPPSemanticRetriever`. `nil` if there is an error.
 */
+ (nullable instancetype)createFromComponents:(MPPSemanticRetrieverComponents *)components
                                        error:(NSError **)error
    NS_SWIFT_NAME(create(fromComponents:));

/**
 * Initializes a new instance of `MPPSemanticRetriever` from the given
 * `MPPSemanticRetrieverComponents`.
 *
 * @param components The components to configure the semantic retriever.
 * @param error An optional error parameter populated when there is an error in initializing.
 * @return A new instance of `MPPSemanticRetriever`. `nil` if there is an error.
 */
- (nullable instancetype)initWithComponents:(MPPSemanticRetrieverComponents *)components
                                      error:(NSError **)error NS_DESIGNATED_INITIALIZER;

/**
 * Inserts a text document into the vector store.
 */
- (BOOL)insertDocumentWithId:(NSString *)recordId
                        text:(NSString *)text
                       error:(NSError **)error NS_SWIFT_NAME(insertDocument(withId:text:));

/**
 * Inserts a text document with associated metadata into the vector store.
 */
- (BOOL)insertDocumentWithId:(NSString *)recordId
                        text:(NSString *)text
                    metadata:(NSDictionary<NSString *, NSString *> *)metadata
                       error:(NSError **)error NS_SWIFT_NAME(insertDocument(withId:text:metadata:));

/**
 * Inserts an image into the vector store.
 */
- (BOOL)insertImageWithId:(NSString *)recordId
                 filePath:(NSString *)filePath
                    error:(NSError **)error NS_SWIFT_NAME(insertImage(withId:filePath:));

/**
 * Inserts an image with associated metadata into the vector store.
 */
- (BOOL)insertImageWithId:(NSString *)recordId
                 filePath:(NSString *)filePath
                 metadata:(NSDictionary<NSString *, NSString *> *)metadata
                    error:(NSError **)error NS_SWIFT_NAME(insertImage(withId:filePath:metadata:));

/**
 * Inserts an audio file into the vector store.
 */
- (BOOL)insertAudioWithId:(NSString *)recordId
                 filePath:(NSString *)filePath
                    error:(NSError **)error NS_SWIFT_NAME(insertAudio(withId:filePath:));

/**
 * Inserts an audio file with associated metadata into the vector store.
 */
- (BOOL)insertAudioWithId:(NSString *)recordId
                 filePath:(NSString *)filePath
                 metadata:(NSDictionary<NSString *, NSString *> *)metadata
                    error:(NSError **)error NS_SWIFT_NAME(insertAudio(withId:filePath:metadata:));

/**
 * Inserts multi-modal parts into the vector store.
 */
- (BOOL)insertContentWithId:(NSString *)recordId
                      parts:(NSArray<MPPTaskPart *> *)parts
                      error:(NSError **)error NS_SWIFT_NAME(insertContent(withId:parts:));

/**
 * Inserts multi-modal parts with associated metadata into the vector store.
 */
- (BOOL)insertContentWithId:(NSString *)recordId
                      parts:(NSArray<MPPTaskPart *> *)parts
                   metadata:(NSDictionary<NSString *, NSString *> *)metadata
                      error:(NSError **)error NS_SWIFT_NAME(insertContent(withId:parts:metadata:));

/**
 * Retrieves the topK nearest records matching the query string.
 */
- (nullable NSArray<MPPRetrievalResult *> *)retrieveWithText:(NSString *)queryText
                                                        topK:(NSInteger)topK
                                                       error:(NSError **)error
    NS_SWIFT_NAME(retrieve(withText:topK:));

/**
 * Retrieves the topK nearest records matching the query string and metadata filter.
 */
- (nullable NSArray<MPPRetrievalResult *> *)
    retrieveWithText:(NSString *)queryText
                topK:(NSInteger)topK
      metadataFilter:(nullable NSDictionary<NSString *, NSString *> *)metadataFilter
               error:(NSError **)error NS_SWIFT_NAME(retrieve(withText:topK:metadataFilter:));

/**
 * Retrieves the topK nearest records matching the multi-modal query parts.
 */
- (nullable NSArray<MPPRetrievalResult *> *)retrieveWithParts:(NSArray<MPPTaskPart *> *)queryParts
                                                         topK:(NSInteger)topK
                                                        error:(NSError **)error
    NS_SWIFT_NAME(retrieve(withParts:topK:));

/**
 * Retrieves the topK nearest records matching the multi-modal query parts and metadata filter.
 */
- (nullable NSArray<MPPRetrievalResult *> *)
    retrieveWithParts:(NSArray<MPPTaskPart *> *)queryParts
                 topK:(NSInteger)topK
       metadataFilter:(nullable NSDictionary<NSString *, NSString *> *)metadataFilter
                error:(NSError **)error NS_SWIFT_NAME(retrieve(withParts:topK:metadataFilter:));

/**
 * Retrieves the topK nearest records matching the precomputed query embedding.
 */
- (nullable NSArray<MPPRetrievalResult *> *)retrieveWithEmbedding:
                                                (NSArray<NSNumber *> *)queryEmbedding
                                                             topK:(NSInteger)topK
                                                            error:(NSError **)error
    NS_SWIFT_NAME(retrieve(withEmbedding:topK:));

/**
 * Retrieves the topK nearest records matching the precomputed query embedding and metadata filter.
 */
- (nullable NSArray<MPPRetrievalResult *> *)
    retrieveWithEmbedding:(NSArray<NSNumber *> *)queryEmbedding
                     topK:(NSInteger)topK
           metadataFilter:(nullable NSDictionary<NSString *, NSString *> *)metadataFilter
                    error:(NSError **)error
    NS_SWIFT_NAME(retrieve(withEmbedding:topK:metadataFilter:));

/**
 * Deletes records from the vector store for the given list of IDs.
 */
- (BOOL)deleteWithIds:(NSArray<NSString *> *)ids
                error:(NSError **)error NS_SWIFT_NAME(delete(withIds:));

/**
 * Deletes records from the vector store matching the metadata filter.
 */
- (BOOL)deleteWithMetadataFilter:(NSDictionary<NSString *, NSString *> *)metadataFilter
                           error:(NSError **)error NS_SWIFT_NAME(delete(withMetadataFilter:));

/**
 * Deletes all records from the vector store.
 */
- (BOOL)deleteAllRecordsWithError:(NSError **)error NS_SWIFT_NAME(deleteAllRecords());

/**
 * Retrieves all unique top-level record identifiers in the vector store.
 */
- (nullable NSArray<NSString *> *)getAllRecordIdsWithError:(NSError **)error
    NS_SWIFT_NAME(getAllRecordIds());

- (instancetype)init NS_UNAVAILABLE;
+ (instancetype)new NS_UNAVAILABLE;

@end

NS_ASSUME_NONNULL_END
