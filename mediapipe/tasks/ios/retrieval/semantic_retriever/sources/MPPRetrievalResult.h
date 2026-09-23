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

NS_ASSUME_NONNULL_BEGIN

/**
 * Represents a single result retrieved from the SemanticRetriever, including similarity/relevance
 * score.
 */
NS_SWIFT_NAME(RetrievalResult)
@interface MPPRetrievalResult : NSObject

@property(nonatomic, readonly, copy) NSString *recordId;
@property(nonatomic, readonly, copy) NSArray<MPPTaskPart *> *content;
@property(nonatomic, readonly, copy, nullable) NSArray<NSNumber *> *embeddings;
@property(nonatomic, readonly, copy) NSDictionary<NSString *, NSString *> *metadata;
@property(nonatomic, readonly) double score;

- (instancetype)initWithRecordId:(NSString *)recordId
                         content:(NSArray<MPPTaskPart *> *)content
                      embeddings:(nullable NSArray<NSNumber *> *)embeddings
                        metadata:(NSDictionary<NSString *, NSString *> *)metadata
                           score:(double)score NS_DESIGNATED_INITIALIZER;

- (instancetype)initWithRecordId:(NSString *)recordId
                         content:(NSArray<MPPTaskPart *> *)content
                        metadata:(NSDictionary<NSString *, NSString *> *)metadata
                           score:(double)score;

- (instancetype)init NS_UNAVAILABLE;
+ (instancetype)new NS_UNAVAILABLE;

@end

NS_ASSUME_NONNULL_END
