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
 * Represents a single record retrieved from the SemanticRetriever.
 */
NS_SWIFT_NAME(RetrievalRecord)
@interface MPPRetrievalRecord : NSObject

@property(nonatomic, readonly, copy) NSString *recordId;
@property(nonatomic, readonly, copy) NSArray<MPPTaskPart *> *content;
@property(nonatomic, readonly) NSArray<NSNumber *> *embeddings;
@property(nonatomic, readonly, copy) NSDictionary<NSString *, NSString *> *metadata;
@property(nonatomic, readonly, copy, nullable) NSString *parentId;
@property(nonatomic, readonly, copy, nullable) NSArray<NSString *> *childIds;

- (instancetype)initWithRecordId:(NSString *)recordId
                         content:(NSArray<MPPTaskPart *> *)content
                      embeddings:(NSArray<NSNumber *> *)embeddings
                        metadata:(NSDictionary<NSString *, NSString *> *)metadata
                        parentId:(nullable NSString *)parentId
                        childIds:(nullable NSArray<NSString *> *)childIds;

- (instancetype)init NS_UNAVAILABLE;
+ (instancetype)new NS_UNAVAILABLE;

@end

NS_ASSUME_NONNULL_END
