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

#import "mediapipe/tasks/ios/retrieval/semantic_retriever/sources/MPPRetrievalRecord.h"

@implementation MPPRetrievalRecord

- (instancetype)initWithRecordId:(NSString *)recordId
                         content:(NSArray<MPPTaskPart *> *)content
                      embeddings:(NSArray<NSNumber *> *)embeddings
                        metadata:(NSDictionary<NSString *, NSString *> *)metadata
                        parentId:(nullable NSString *)parentId
                        childIds:(nullable NSArray<NSString *> *)childIds {
  self = [super init];
  if (self) {
    _recordId = [recordId copy];
    _content = [content copy];
    _embeddings = embeddings;
    _metadata = [metadata copy];
    _parentId = [parentId copy];
    _childIds = [childIds copy];
  }
  return self;
}

@end
