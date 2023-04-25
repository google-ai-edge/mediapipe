// Copyright 2022 The MediaPipe Authors.
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

NS_ASSUME_NONNULL_BEGIN

/**
 * MediaPipe Tasks result base class. Any MediaPipe task result class should extend
 * this class.
 */
NS_SWIFT_NAME(TaskResult)

@interface MPPTaskResult : NSObject <NSCopying>
/**
 * Timestamp that is associated with the task result object.
 */
@property(nonatomic, assign, readonly) NSInteger timestampInMilliseconds;

- (instancetype)init NS_UNAVAILABLE;

- (instancetype)initWithTimestampInMilliseconds:(NSInteger)timestampInMilliseconds
    NS_DESIGNATED_INITIALIZER;

@end

NS_ASSUME_NONNULL_END
