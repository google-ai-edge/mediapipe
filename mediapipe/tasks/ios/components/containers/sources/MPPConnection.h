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

#import <Foundation/Foundation.h>

NS_ASSUME_NONNULL_BEGIN

/** The value class representing a landmark connection. */
NS_SWIFT_NAME(Connection)
@interface MPPConnection : NSObject

@property(nonatomic, readonly) NSUInteger start;

@property(nonatomic, readonly) NSUInteger end;

/**
 * Initializes a new `MPPConnection` with the start and end landmarks integer constants.
 *
 * @param start The integer representing the starting landmark of the connection.
 * @param end The integer representing the ending landmark of the connection.
 *
 * @return An instance of `MPPConnection` initialized with the given start and end landmarks integer
 * constants.
 */
- (instancetype)initWithStart:(NSUInteger)start end:(NSUInteger)end NS_DESIGNATED_INITIALIZER;

- (instancetype)init NS_UNAVAILABLE;

+ (instancetype)new NS_UNAVAILABLE;

@end

NS_ASSUME_NONNULL_END
