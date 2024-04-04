// Copyright 2024 The MediaPipe Authors.
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

/** A wrapper class to store pointer to a float array and its size. */
@interface MPPFloatBuffer : NSObject <NSCopying>

/** Capacity of the array in number of elements. */
@property(nonatomic, readonly) NSUInteger length;

/** Pointer to float array wrapped by `FloatBuffer`. */
@property(nonatomic, readonly) float *data;

/**
 * Initializes a new `FloatBuffer` by copying the elements of the given pointer to a float array.
 * If `data` = `NULL`, all elements of the buffer are initialized to zero.
 *
 * @param data A pointer to a float array whose values are to be copied into the buffer.
 * @param length Length of the float array.
 *
 * @return A new instance of `FloatBuffer` initialized with the elements of the given float array.
 */
- (instancetype)initWithData:(nullable const float *)data length:(NSUInteger)length;

/**
 * Initializes a `FloatBuffer` of the specified length with zeros.
 *
 * @param length Number of elements the `FloatBuffer` can hold.
 *
 * @return A new instance of `FloatBuffer` of the given length and all elements initialized to zero.
 */
- (instancetype)initWithLength:(NSUInteger)length;

/** Clears the `FloatBuffer` by setting all elements to zero */
- (void)clear;

@end

NS_ASSUME_NONNULL_END
