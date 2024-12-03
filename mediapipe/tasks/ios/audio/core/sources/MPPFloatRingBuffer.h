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
#import "mediapipe/tasks/ios/audio/core/sources/MPPFloatBuffer.h"

NS_ASSUME_NONNULL_BEGIN

/** A wrapper class which stores a buffer that is written in circular fashion. */
@interface MPPFloatRingBuffer : NSObject

/**
 * A copy of all the internal ring buffer elements in order.
 */
@property(nullable, nonatomic, readonly) MPPFloatBuffer *floatBuffer;

/**
 * Capacity of the ring buffer in number of elements.
 */
@property(nonatomic, readonly) NSUInteger length;

/**
 * Initializes a new `FloatRingBuffer` with the given length. All elements of the `FloatRingBuffer`
 * will be initialized to zero.
 *
 * @param length Total capacity of the ring buffer.
 *
 * @return A new instance of `FloatRingBuffer` with the given length and all elements initialized to
 * zero.
 */
- (instancetype)initWithLength:(NSUInteger)length;

/**
 * Loads a slice of a `FloatBuffer` to the ring buffer. If the float buffer is longer than ring
 * buffer's capacity, samples with lower indices in the array will be ignored.
 *
 * @param floatBuffer A float buffer whose values are to be loaded into the ring buffer.
 * @param offset Offset in float buffer from which elements are to be loaded into the ring buffer.
 * @param length Number of elements to be copied into the ring buffer, starting from `offset`.
 *
 * @return Boolean indicating success or failure of loading operation.
 */
- (BOOL)loadFloatBuffer:(MPPFloatBuffer *)floatBuffer
                 offset:(NSUInteger)offset
                 length:(NSUInteger)length
                  error:(NSError **)error;

/**
 * Returns a `FloatBuffer` with a copy of `length` number of the ring buffer elements in order
 * starting at offset, i.e, `buffer[offset:offset+length]`.
 *
 * @param offset Offset in the ring buffer from which elements are to be returned.
 *
 * @param length Number of elements to be returned.
 *
 * @return A new `FloatBuffer` if `offset + length` is within the bounds of the ring buffer,
 * otherwise nil.
 */
- (nullable MPPFloatBuffer *)floatBufferWithOffset:(NSUInteger)offset
                                            length:(NSUInteger)length
                                             error:(NSError **)error;

/**
 * Clears the `FloatRingBuffer` by setting all the elements to zero .
 */
- (void)clear;

@end

NS_ASSUME_NONNULL_END
