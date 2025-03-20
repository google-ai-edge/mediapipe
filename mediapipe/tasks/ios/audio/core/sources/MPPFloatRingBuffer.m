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

#import "mediapipe/tasks/ios/audio/core/sources/MPPFloatRingBuffer.h"
#import "mediapipe/tasks/ios/common/sources/MPPCommon.h"
#import "mediapipe/tasks/ios/common/utils/sources/MPPCommonUtils.h"

@implementation MPPFloatRingBuffer {
  // Index of the oldest data in the buffer.
  //
  // During a read operation, data is read starting from this location of the underlying flat buffer
  // of the ring buffer wrapping to the beginning in a circular fashion.
  //
  // During a write operation, data is written starting from this location of the underlying flat
  // buffer wrapping to the beginning in a circular fashion.
  NSUInteger _nextIndex;
  MPPFloatBuffer *_buffer;
}

- (instancetype)initWithLength:(NSUInteger)length {
  self = [self init];
  if (self) {
    _buffer = [[MPPFloatBuffer alloc] initWithLength:length];
  }
  return self;
}

- (BOOL)loadFloatBuffer:(MPPFloatBuffer *)floatBuffer
                 offset:(NSUInteger)offset
                 length:(NSUInteger)length
                  error:(NSError **)error {
  NSUInteger lengthToCopy = length;
  NSUInteger newOffset = offset;

  if (offset + length > floatBuffer.length) {
    [MPPCommonUtils
        createCustomError:error
                 withCode:MPPTasksErrorCodeInvalidArgumentError
              description:
                  [NSString stringWithFormat:@"Index out of range. `offset` (%lu) + `length` (%lu) "
                                             @"must be <= `floatBuffer.length` (%lu)",
                                             offset, length, floatBuffer.length]];
    return NO;
  }

  // If buffer can't hold all the data, only keep the most recent data of size `buffer.size`.
  if (length >= _buffer.length) {
    lengthToCopy = _buffer.length;
    newOffset = offset + (length - _buffer.length);
  }

  if (_nextIndex + lengthToCopy < _buffer.length) {
    // Since length of new data to be copied starting from `_nextIndex` is within the bounds of the
    // flat buffer, copy data directly to the locations beginning at `_nextIndex`.
    memcpy(_buffer.data + _nextIndex, floatBuffer.data + newOffset, sizeof(float) * lengthToCopy);
  } else {
    // If length of new data to be copied starting from `_nextIndex` exceeds the bounds of the flat
    // buffer, new data should be wrapped to the beginning of the buffer by performing copy in two
    // chunks. First chunk of the data is copied to the end of the flat buffer.
    NSUInteger firstChunkLength = _buffer.length - _nextIndex;
    memcpy(_buffer.data + _nextIndex, floatBuffer.data + newOffset,
           sizeof(float) * firstChunkLength);

    // Second chunk of the new data must be copied to the beginning of the ring buffer.
    memcpy(_buffer.data, floatBuffer.data + newOffset + firstChunkLength,
           sizeof(float) * (lengthToCopy - firstChunkLength));
  }

  // Wrap `_nextIndex` to the new beginning (current oldest element) of the ring buffer.
  _nextIndex = (_nextIndex + lengthToCopy) % _buffer.length;

  return YES;
}

- (MPPFloatBuffer *)floatBuffer {
  return [self floatBufferWithOffset:0 length:self.length error:nil];
}

- (nullable MPPFloatBuffer *)floatBufferWithOffset:(NSUInteger)offset
                                            length:(NSUInteger)length
                                             error:(NSError **)error {
  if (offset + length > _buffer.length) {
    [MPPCommonUtils
        createCustomError:error
                 withCode:MPPTasksErrorCodeInvalidArgumentError
              description:
                  [NSString stringWithFormat:@"Index out of range. `offset` (%lu) + `length` (%lu) "
                                             @"must be <= `length` (%lu)",
                                             offset, length, self.length]];
    return nil;
  }

  MPPFloatBuffer *bufferToReturn = [[MPPFloatBuffer alloc] initWithLength:length];

  // Return buffer in correct order.
  // Compute wrapped offset in flat ring buffer array.
  NSInteger correctOffset = (_nextIndex + offset) % _buffer.length;

  if ((correctOffset + length) <= _buffer.length) {
    // If no: of elements to be returned does not exceed the lenegth of the flat ring buffer,
    // directly copy the elements to be returned to the output buffer.
    memcpy(bufferToReturn.data, _buffer.data + correctOffset, sizeof(float) * length);
  } else {
    // If no: elements to be copied exceeds the length of the flat ring buffer, wrap to the
    // beginning of the ring buffer by performing copy in two chunks to the output buffer. Copy the
    // oldest elements of the ring buffer starting at the `correctOffset` to the beginning of the
    // output buffer.
    NSInteger firstChunkLength = _buffer.length - correctOffset;
    memcpy(bufferToReturn.data, _buffer.data + correctOffset, sizeof(float) * firstChunkLength);

    // Next copy the chunk at the beginning of the flat ring buffer array to the end of the output
    // buffer.
    memcpy(bufferToReturn.data + firstChunkLength, _buffer.data,
           sizeof(float) * (length - firstChunkLength));
  }

  return bufferToReturn;
}

- (void)clear {
  [_buffer clear];
}

- (NSUInteger)length {
  return _buffer.length;
}

@end
