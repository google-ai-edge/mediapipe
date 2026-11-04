// Copyright 2026 The MediaPipe Authors.
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

import Foundation

#if !MEDIAPIPE_SINGLE_MODULE
  import MediaPipeTasksCore
#endif

/// Circular floating-point sample buffer retaining the most recent `length` elements in order.
public final class FloatRingBuffer {
  private var storage: [Float]
  private var nextIndex = 0

  /// The capacity of the ring buffer.
  public var length: Int {
    storage.count
  }

  /// A copy of all ring buffer elements ordered from oldest to newest.
  public var floatBuffer: FloatBuffer {
    copyElements(offset: 0, length: length)
  }

  /// Creates a `FloatRingBuffer` of `length` elements initialized to zero.
  public init(length: Int) {
    self.storage = [Float](repeating: 0.0, count: max(0, length))
  }

  /// Appends `length` elements from `floatBuffer` starting at `offset` into the circular buffer.
  ///
  /// If `length` exceeds the ring buffer capacity, only the most recent `self.length` elements are
  /// retained.
  public func load(floatBuffer: FloatBuffer, offset: Int, length: Int) throws {
    let sourceElements = floatBuffer.elements
    guard
      offset >= 0,
      length >= 0,
      offset <= sourceElements.count,
      length <= sourceElements.count - offset
    else {
      throw TasksError.invalidArgument(
        "Index out of range. `offset` (\(offset)) + `length` (\(length)) must be <= "
          + "`floatBuffer.length` (\(sourceElements.count))."
      )
    }
    guard self.length > 0, length > 0 else { return }

    let slice = sourceElements[offset..<(offset + length)].suffix(self.length)
    let firstChunk = slice.prefix(self.length - nextIndex)
    let secondChunk = slice.dropFirst(firstChunk.count)
    storage.replaceSubrange(nextIndex..<(nextIndex + firstChunk.count), with: firstChunk)
    storage.replaceSubrange(0..<secondChunk.count, with: secondChunk)
    nextIndex = (nextIndex + slice.count) % self.length
  }

  /// Returns `length` chronologically ordered elements starting at `offset` from the oldest
  /// element.
  public func floatBuffer(offset: Int, length: Int) throws -> FloatBuffer {
    guard
      offset >= 0,
      length >= 0,
      offset <= self.length,
      length <= self.length - offset
    else {
      throw TasksError.invalidArgument(
        "Index out of range. `offset` (\(offset)) + `length` (\(length)) must be <= "
          + "`length` (\(self.length))."
      )
    }

    return copyElements(offset: offset, length: length)
  }

  /// Resets all elements in the ring buffer to `0.0` and rewinds the write cursor.
  public func clear() {
    storage.withUnsafeMutableBufferPointer { buffer in
      buffer.update(repeating: 0.0)
    }
    nextIndex = 0
  }

  private func copyElements(offset: Int, length: Int) -> FloatBuffer {
    guard length > 0, self.length > 0 else {
      return FloatBuffer(length: 0)
    }

    let wrappedOffset = (nextIndex + offset) % self.length
    let firstChunk = storage[wrappedOffset...].prefix(length)
    let secondChunk = storage.prefix(length - firstChunk.count)
    return FloatBuffer(elements: Array(firstChunk + secondChunk))
  }
}
