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

/// A contiguous buffer of 32-bit floating-point audio samples.
public final class FloatBuffer {
  private var storage: [Float]

  /// The number of elements in the buffer.
  public var length: Int {
    storage.count
  }

  /// A copy of the buffer's elements.
  public var elements: [Float] {
    storage
  }

  /// Creates a `FloatBuffer` of `length` elements initialized to zero.
  public convenience init(length: Int) {
    self.init(elements: [Float](repeating: 0.0, count: max(0, length)))
  }

  /// Creates a `FloatBuffer` by copying `length` elements from `data`, or zero-initializing when
  /// `data` is `nil`.
  public convenience init(data: UnsafePointer<Float>?, length: Int) {
    let clampedLength = max(0, length)
    if let data, clampedLength > 0 {
      self.init(elements: Array(UnsafeBufferPointer(start: data, count: clampedLength)))
    } else {
      self.init(length: clampedLength)
    }
  }

  /// Creates a `FloatBuffer` copied from `elements`.
  public init(elements: [Float]) {
    self.storage = elements
  }

  /// Resets all elements in the buffer to `0.0`.
  public func clear() {
    storage.withUnsafeMutableBufferPointer { buffer in
      buffer.update(repeating: 0.0)
    }
  }

  /// Returns an independent deep copy of this `FloatBuffer`.
  public func copy() -> FloatBuffer {
    FloatBuffer(elements: storage)
  }

  /// Calls `body` with an immutable `UnsafeBufferPointer` to the buffer's elements.
  public func withUnsafeBufferPointer<Result>(
    _ body: (UnsafeBufferPointer<Float>) throws -> Result
  ) rethrows -> Result {
    try storage.withUnsafeBufferPointer(body)
  }

  /// Calls `body` with an `UnsafeMutableBufferPointer` to the buffer's elements.
  public func withUnsafeMutableBufferPointer<Result>(
    _ body: (UnsafeMutableBufferPointer<Float>) throws -> Result
  ) rethrows -> Result {
    try storage.withUnsafeMutableBufferPointer { try body($0) }
  }
}

extension FloatBuffer: Equatable {
  public static func == (lhs: FloatBuffer, rhs: FloatBuffer) -> Bool {
    lhs.storage == rhs.storage
  }
}
