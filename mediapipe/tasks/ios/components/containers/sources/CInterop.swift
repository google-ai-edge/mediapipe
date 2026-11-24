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

extension Array {
  /// Creates an array by mapping `count` elements from a nullable C array pointer, or an empty
  /// array when `cArray` is `nil` or `count <= 0`.
  package init<CElement>(
    cArray: UnsafePointer<CElement>?,
    count: some BinaryInteger,
    transform: (CElement) -> Element
  ) {
    guard let cArray, let elementCount = Int(exactly: count), elementCount > 0 else {
      self = []
      return
    }
    self = UnsafeBufferPointer(start: cArray, count: elementCount).map(transform)
  }
}

extension String {
  /// Creates a non-empty Swift string from a nullable UTF-8 C string pointer, returning `nil` when
  /// `cString` is `nil` or empty (`""`).
  package init?(nonEmptyCString cString: UnsafePointer<CChar>?) {
    guard let cString, cString.pointee != 0 else { return nil }
    self.init(cString: cString)
  }
}
