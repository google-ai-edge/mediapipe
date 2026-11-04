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
package import third_party_mediapipe_tasks_c_components_processors_classifier_options

/// Classifier options shared across MediaPipe classification tasks.
public struct ClassifierOptions: Sendable, Equatable, Hashable {
  /// The locale to use for display names specified through the TFLite Model Metadata, if any.
  public var displayNamesLocale: String?

  /// The maximum number of top-scored classification results to return. If `< 0`, all available
  /// results are returned.
  public var maxResults: Int

  /// Score threshold to override the one provided in the model metadata. Results below this value
  /// are rejected.
  public var scoreThreshold: Float

  /// Allowlist of category names. If non-empty, classification results whose category name is not
  /// in this list are filtered out. Mutually exclusive with `categoryDenylist`.
  public var categoryAllowlist: [String]

  /// Denylist of category names. If non-empty, classification results whose category name is in
  /// this list are filtered out. Mutually exclusive with `categoryAllowlist`.
  public var categoryDenylist: [String]

  /// Creates `ClassifierOptions` with the given filtering and locale configuration.
  public init(
    displayNamesLocale: String? = nil,
    maxResults: Int = -1,
    scoreThreshold: Float = 0.0,
    categoryAllowlist: [String] = [],
    categoryDenylist: [String] = []
  ) {
    self.displayNamesLocale = displayNamesLocale
    self.maxResults = maxResults
    self.scoreThreshold = scoreThreshold
    self.categoryAllowlist = categoryAllowlist
    self.categoryDenylist = categoryDenylist
  }

  /// Invokes `body` with a C `MpClassifierOptions` struct whose pointer fields remain valid for
  /// the duration of the closure call.
  package func withCOptions<Result>(
    _ body: (MpClassifierOptions) throws -> Result
  ) rethrows -> Result {
    try displayNamesLocale.withCString { localePtr in
      try categoryAllowlist.withCStringArray { allowlistPtr, allowlistCount in
        try categoryDenylist.withCStringArray { denylistPtr, denylistCount in
          var cOptions = MpClassifierOptions()
          cOptions.display_names_locale = localePtr
          cOptions.max_results = Int32(clamping: maxResults)
          cOptions.score_threshold = scoreThreshold
          cOptions.category_allowlist = allowlistPtr
          cOptions.category_allowlist_count = allowlistCount
          cOptions.category_denylist = denylistPtr
          cOptions.category_denylist_count = denylistCount
          return try body(cOptions)
        }
      }
    }
  }
}

extension [String] {
  /// Packs the string elements into a single contiguous null-terminated UTF-8 buffer and invokes
  /// `body` with a pointer to the C string pointer array and its element count.
  package func withCStringArray<Result>(
    _ body: (UnsafeMutablePointer<UnsafePointer<CChar>?>?, UInt32) throws -> Result
  ) rethrows -> Result {
    guard !isEmpty else {
      return try body(nil, 0)
    }

    var contiguousCChars: [CChar] = []
    contiguousCChars.reserveCapacity(reduce(0) { $0 + $1.utf8.count + 1 })
    var offsets: [Int] = []
    offsets.reserveCapacity(count)

    for element in self {
      offsets.append(contiguousCChars.count)
      contiguousCChars.append(contentsOf: element.utf8CString)
    }

    return try contiguousCChars.withUnsafeBufferPointer { charBuffer in
      guard let baseAddress = charBuffer.baseAddress else {
        return try body(nil, 0)
      }
      var cStringPointers: [UnsafePointer<CChar>?] = offsets.map { baseAddress + $0 }
      return try cStringPointers.withUnsafeMutableBufferPointer { pointerBuffer in
        try body(pointerBuffer.baseAddress, UInt32(clamping: pointerBuffer.count))
      }
    }
  }
}
