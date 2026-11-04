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
package import third_party_mediapipe_tasks_c_components_processors_embedder_options

/// Embedder options shared across MediaPipe embedding extraction tasks.
public struct EmbedderOptions: Sendable, Equatable, Hashable {
  /// Whether to normalize the returned feature vector with L2 norm.
  public var l2Normalize: Bool

  /// Whether the returned embedding should be quantized to bytes via scalar quantization.
  public var quantize: Bool

  private var cOptions: MpEmbedderOptions {
    MpEmbedderOptions(l2_normalize: l2Normalize, quantize: quantize)
  }

  /// Creates `EmbedderOptions` with the specified normalization and quantization settings.
  public init(l2Normalize: Bool = false, quantize: Bool = false) {
    self.l2Normalize = l2Normalize
    self.quantize = quantize
  }

  /// Invokes `body` with a C `MpEmbedderOptions` value.
  package func withCOptions<Result>(
    _ body: (MpEmbedderOptions) throws -> Result
  ) rethrows -> Result {
    try body(cOptions)
  }
}
