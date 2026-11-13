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
package import third_party_mediapipe_tasks_c_core_base_options
package import third_party_mediapipe_tasks_c_text_text_embedder_text_embedder

#if !MEDIAPIPE_SINGLE_MODULE
  @_exported import MediaPipeTasksCore
#endif

/// Configuration options for `TextEmbedder`.
public struct TextEmbedderOptions: Sendable, Equatable, Hashable {
  /// Model asset and hardware delegate configuration.
  public var baseOptions: BaseOptions

  /// Whether to normalize returned feature vectors with L2 norm.
  public var l2Normalize: Bool

  /// Whether to scalar-quantize returned embeddings to bytes.
  public var quantize: Bool

  /// Creates options with the specified model, normalization, and quantization settings.
  public init(
    baseOptions: BaseOptions = BaseOptions(),
    l2Normalize: Bool = false,
    quantize: Bool = false
  ) {
    self.baseOptions = baseOptions
    self.l2Normalize = l2Normalize
    self.quantize = quantize
  }

  package func withCOptions<Result>(
    _ body: (MpTextEmbedderOptions) throws -> Result
  ) rethrows -> Result {
    let embedderOptions = EmbedderOptions(l2Normalize: l2Normalize, quantize: quantize)
    return try baseOptions.withCOptions { cBase in
      try embedderOptions.withCOptions { cEmbedder in
        try body(MpTextEmbedderOptions(base_options: cBase, embedder_options: cEmbedder))
      }
    }
  }
}
