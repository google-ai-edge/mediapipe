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
internal import third_party_mediapipe_tasks_c_retrieval_universal_embedder_universal_embedder

#if !MEDIAPIPE_SINGLE_MODULE
  import MediaPipeTasksCore
#endif

/// Activation precision used during `UniversalEmbedder` model execution.
public enum ActivationDataType: Sendable, Equatable, Hashable, CaseIterable {
  /// Uses the precision configured in the model metadata.
  case `default`
  /// 32-bit floating-point activations.
  case float32
  /// 16-bit floating-point activations.
  case float16
  /// 16-bit integer activations.
  case int16
  /// 8-bit integer activations.
  case int8

  var cActivationDataType: MpActivationDataType {
    switch self {
    case .default:
      return kMpActivationDataTypeDefault
    case .float32:
      return kMpActivationDataTypeFloat32
    case .float16:
      return kMpActivationDataTypeFloat16
    case .int16:
      return kMpActivationDataTypeInt16
    case .int8:
      return kMpActivationDataTypeInt8
    }
  }
}

/// Configuration options for `UniversalEmbedder`.
public struct UniversalEmbedderOptions: Sendable, Equatable, Hashable {
  /// Shared model asset and default accelerator configuration.
  public var baseOptions: BaseOptions
  /// Whether output embedding vectors are L2-normalized (`true` by default).
  public var l2Normalize: Bool
  /// Maximum sequence length in tokens for text encoder signatures, or `0` for model defaults.
  public var maxInputLength: Int
  /// Vision tokens per image for vision encoder signatures, or `0` for model defaults.
  public var visionTokensPerImage: Int
  /// Activation precision override for encoder execution.
  public var activationDataType: ActivationDataType
  /// Directory path for storing compiled model cache artifacts, or `nil` to disable disk caching.
  public var cacheDirectory: String?
  /// Hardware backend override for the text encoder (`.default` inherits `baseOptions.delegate`).
  public var textDelegate: Delegate
  /// Hardware backend override for the vision encoder (`.default` inherits `baseOptions.delegate`).
  public var visionDelegate: Delegate
  /// Hardware backend override for the audio encoder (`.default` inherits `baseOptions.delegate`).
  public var audioDelegate: Delegate

  /// Creates configuration options for `UniversalEmbedder`.
  public init(
    baseOptions: BaseOptions = BaseOptions(),
    l2Normalize: Bool = true,
    maxInputLength: Int = 0,
    visionTokensPerImage: Int = 0,
    activationDataType: ActivationDataType = .default,
    cacheDirectory: String? = nil,
    textDelegate: Delegate = .default,
    visionDelegate: Delegate = .default,
    audioDelegate: Delegate = .default
  ) {
    self.baseOptions = baseOptions
    self.l2Normalize = l2Normalize
    self.maxInputLength = maxInputLength
    self.visionTokensPerImage = visionTokensPerImage
    self.activationDataType = activationDataType
    self.cacheDirectory = cacheDirectory
    self.textDelegate = textDelegate
    self.visionDelegate = visionDelegate
    self.audioDelegate = audioDelegate
  }

  func withCOptions<R>(_ body: (inout MpUniversalEmbedderOptions) throws -> R) throws -> R {
    guard maxInputLength >= 0, let cMaxInputLength = Int32(exactly: maxInputLength),
      visionTokensPerImage >= 0, let cVisionTokens = Int32(exactly: visionTokensPerImage)
    else {
      throw TasksError.invalidArgument(
        "maxInputLength and visionTokensPerImage must be non-negative Int32 values.")
    }
    let nonEmptyCache = (cacheDirectory?.isEmpty == false) ? cacheDirectory : nil
    return try baseOptions.withCOptions { cBase in
      try nonEmptyCache.withCString { cachePtr in
        var cOptions = MpUniversalEmbedderOptions(
          base_options: cBase, l2_normalize: l2Normalize,
          max_input_length: cMaxInputLength, vision_tokens_per_image: cVisionTokens,
          activation_data_type: activationDataType.cActivationDataType, cache_dir: cachePtr,
          text_delegate: textDelegate.cDelegate, vision_delegate: visionDelegate.cDelegate,
          audio_delegate: audioDelegate.cDelegate)
        return try body(&cOptions)
      }
    }
  }
}
