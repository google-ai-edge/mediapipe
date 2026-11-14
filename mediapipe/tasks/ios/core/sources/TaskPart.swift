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

/// UTF-8 text segment within a multimodal content block.
public struct TextPart: Sendable, Equatable, Hashable {
  /// UTF-8 string payload.
  public let text: String
  /// Creates a text content segment.
  public init(text: String) { self.text = text }
}

/// Image segment within a multimodal content block.
public struct ImagePart: Sendable, Equatable, Hashable {
  /// Filesystem path or resource handle for persistent storage or on-demand loading.
  public let filePath: String?
  /// Encoded in-memory image bytes (such as JPEG, PNG, or TGA).
  public let data: Data?

  /// Creates an image segment with optional `filePath` and encoded `data`.
  public init(filePath: String? = nil, data: Data? = nil) {
    self.filePath = filePath
    self.data = data
  }
}

/// Audio segment within a multimodal content block.
public struct AudioPart: Sendable, Equatable, Hashable {
  /// Filesystem path or resource handle for persistent storage or on-demand loading.
  public let filePath: String?
  /// Encoded in-memory audio file bytes (such as WAV or M4A).
  public let data: Data?
  /// Pre-decoded mono PCM float samples in `[-1.0, 1.0]`.
  public let samples: [Float]?

  /// Creates an audio segment with optional `filePath`, encoded `data`, and decoded `samples`.
  public init(filePath: String? = nil, data: Data? = nil, samples: [Float]? = nil) {
    self.filePath = filePath
    self.data = data
    self.samples = samples
  }
}

/// Single modality item within a multimodal content sequence.
public enum TaskPart: Sendable, Equatable, Hashable {
  /// UTF-8 text segment.
  case text(TextPart)
  /// Image segment.
  case image(ImagePart)
  /// Audio segment.
  case audio(AudioPart)

  /// Creates a text segment from `text`.
  public static func text(_ text: String) -> Self { .text(TextPart(text: text)) }
  /// Creates an image segment backed by encoded `data`.
  public static func image(data: Data) -> Self { .image(ImagePart(data: data)) }
  /// Creates an image segment referencing `filePath`.
  public static func image(filePath: String) -> Self { .image(ImagePart(filePath: filePath)) }
  /// Creates an audio segment backed by decoded PCM `samples`.
  public static func audio(samples: [Float]) -> Self { .audio(AudioPart(samples: samples)) }
  /// Creates an audio segment backed by encoded audio `data`.
  public static func audio(data: Data) -> Self { .audio(AudioPart(data: data)) }
  /// Creates an audio segment referencing `filePath`.
  public static func audio(filePath: String) -> Self { .audio(AudioPart(filePath: filePath)) }
}
