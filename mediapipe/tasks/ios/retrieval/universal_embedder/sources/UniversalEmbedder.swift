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

import AVFoundation
import Foundation
import os
internal import third_party_mediapipe_tasks_c_audio_core_common
internal import third_party_mediapipe_tasks_c_core_mp_status
package import third_party_mediapipe_tasks_c_retrieval_universal_embedder_universal_embedder
internal import third_party_mediapipe_tasks_c_vision_core_image

#if !MEDIAPIPE_SINGLE_MODULE
  @_exported import MediaPipeTasksAudioCore
  @_exported import MediaPipeTasksComponents
  @_exported import MediaPipeTasksCore
  @_exported import MediaPipeTasksVisionCore
#endif

/// Extracts multimodal embeddings from text, images, audio, or composite `TaskPart` sequences.
///
/// `@unchecked Sendable` is safe because `embedder` is immutable after initialization and all
/// native inference calls are serialized through `lock`.
public final class UniversalEmbedder: @unchecked Sendable {
  private let embedder: MpUniversalEmbedderPtr
  package let lock = NSLock()

  /// Underlying C handle exposed to `SemanticRetriever` within package `MediaPipeTasks`.
  package var cEmbedder: MpUniversalEmbedderPtr { embedder }

  /// Creates an embedder configured with `options`.
  public init(options: UniversalEmbedderOptions) throws {
    var created: MpUniversalEmbedderPtr?
    try options.withCOptions { cOptions in
      try MpStatus.check(defaultMessage: "Failed to create UniversalEmbedder.") { errorMsg in
        MpUniversalEmbedderCreate(&cOptions, &created, &errorMsg)
      }
    }
    guard let validEmbedder = created else {
      throw TasksError.internalError("MpUniversalEmbedderCreate returned a null handle.")
    }
    self.embedder = validEmbedder
  }

  /// Creates an embedder from a local `.litertlm` model file at `modelPath` with default options.
  public convenience init(modelPath: String) throws {
    try self.init(
      options: UniversalEmbedderOptions(baseOptions: BaseOptions(modelAssetPath: modelPath)))
  }

  deinit {
    _ = MpUniversalEmbedderClose(embedder, nil)
  }

  /// Computes an embedding for UTF-8 `text`.
  public func embed(text: String) throws -> EmbeddingResult {
    try text.withCString { ptr in
      try invokeEmbed { MpUniversalEmbedderEmbedText(embedder, ptr, &$0, &$1) }
    }
  }

  /// Computes an embedding for compressed `imageBytes` (such as JPEG, PNG, or TGA).
  public func embed(imageBytes: Data) throws -> EmbeddingResult {
    guard !imageBytes.isEmpty, let count = Int32(exactly: imageBytes.count) else {
      throw TasksError.invalidArgument("Image bytes must be non-empty and fit in Int32.")
    }
    return try imageBytes.withUnsafeBytes { raw in
      try raw.withMemoryRebound(to: CChar.self) { b in
        try invokeEmbed { MpUniversalEmbedderEmbedImage(embedder, b.baseAddress, count, &$0, &$1) }
      }
    }
  }

  /// Computes an embedding for an `MPImage`.
  public func embed(image: MPImage) throws -> EmbeddingResult {
    guard image.floatData == nil else {
      throw TasksError.invalidArgument("Float32 MPImage is not supported by UniversalEmbedder.")
    }
    return try image.withCImage { cImage in
      try invokeEmbed { MpUniversalEmbedderEmbedMpImage(embedder, cImage, &$0, &$1) }
    }
  }

  /// Computes an embedding for mono PCM samples in `audioFloatBuffer`.
  public func embed(audioFloatBuffer: FloatBuffer) throws -> EmbeddingResult {
    guard audioFloatBuffer.length > 0, let count = Int32(exactly: audioFloatBuffer.length) else {
      throw TasksError.invalidArgument("Audio float buffer must not be empty.")
    }
    return try audioFloatBuffer.withUnsafeBufferPointer { s in
      try invokeEmbed { MpUniversalEmbedderEmbedAudio(embedder, s.baseAddress, count, &$0, &$1) }
    }
  }

  /// Computes an embedding for `audio`.
  public func embed(audio: AudioData) throws -> EmbeddingResult {
    try audio.withCAudioData { cAudio in
      guard cAudio.audio_data_size > 0, Int32(exactly: cAudio.audio_data_size) != nil else {
        throw TasksError.invalidArgument("AudioData must be non-empty and fit in Int32.")
      }
      return try withUnsafePointer(to: cAudio) { ptr in
        try invokeEmbed { MpUniversalEmbedderEmbedMpAudioData(embedder, ptr, &$0, &$1) }
      }
    }
  }

  /// Computes a joint embedding across multimodal `content` parts.
  public func embed(content: [TaskPart]) throws -> EmbeddingResult {
    guard !content.isEmpty, let n = Int32(exactly: content.count) else {
      throw TasksError.invalidArgument("Multimodal content must not be empty.")
    }
    var cParts = Array(repeating: MpUniversalEmbedderPart(), count: content.count)
    func embedBoundParts(at index: Int) throws -> EmbeddingResult {
      guard index < content.count else {
        return try cParts.withUnsafeBufferPointer { p in
          try invokeEmbed { MpUniversalEmbedderEmbedContent(embedder, p.baseAddress, n, &$0, &$1) }
        }
      }
      switch content[index] {
      case .text(let textPart):
        return try textPart.text.withCString { textPtr in
          cParts[index].type = kMpUniversalEmbedderPartTypeText
          cParts[index].value.text = textPtr
          return try embedBoundParts(at: index + 1)
        }
      case .image(let imagePart):
        let imgData = try Self.resolveImageData(from: imagePart)
        guard let count = Int32(exactly: imgData.count) else {
          throw TasksError.invalidArgument("Image part exceeds Int32 size limit.")
        }
        return try imgData.withUnsafeBytes { raw in
          try raw.withMemoryRebound(to: CChar.self) { bytes in
            cParts[index].type = kMpUniversalEmbedderPartTypeImage
            cParts[index].value.image = MpUniversalEmbedderImagePart(
              bytes: bytes.baseAddress, size: count)
            return try embedBoundParts(at: index + 1)
          }
        }
      case .audio(let audioPart):
        let samples = try Self.resolveAudioSamples(from: audioPart)
        guard let count = Int32(exactly: samples.count) else {
          throw TasksError.invalidArgument("Audio part exceeds Int32 sample count limit.")
        }
        return try samples.withUnsafeBufferPointer { buf in
          cParts[index].type = kMpUniversalEmbedderPartTypeAudio
          cParts[index].value.audio = MpUniversalEmbedderAudioPart(
            samples: buf.baseAddress, size: count)
          return try embedBoundParts(at: index + 1)
        }
      }
    }
    return try embedBoundParts(at: 0)
  }

  /// Computes the cosine similarity between `embedding1` and `embedding2` in `[-1.0, 1.0]`.
  public static func cosineSimilarity(embedding1: Embedding, embedding2: Embedding) throws -> Double
  {
    try CosineSimilarity.compute(between: embedding1, and: embedding2)
  }

  private func invokeEmbed(
    _ call: (inout MpUniversalEmbedderResult, inout UnsafeMutablePointer<CChar>?) -> MpStatus
  ) throws -> EmbeddingResult {
    var cResult = MpUniversalEmbedderResult()
    try lock.withLock {
      try MpStatus.check(defaultMessage: "Embed failed.") { call(&cResult, &$0) }
    }
    defer { MpUniversalEmbedderCloseResult(&cResult) }
    return EmbeddingResult(cResult: cResult)
  }

  private static func resolveImageData(from part: ImagePart) throws -> Data {
    if let data = part.data, !data.isEmpty { return data }
    guard let path = part.filePath, !path.isEmpty else {
      throw TasksError.invalidArgument("ImagePart requires non-empty data or filePath.")
    }
    let loaded = try Data(contentsOf: URL(fileURLWithPath: path))
    guard !loaded.isEmpty else { throw TasksError.invalidArgument("Empty image at \(path).") }
    return loaded
  }

  package static func resolveAudioSamples(from part: AudioPart) throws -> [Float] {
    if let samples = part.samples, !samples.isEmpty { return samples }
    if let data = part.data, !data.isEmpty {
      let ext = URL(fileURLWithPath: part.filePath ?? "").pathExtension
      let tempURL = FileManager.default.temporaryDirectory.appendingPathComponent(
        "\(UUID().uuidString).\(ext.isEmpty ? "wav" : ext)")
      try data.write(to: tempURL)
      defer {
        do {
          try FileManager.default.removeItem(at: tempURL)
        } catch {
          os.Logger(subsystem: "com.google.mediapipe.tasks", category: "UniversalEmbedder")
            .warning("Failed to remove temporary audio file: \(error.localizedDescription)")
        }
      }
      return try loadAudio(from: tempURL)
    }
    guard let path = part.filePath, !path.isEmpty else {
      throw TasksError.invalidArgument("AudioPart requires samples, data, or filePath.")
    }
    return try loadAudio(from: URL(fileURLWithPath: path))
  }

  private static func loadAudio(from url: URL) throws -> [Float] {
    let file = try AVAudioFile(forReading: url, commonFormat: .pcmFormatFloat32, interleaved: false)
    let chunkCapacity: AVAudioFrameCount = 65_536
    guard file.length > 0,
      let buf = AVAudioPCMBuffer(pcmFormat: file.processingFormat, frameCapacity: chunkCapacity)
    else {
      throw TasksError.invalidArgument("Audio file contains no readable PCM frames.")
    }
    var samples: [Float] = []
    while file.framePosition < file.length {
      buf.frameLength = 0
      try file.read(into: buf)
      let frames = Int(buf.frameLength)
      if frames == 0 { break }
      let ch = Int(buf.format.channelCount)
      guard ch > 0, let data = buf.floatChannelData else {
        throw TasksError.internalError("Failed to read Float32 PCM channels from audio file.")
      }
      if ch == 1 {
        samples.append(contentsOf: UnsafeBufferPointer(start: data[0], count: frames))
      } else {
        let scale = 1.0 / Float(ch)
        for i in 0..<frames {
          samples.append((0..<ch).reduce(Float.zero) { $0 + data[$1][i] } * scale)
        }
      }
    }
    guard !samples.isEmpty else {
      throw TasksError.invalidArgument("Audio file contains no readable PCM frames.")
    }
    return samples
  }
}

extension UniversalEmbedder: EmbeddingProvider {}
