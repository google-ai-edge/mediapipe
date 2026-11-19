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
import MediaPipeTasksAudioClassifier
import Testing
import third_party_mediapipe_tasks_c_audio_audio_classifier_audio_classifier_lib
import third_party_mediapipe_tasks_c_components_containers_classification_result
import third_party_mediapipe_tasks_c_core_mp_status

private final class BundleToken {}

private final class StreamCollector: @unchecked Sendable {
  private let lock = NSLock()
  private let semaphore = DispatchSemaphore(value: 0)
  private var items: [Result<AudioClassifierResult, any Error>] = []

  func record(_ result: Result<AudioClassifierResult, any Error>) {
    lock.withLock { items.append(result) }
    semaphore.signal()
  }

  func waitFor(count: Int) -> Bool {
    let deadline = DispatchTime.now() + 5.0
    for _ in 0..<count where semaphore.wait(timeout: deadline) == .timedOut { return false }
    return true
  }

  var snapshot: [Result<AudioClassifierResult, any Error>] { lock.withLock { items } }
}

private enum TestFixture {
  static let yamnet = "yamnet_audio_classifier_with_metadata"
  static let speech16k = "speech_16000_hz_mono"
  static let speech48k = "speech_48000_hz_mono"
  static let twoHeadsModel = "two_heads"
  static let twoHeads16k = "two_heads_16000_hz_mono"
  static let twoHeads44k = "two_heads_44100_hz_mono"
  static let samples = 15600
  static let rate = 16000.0
}

struct AudioClassifierSwiftTests {
  private func path(
    _ name: String,
    _ ext: String,
    sourceLocation: SourceLocation = #_sourceLocation
  ) throws -> String {
    try #require(
      Bundle(for: BundleToken.self).path(forResource: name, ofType: ext),
      sourceLocation: sourceLocation
    )
  }

  private func expectInvalidArgument(
    sourceLocation: SourceLocation = #_sourceLocation,
    _ body: () throws -> some Any
  ) throws {
    let error = try #require(throws: TasksError.self, sourceLocation: sourceLocation) {
      _ = try body()
    }
    #expect(error.code == .invalidArgument, sourceLocation: sourceLocation)
  }

  private func loadBlocks(
    _ name: String,
    clip: Bool = false,
    sourceLocation: SourceLocation = #_sourceLocation
  ) throws -> [(AudioData, Int)] {
    let file = try AVAudioFile(
      forReading: URL(fileURLWithPath: try path(name, "wav", sourceLocation: sourceLocation))
    )
    let cap = AVAudioFrameCount(file.length)
    let pcm = try #require(
      AVAudioPCMBuffer(pcmFormat: file.processingFormat, frameCapacity: cap),
      sourceLocation: sourceLocation
    )
    try file.read(into: pcm)
    let ptr = try #require(pcm.floatChannelData, sourceLocation: sourceLocation)[0]
    let (sr, n) = (pcm.format.sampleRate, Int(pcm.frameLength))
    let step = clip ? n : Int(ceil(Double(TestFixture.samples) * sr / TestFixture.rate))
    return try stride(from: 0, to: n, by: step).map { off in
      let len = min(n - off, step)
      let audio = AudioData(format: AudioDataFormat(sampleRate: sr), sampleCount: len)
      try audio.load(buffer: FloatBuffer(data: ptr + off, length: len), offset: 0, length: len)
      return (audio, Int(Double(off) / sr * 1000.0))
    }
  }

  private func opts(
    _ model: String,
    mode: AudioRunningMode = .audioClips,
    maxResults: Int = -1,
    scoreThreshold: Float = 0.0,
    categoryAllowlist: [String] = [],
    categoryDenylist: [String] = [],
    cb: AudioClassifierOptions.AudioStreamResultHandler? = nil
  ) -> AudioClassifierOptions {
    AudioClassifierOptions(
      baseOptions: BaseOptions(modelAssetPath: model),
      runningMode: mode,
      maxResults: maxResults,
      scoreThreshold: scoreThreshold,
      categoryAllowlist: categoryAllowlist,
      categoryDenylist: categoryDenylist,
      audioStreamResultHandler: cb
    )
  }

  private func makeStream(_ model: String) throws -> (AudioClassifier, StreamCollector) {
    let box = StreamCollector()
    let options = opts(model, mode: .audioStream) { box.record($0) }
    return (try AudioClassifier(options: options), box)
  }

  private func topCategories(
    _ options: AudioClassifierOptions,
    _ clip: AudioData,
    sourceLocation: SourceLocation = #_sourceLocation
  ) throws -> [ResultCategory] {
    let res = try AudioClassifier(options: options).classify(audioClip: clip)
    return try #require(
      res.classificationResults.first?.classifications.first,
      sourceLocation: sourceLocation
    ).categories
  }

  private func assertSpeechClipClassification(
    clipName: String,
    sourceLocation: SourceLocation = #_sourceLocation
  ) throws {
    let model = try path(TestFixture.yamnet, "tflite", sourceLocation: sourceLocation)
    let classifier = try AudioClassifier(modelPath: model)
    let clip = try #require(
      loadBlocks(clipName, clip: true, sourceLocation: sourceLocation).first,
      sourceLocation: sourceLocation
    ).0
    let res = try classifier.classify(audioClip: clip)
    #expect(res.timestampInMilliseconds == 0, sourceLocation: sourceLocation)
    #expect(res.classificationResults.count == 5, sourceLocation: sourceLocation)
    try #require(res.classificationResults.count >= 4, sourceLocation: sourceLocation)
    for (i, ts) in [0, 975, 1950, 2925].enumerated() {
      let chunk = res.classificationResults[i]
      let head = try #require(chunk.classifications.first, sourceLocation: sourceLocation)
      #expect(chunk.timestampInMilliseconds == ts, sourceLocation: sourceLocation)
      #expect(head.headName == "scores", sourceLocation: sourceLocation)
      let top = try #require(head.categories.first, sourceLocation: sourceLocation)
      #expect(head.categories.count == 521, sourceLocation: sourceLocation)
      #expect(top.categoryName == "Speech", sourceLocation: sourceLocation)
      #expect(top.score > 0.9, sourceLocation: sourceLocation)
    }
  }

  private func assertSpeechStreamClassification(
    clipName: String,
    sourceLocation: SourceLocation = #_sourceLocation
  ) throws {
    let model = try path(TestFixture.yamnet, "tflite", sourceLocation: sourceLocation)
    let blocks = try loadBlocks(clipName, sourceLocation: sourceLocation)
    let (stream, box) = try makeStream(model)
    for (block, timestamp) in blocks {
      try stream.classifyAsync(audioBlock: block, timestampInMilliseconds: timestamp)
    }
    try stream.close()
    try #require(box.waitFor(count: 5) == true, sourceLocation: sourceLocation)
    try #require(box.snapshot.count == 5, sourceLocation: sourceLocation)
    for (i, ts) in [0, 975, 1950, 2925].enumerated() {
      let res = try box.snapshot[i].get()
      let top = try #require(
        res.classificationResults.first?.classifications.first?.categories.first,
        sourceLocation: sourceLocation
      )
      #expect(res.timestampInMilliseconds == ts, sourceLocation: sourceLocation)
      #expect(top.categoryName == "Speech", sourceLocation: sourceLocation)
      #expect(top.score > 0.9, sourceLocation: sourceLocation)
    }
  }

  @Test func `init rejects empty modelPath`() throws {
    try expectInvalidArgument { try AudioClassifier(modelPath: "") }
  }

  @Test func `init rejects mutually exclusive allowlist and denylist`() throws {
    let model = try path(TestFixture.yamnet, "tflite")
    let conflict = opts(model, categoryAllowlist: ["Speech"], categoryDenylist: ["Speech"])
    try expectInvalidArgument { try AudioClassifier(options: conflict) }
  }

  @Test func `init rejects audioStreamResultHandler in audioClips mode`() throws {
    let model = try path(TestFixture.yamnet, "tflite")
    let withHandler = opts(model) { _ in }
    #expect(withHandler.audioStreamResultHandler != nil)
    #expect(opts(model).audioStreamResultHandler == nil)
    try expectInvalidArgument { try AudioClassifier(options: withHandler) }
  }

  @Test func `init rejects audioStream mode without result handler`() throws {
    let model = try path(TestFixture.yamnet, "tflite")
    try expectInvalidArgument { try AudioClassifier(options: opts(model, mode: .audioStream)) }
  }

  @Test func `createAudioRecord initializes mono record and rejects unsupported channels`() throws {
    let rec = try AudioClassifier.createAudioRecord(
      channelCount: 1,
      sampleRate: TestFixture.rate,
      bufferLength: TestFixture.samples
    )
    #expect(rec.audioDataFormat.sampleRate == TestFixture.rate)
    #expect(rec.bufferLength == TestFixture.samples)

    try expectInvalidArgument {
      try AudioClassifier.createAudioRecord(
        channelCount: 3,
        sampleRate: TestFixture.rate,
        bufferLength: 9
      )
    }
  }

  @Test func `classify 16kHz speech audio clip`() throws {
    try assertSpeechClipClassification(clipName: TestFixture.speech16k)
  }

  @Test func `classify 48kHz speech audio clip`() throws {
    try assertSpeechClipClassification(clipName: TestFixture.speech48k)
  }

  @Test func `classify audio clip with modelAssetBuffer`() throws {
    let model = try path(TestFixture.yamnet, "tflite")
    let speech = try #require(loadBlocks(TestFixture.speech16k, clip: true).first).0
    let modelData = try Data(contentsOf: URL(fileURLWithPath: model))
    let bufferClassifier = try AudioClassifier(
      options: AudioClassifierOptions(baseOptions: BaseOptions(modelAssetBuffer: modelData))
    )
    let bufferTop = try #require(
      bufferClassifier.classify(audioClip: speech).classificationResults.first?.classifications
        .first?.categories.first
    )
    #expect(bufferTop.categoryName == "Speech")
  }

  @Test func `classify silent audio clip returns Silence category`() throws {
    let classifier = try AudioClassifier(modelPath: try path(TestFixture.yamnet, "tflite"))
    let silence = AudioData(
      format: AudioDataFormat(sampleRate: TestFixture.rate),
      sampleCount: 14000
    )
    let silentResult = try classifier.classify(audioClip: silence)
    let silentHead = try #require(silentResult.classificationResults.first)
    #expect(silentHead.classifications.first?.categories.first?.categoryName == "Silence")
  }

  @Test func `classify applies categoryAllowlist and categoryDenylist`() throws {
    let model = try path(TestFixture.yamnet, "tflite")
    let speech = try #require(loadBlocks(TestFixture.speech16k, clip: true).first).0

    let allow = opts(model, maxResults: 1, scoreThreshold: 0.9, categoryAllowlist: ["Speech"])
    let allowCats = try topCategories(allow, speech)
    #expect(allowCats.count == 1)
    #expect(allowCats.first?.categoryName == "Speech")

    let denyCats = try topCategories(opts(model, categoryDenylist: ["Speech"]), speech)
    #expect(denyCats.count == 520)
    #expect(denyCats.first?.categoryName != "Speech")
  }

  @Test func `classify multi-head audio clips across sample rates`() throws {
    let twoHeads = try AudioClassifier(modelPath: try path(TestFixture.twoHeadsModel, "tflite"))
    let clip44k = try #require(loadBlocks(TestFixture.twoHeads44k, clip: true).first).0
    let th44k = try twoHeads.classify(audioClip: clip44k).classificationResults
    try #require(th44k.count == 2)
    try #require(th44k.allSatisfy { $0.classifications.count == 2 } == true)
    #expect(th44k[0].classifications[0].categories.first?.categoryName == "Environmental noise")
    #expect(
      th44k[0].classifications[1].categories.first?.categoryName == "Chestnut-crowned Antpitta"
    )
    #expect(
      th44k[1].classifications[1].categories.first?.categoryName == "White-breasted Wood-Wren"
    )

    let clip16k = try #require(loadBlocks(TestFixture.twoHeads16k, clip: true).first).0
    let th16k = try twoHeads.classify(audioClip: clip16k).classificationResults
    try #require(th16k.isEmpty == false)
    #expect(th16k[0].classifications.count == 2)
    #expect(th16k[0].classifications[0].categories.count == 521)
    #expect(th16k[0].classifications[1].categories.count == 5)
  }

  @Test func `classify rejects mismatched running modes and calls after close`() throws {
    let model = try path(TestFixture.yamnet, "tflite")
    let speech = try #require(loadBlocks(TestFixture.speech16k, clip: true).first).0

    let clipsClassifier = try AudioClassifier(modelPath: model)
    try expectInvalidArgument {
      try clipsClassifier.classifyAsync(audioBlock: speech, timestampInMilliseconds: 0)
    }
    try clipsClassifier.close()
    try expectInvalidArgument { try clipsClassifier.classify(audioClip: speech) }

    let (streamClassifier, _) = try makeStream(model)
    try expectInvalidArgument { try streamClassifier.classify(audioClip: speech) }
    try streamClassifier.close()
  }

  @Test func `close releases the audio stream result handler`() throws {
    let model = try path(TestFixture.yamnet, "tflite")
    weak var collector: StreamCollector?
    let classifier: AudioClassifier
    do {
      // The result handler holds the only strong reference to the collector.
      let (stream, box) = try makeStream(model)
      classifier = stream
      collector = box
    }
    try withExtendedLifetime(classifier) {
      #expect(collector != nil)
      try classifier.close()
      #expect(collector == nil)
    }
  }

  @Test func `classifyAsync streams 16kHz speech blocks`() throws {
    try assertSpeechStreamClassification(clipName: TestFixture.speech16k)
  }

  @Test func `classifyAsync streams 48kHz speech blocks`() throws {
    try assertSpeechStreamClassification(clipName: TestFixture.speech48k)
  }

  @Test func `classifyAsync rejects out-of-range and non-monotonic timestamps`() throws {
    let model = try path(TestFixture.yamnet, "tflite")
    let blocks = try loadBlocks(TestFixture.speech16k)
    try #require(blocks.count >= 2)

    let (stream, box) = try makeStream(model)
    try expectInvalidArgument {
      try stream.classifyAsync(audioBlock: blocks[0].0, timestampInMilliseconds: -1)
    }
    try expectInvalidArgument {
      try stream.classifyAsync(audioBlock: blocks[0].0, timestampInMilliseconds: .max)
    }
    try stream.classifyAsync(audioBlock: blocks[1].0, timestampInMilliseconds: blocks[1].1)
    try expectInvalidArgument {
      try stream.classifyAsync(audioBlock: blocks[0].0, timestampInMilliseconds: blocks[0].1)
    }
    try stream.close()
    #expect(box.waitFor(count: 1) == true)
  }

  @Test func `AudioStreamContext decodes results and forwards errors`() async throws {
    let deliveredResult: AudioClassifierResult =
      try await withCheckedThrowingContinuation { continuation in
        let context = AudioStreamContext { continuation.resume(with: $0) }
        var classification = MpClassificationResult(
          classifications: nil, classifications_count: 0, timestamp_ms: 321, has_timestamp_ms: true
        )
        withExtendedLifetime(context) {
          withUnsafeMutablePointer(to: &classification) { classificationPtr in
            var cResult = MpAudioClassifierResult(results: classificationPtr, results_count: 1)
            withUnsafePointer(to: &cResult) { cResultPtr in
              AudioStreamContext.receive(
                userData: context.userData, status: kMpOk, errorMessage: nil, result: cResultPtr)
            }
          }
        }
      }
    #expect(deliveredResult.classificationResults.count == 1)
    #expect(deliveredResult.timestampInMilliseconds == 321)

    let messageOutcome: Result<AudioClassifierResult, any Error> =
      await withCheckedContinuation { continuation in
        let context = AudioStreamContext { continuation.resume(returning: $0) }
        withExtendedLifetime(context) {
          "Stream error.".withCString { msg in
            AudioStreamContext.receive(
              userData: context.userData, status: kMpInvalidArgument, errorMessage: msg,
              result: nil)
          }
        }
      }
    let messageError = try #require(throws: TasksError.self) { _ = try messageOutcome.get() }
    #expect(messageError.code == .invalidArgument)
    #expect(messageError.message == "Stream error.")

    let nilMessageOutcome: Result<AudioClassifierResult, any Error> =
      await withCheckedContinuation { continuation in
        let context = AudioStreamContext { continuation.resume(returning: $0) }
        withExtendedLifetime(context) {
          AudioStreamContext.receive(
            userData: context.userData, status: kMpNotFound, errorMessage: nil, result: nil)
        }
      }
    let nilMessageError = try #require(throws: TasksError.self) { _ = try nilMessageOutcome.get() }
    #expect(nilMessageError.code == .notFound)

    let emptyMessageOutcome: Result<AudioClassifierResult, any Error> =
      await withCheckedContinuation { continuation in
        let context = AudioStreamContext { continuation.resume(returning: $0) }
        withExtendedLifetime(context) {
          "".withCString { emptyMsg in
            AudioStreamContext.receive(
              userData: context.userData, status: kMpInternal, errorMessage: emptyMsg, result: nil)
          }
        }
      }
    let emptyMessageError = try #require(throws: TasksError.self) {
      _ = try emptyMessageOutcome.get()
    }
    #expect(emptyMessageError.code == .internalError)

    let nullResultOutcome: Result<AudioClassifierResult, any Error> =
      await withCheckedContinuation { continuation in
        let context = AudioStreamContext { continuation.resume(returning: $0) }
        withExtendedLifetime(context) {
          AudioStreamContext.receive(
            userData: context.userData, status: kMpOk, errorMessage: nil, result: nil)
        }
      }
    let nullResultError = try #require(throws: TasksError.self) {
      _ = try nullResultOutcome.get()
    }
    #expect(nullResultError.code == .internalError)
  }

  @Test func `AudioStreamContext delivers a result only to the context of its user data`()
    async throws
  {
    let (otherStream, otherContinuation) = AsyncStream<Int>.makeStream()
    let other = AudioStreamContext { outcome in
      otherContinuation.yield((try? outcome.get().timestampInMilliseconds) ?? -1)
    }
    let deliveredTimestamp: Int = try await withCheckedThrowingContinuation { continuation in
      let context = AudioStreamContext { outcome in
        continuation.resume(with: outcome.map(\.timestampInMilliseconds))
      }
      var targetClassification = MpClassificationResult(
        classifications: nil, classifications_count: 0, timestamp_ms: 12, has_timestamp_ms: true
      )
      var sentinelClassification = MpClassificationResult(
        classifications: nil, classifications_count: 0, timestamp_ms: 99, has_timestamp_ms: true
      )
      withExtendedLifetime((other, context)) {
        withUnsafeMutablePointer(to: &targetClassification) { targetPtr in
          withUnsafeMutablePointer(to: &sentinelClassification) { sentinelPtr in
            var targetResult = MpAudioClassifierResult(results: targetPtr, results_count: 1)
            var sentinelResult = MpAudioClassifierResult(results: sentinelPtr, results_count: 1)
            withUnsafePointer(to: &targetResult) { targetResultPtr in
              AudioStreamContext.receive(
                userData: nil, status: kMpOk, errorMessage: nil, result: targetResultPtr)
              AudioStreamContext.cCallback(context.userData, kMpOk, nil, targetResultPtr)
            }
            withUnsafePointer(to: &sentinelResult) { sentinelResultPtr in
              AudioStreamContext.cCallback(other.userData, kMpOk, nil, sentinelResultPtr)
            }
          }
        }
      }
    }

    #expect(deliveredTimestamp == 12)
    var otherIterator = otherStream.makeAsyncIterator()
    #expect(await otherIterator.next() == 99)
  }
}
