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
import MediaPipeTasksAudioCore
import MediaPipeTasksCore
import Testing

struct AudioRecordTests {
  private static let monoFormat = AudioDataFormat(sampleRate: 16000.0, channelCount: 1)
  private static let stereoFormat = AudioDataFormat(sampleRate: 16000.0, channelCount: 2)

  @Test(arguments: [0, 3])
  func `init rejects unsupported channel count`(channelCount: Int) {
    #expect(throws: TasksError.self) {
      _ = try AudioRecord(
        audioDataFormat: AudioDataFormat(sampleRate: 16000.0, channelCount: channelCount),
        bufferLength: 6
      )
    }
  }

  @Test(arguments: [0, -2, 5])
  func `init rejects bufferLength that is not a positive multiple of channelCount`(
    bufferLength: Int
  ) {
    #expect(throws: TasksError.self) {
      _ = try AudioRecord(audioDataFormat: Self.stereoFormat, bufferLength: bufferLength)
    }
  }

  @Test
  func `loadBuffer and readBuffer round-trip samples and stop clears buffer`() throws {
    let record = try AudioRecord(audioDataFormat: Self.stereoFormat, bufferLength: 4)
    let samples = FloatBuffer(elements: [0.1, -0.1, 0.2, -0.2])
    try record.loadBuffer(samples, offset: 0, length: 4)
    #expect(try record.readBuffer(offset: 0, length: 4) == samples)

    try record.stop()
    #expect(try record.readBuffer(offset: 0, length: 4).elements == [0.0, 0.0, 0.0, 0.0])
  }

  @Test
  func `startRecording throws when microphone permission is not granted`() throws {
    let record = try AudioRecord(audioDataFormat: Self.stereoFormat, bufferLength: 4)
    guard AVAudioSession.sharedInstance().recordPermission != .granted else { return }

    #expect(throws: TasksError.self) {
      try record.startRecording()
    }
  }

  @Test
  func `read and AudioData load throw when recording is not active`() throws {
    let record = try AudioRecord(audioDataFormat: Self.stereoFormat, bufferLength: 4)
    let audioData = AudioData(format: Self.stereoFormat, sampleCount: 2)

    #expect(throws: TasksError.self) {
      _ = try record.read(offset: 0, length: 4)
    }
    #expect(throws: TasksError.self) {
      try audioData.load(audioRecord: record)
    }
  }

  @Test
  func `AudioData load rejects AudioRecord with mismatched format`() throws {
    let audioData = AudioData(format: Self.stereoFormat, sampleCount: 2)
    let monoRecord = try AudioRecord(audioDataFormat: Self.monoFormat, bufferLength: 4)

    #expect(throws: TasksError.self) {
      try audioData.load(audioRecord: monoRecord)
    }
  }

  @Test
  func `convert transforms PCM buffer in a single pass without duplicating frames`() throws {
    let pcmFormat = try #require(
      AVAudioFormat(
        commonFormat: .pcmFormatFloat32, sampleRate: 16000.0, channels: 2, interleaved: true
      )
    )
    let pcmBuffer = try #require(AVAudioPCMBuffer(pcmFormat: pcmFormat, frameCapacity: 2))
    pcmBuffer.frameLength = 2
    let channelPtr = try #require(pcmBuffer.floatChannelData)[0]
    channelPtr[0] = 0.25
    channelPtr[1] = 0.50
    channelPtr[2] = 0.75
    channelPtr[3] = 1.00

    let converter = try #require(AVAudioConverter(from: pcmFormat, to: pcmFormat))
    let converted = try AudioRecord.convert(pcmBuffer: pcmBuffer, using: converter)
    #expect(converted.frameLength == 2)

    let record = try AudioRecord(audioDataFormat: Self.stereoFormat, bufferLength: 4)
    try record.loadAudioPCMBuffer(converted)
    #expect(try record.readBuffer(offset: 0, length: 4).elements == [0.25, 0.50, 0.75, 1.00])
  }

  @Test
  func `loadAudioPCMBuffer rejects empty, channel-mismatched, and non-interleaved buffers`() throws
  {
    let record = try AudioRecord(audioDataFormat: Self.stereoFormat, bufferLength: 4)
    let monoRecord = try AudioRecord(audioDataFormat: Self.monoFormat, bufferLength: 4)
    let interleavedStereoFormat = try #require(
      AVAudioFormat(
        commonFormat: .pcmFormatFloat32, sampleRate: 16000.0, channels: 2, interleaved: true
      )
    )

    let emptyBuffer = try #require(
      AVAudioPCMBuffer(pcmFormat: interleavedStereoFormat, frameCapacity: 2)
    )
    #expect(throws: TasksError.self) {
      try record.loadAudioPCMBuffer(emptyBuffer)
    }

    let stereoBuffer = try #require(
      AVAudioPCMBuffer(pcmFormat: interleavedStereoFormat, frameCapacity: 2)
    )
    stereoBuffer.frameLength = 2
    #expect(throws: TasksError.self) {
      try monoRecord.loadAudioPCMBuffer(stereoBuffer)
    }

    let nonInterleavedFormat = try #require(
      AVAudioFormat(
        commonFormat: .pcmFormatFloat32, sampleRate: 16000.0, channels: 2, interleaved: false
      )
    )
    let nonInterleavedBuffer = try #require(
      AVAudioPCMBuffer(pcmFormat: nonInterleavedFormat, frameCapacity: 2)
    )
    nonInterleavedBuffer.frameLength = 2
    #expect(throws: TasksError.self) {
      try record.loadAudioPCMBuffer(nonInterleavedBuffer)
    }
  }
}
