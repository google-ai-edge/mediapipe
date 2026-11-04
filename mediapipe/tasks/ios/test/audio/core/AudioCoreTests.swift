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
import MediaPipeTasksAudioCore
import MediaPipeTasksCore
import Testing
import third_party_mediapipe_tasks_c_audio_core_common

struct AudioCoreTests {
  @Test
  func `AudioDataFormat and AudioRunningMode bridge properties and C enums`() {
    let monoFormat = AudioDataFormat(sampleRate: 16000.0)
    #expect(monoFormat.channelCount == 1)
    #expect(monoFormat.sampleRate == 16000.0)

    let stereoFormat = AudioDataFormat(sampleRate: 44100.0, channelCount: 2)
    #expect(stereoFormat.channelCount == 2)
    #expect(monoFormat != stereoFormat)

    #expect(AudioRunningMode.audioClips.cRunningMode == kMpAudioRunningModeAudioClips)
    #expect(AudioRunningMode.audioStream.cRunningMode == kMpAudioRunningModeAudioStream)
    #expect(AudioRunningMode.audioClips.displayName == "Audio Clips")
    #expect(AudioRunningMode.audioStream.displayName == "Audio Stream")
  }

  @Test
  func `FloatBuffer and FloatRingBuffer wrap circular writes, slices, and clear`() throws {
    let initial = FloatBuffer(elements: [1.0, 2.0, 3.0, 4.0])
    let cloned = initial.copy()
    #expect(cloned == initial)
    initial.clear()
    #expect(initial.elements == [0.0, 0.0, 0.0, 0.0])
    #expect(cloned.elements == [1.0, 2.0, 3.0, 4.0])

    let ring = FloatRingBuffer(length: 5)
    #expect(ring.floatBuffer.elements == [0.0, 0.0, 0.0, 0.0, 0.0])

    let firstChunk = FloatBuffer(elements: [1.0, 2.0, 3.0])
    try ring.load(floatBuffer: firstChunk, offset: 0, length: 3)
    #expect(ring.floatBuffer.elements == [0.0, 0.0, 1.0, 2.0, 3.0])

    let secondChunk = FloatBuffer(elements: [4.0, 5.0, 6.0, 7.0])
    try ring.load(floatBuffer: secondChunk, offset: 0, length: 4)
    #expect(ring.floatBuffer.elements == [3.0, 4.0, 5.0, 6.0, 7.0])
    #expect(try ring.floatBuffer(offset: 1, length: 3).elements == [4.0, 5.0, 6.0])

    let oversizedChunk = FloatBuffer(elements: [10.0, 20.0, 30.0, 40.0, 50.0, 60.0])
    try ring.load(floatBuffer: oversizedChunk, offset: 0, length: 6)
    #expect(ring.floatBuffer.elements == [20.0, 30.0, 40.0, 50.0, 60.0])

    #expect(throws: TasksError.self) {
      try ring.load(floatBuffer: firstChunk, offset: 2, length: 2)
    }
    #expect(throws: TasksError.self) {
      _ = try ring.floatBuffer(offset: 3, length: 3)
    }

    ring.clear()
    #expect(ring.floatBuffer.elements == [0.0, 0.0, 0.0, 0.0, 0.0])
  }

  @Test
  func `AudioData loads FloatBuffer and bridges to MpAudioData`() throws {
    let format = AudioDataFormat(sampleRate: 16000.0, channelCount: 2)
    let audioData = AudioData(format: format, sampleCount: 3)
    #expect(audioData.bufferLength == 3)
    #expect(audioData.buffer.length == 6)

    let input = FloatBuffer(elements: [0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    try audioData.load(buffer: input, offset: 0, length: 6)
    #expect(audioData.buffer == input)

    try audioData.withCAudioData { cAudioData in
      #expect(cAudioData.num_channels == 2)
      #expect(cAudioData.sample_rate == 16000.0)
      #expect(cAudioData.audio_data_size == 6)
      let ptr = try #require(cAudioData.audio_data)
      let bridged = Array(UnsafeBufferPointer(start: ptr, count: Int(cAudioData.audio_data_size)))
      #expect(bridged == [0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    }
  }
}
