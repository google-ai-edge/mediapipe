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

/// Describes the channel count and sample rate of an audio stream or buffer.
public struct AudioDataFormat: Sendable, Equatable, Hashable {
  /// The default mono channel count (`1`).
  public static let defaultChannelCount = 1

  /// Number of audio channels.
  public let channelCount: Int

  /// Sample rate in Hz.
  public let sampleRate: Double

  /// Creates an `AudioDataFormat` with the given channel count and sample rate.
  public init(channelCount: Int, sampleRate: Double) {
    self.channelCount = channelCount
    self.sampleRate = sampleRate
  }

  /// Creates a mono `AudioDataFormat` using `defaultChannelCount` and the given sample rate.
  public init(sampleRate: Double) {
    self.init(channelCount: Self.defaultChannelCount, sampleRate: sampleRate)
  }
}
