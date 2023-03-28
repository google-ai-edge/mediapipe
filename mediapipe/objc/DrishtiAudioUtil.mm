// Copyright 2023 The MediaPipe Authors.
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

#import "mediapipe/objc/MediaPipeAudioUtil.h"

#include <limits>

namespace {
// `float` is 32-bit.
static_assert(std::numeric_limits<float>::is_iec559);
using float32_t = float;

template <typename SampleDataType>
float GetSample(const void* data, int index);

template <>
float GetSample<float32_t>(const void* data, int index) {
  return reinterpret_cast<const float32_t*>(data)[index];
};

template <>
float GetSample<SInt16>(const void* data, int index) {
  // Convert to the [-1, 1] range.
  return static_cast<float>(reinterpret_cast<const SInt16*>(data)[index]) /
         static_cast<float>(std::numeric_limits<SInt16>::max());
};

template <typename SampleDataType>
std::unique_ptr<mediapipe::Matrix> MakeMatrix(const AudioBuffer* buffers, int channels,
                                            CMItemCount frames, bool interleaved) {
  // Create the matrix and fill it accordingly. Its dimensions are `channels x frames`.
  auto matrix = std::make_unique<mediapipe::Matrix>(channels, frames);
  // Split the case of interleaved and non-interleaved samples (see
  // https://developer.apple.com/documentation/coremedia/1489723-cmsamplebuffercreate#discussion)
  // - however, the resulting operations coincide when `channels == 1`.
  if (interleaved) {
    // A single buffer contains interleaved samples for all the channels {L, R, L, R, L, R, ...}.
    const void* samples = buffers[0].mData;
    for (int channel = 0; channel < channels; ++channel) {
      for (int frame = 0; frame < frames; ++frame) {
        (*matrix)(channel, frame) = GetSample<SampleDataType>(samples, channels * frame + channel);
      }
    }
  } else {
    // Non-interleaved audio: each channel's samples are stored in a separate buffer:
    // {{L, L, L, L, ...}, {R, R, R, R, ...}}.
    for (int channel = 0; channel < channels; ++channel) {
      const void* samples = buffers[channel].mData;
      for (int frame = 0; frame < frames; ++frame) {
        (*matrix)(channel, frame) = GetSample<SampleDataType>(samples, frame);
      }
    }
  }
  return matrix;
}
}  // namespace

absl::StatusOr<std::unique_ptr<mediapipe::Matrix>> MediaPipeConvertAudioBufferListToAudioMatrix(
    const AudioBufferList* audioBufferList, const AudioStreamBasicDescription* streamHeader,
    CMItemCount numFrames) {
  // Sort out the channel count and whether the data is interleaved.
  // Note that we treat "interleaved" mono audio as non-interleaved.
  CMItemCount numChannels = 1;
  bool isAudioInterleaved = false;
  if (streamHeader->mChannelsPerFrame > 1) {
    if (streamHeader->mFormatFlags & kAudioFormatFlagIsNonInterleaved) {
      numChannels = audioBufferList->mNumberBuffers;
      isAudioInterleaved = false;
    } else {
      numChannels = audioBufferList->mBuffers[0].mNumberChannels;
      isAudioInterleaved = true;
    }
    if (numChannels <= 1) {
      return absl::InternalError("AudioStreamBasicDescription indicates more than 1 channel, "
                                 "but the buffer data declares an incompatible number of channels");
    }
  }

  if ((streamHeader->mFormatFlags & kAudioFormatFlagIsFloat) &&
      streamHeader->mBitsPerChannel == 32) {
    return MakeMatrix<float32_t>(audioBufferList->mBuffers, numChannels, numFrames,
                                 isAudioInterleaved);
  }
  if ((streamHeader->mFormatFlags & kAudioFormatFlagIsSignedInteger) &&
      streamHeader->mBitsPerChannel == 16) {
    return MakeMatrix<SInt16>(audioBufferList->mBuffers, numChannels, numFrames,
                              isAudioInterleaved);
  }
  return absl::InternalError("Incompatible audio sample storage format");
}
