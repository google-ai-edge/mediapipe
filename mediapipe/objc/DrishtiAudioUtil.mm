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

#include "third_party/eigen3/Eigen/Core"

namespace {
using Eigen::Index;
using Eigen::Map;
using Eigen::VectorXf;
using VectorXi16 = Eigen::Vector<SInt16, Eigen::Dynamic>;

// `float` is 32-bit.
static_assert(std::numeric_limits<float>::is_iec559);

// Reads an array of `size` elements of type `float` at `samples` and writes it into `target`,
// which is an Eigen expression compatible with a `VectorXf` of size `size`.
template <typename OutputVector>
void CopyBufferToFloatVector(const float* samples, CMItemCount size, OutputVector target) {
  target = Map<const VectorXf>(samples, static_cast<Index>(size));
};

// Reads an array of `size` elements of type `SInt16` at `samples` and writes it into `target`,
// which is an Eigen expression compatible with a `VectorXf` of size `size`.
template <typename OutputVector>
void CopyBufferToFloatVector(const SInt16* samples, CMItemCount size, OutputVector target) {
  // Convert to the [-1, 1] range.
  constexpr float kRangeMax = static_cast<float>(std::numeric_limits<SInt16>::max());
  target = Map<const VectorXi16>(samples, static_cast<Index>(size)).cast<float>() / kRangeMax;
};

template <typename SampleDataType>
std::unique_ptr<mediapipe::Matrix> MakeMatrix(const AudioBuffer* buffers, CMItemCount channels,
                                            CMItemCount frames, bool interleaved) {
  // Create the matrix and fill it accordingly. Its dimensions are `channels x frames`.
  auto matrix = std::make_unique<mediapipe::Matrix>(channels, frames);
  // Split the cases of interleaved and non-interleaved samples (see
  // https://developer.apple.com/documentation/coremedia/1489723-cmsamplebuffercreate#discussion)
  // - however, the resulting operations coincide when `channels == 1`.
  if (interleaved) {
    // A single buffer contains interleaved samples for all the channels {L, R, L, R, L, R, ...}.
    // This corresponds to Eigen's default column-major matrix layout.
    const SampleDataType* samples = reinterpret_cast<const SampleDataType*>(buffers[0].mData);
    CopyBufferToFloatVector(/*samples=*/samples, /*size=*/channels * frames,
                            /*target=*/matrix->reshaped());
  } else {
    // Non-interleaved audio: each channel's samples are stored in a separate buffer:
    // {{L, L, L, L, ...}, {R, R, R, R, ...}}.
    for (CMItemCount channel = 0; channel < channels; ++channel) {
      const SampleDataType* samples =
          reinterpret_cast<const SampleDataType*>(buffers[channel].mData);
      CopyBufferToFloatVector(/*samples=*/samples, /*size=*/frames,
                              /*target=*/matrix->row(static_cast<Index>(channel)));
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
    return MakeMatrix<float>(audioBufferList->mBuffers, numChannels, numFrames, isAudioInterleaved);
  }
  if ((streamHeader->mFormatFlags & kAudioFormatFlagIsSignedInteger) &&
      streamHeader->mBitsPerChannel == 16) {
    return MakeMatrix<SInt16>(audioBufferList->mBuffers, numChannels, numFrames,
                              isAudioInterleaved);
  }
  return absl::InternalError("Unsupported audio sample storage format");
}
