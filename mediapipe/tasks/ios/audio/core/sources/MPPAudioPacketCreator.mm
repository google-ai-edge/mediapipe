// Copyright 2024 The MediaPipe Authors.
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

#import "mediapipe/tasks/ios/audio/core/sources/MPPAudioPacketCreator.h"

#import "mediapipe/tasks/ios/common/sources/MPPCommon.h"
#import "mediapipe/tasks/ios/common/utils/sources/MPPCommonUtils.h"

#include "mediapipe/framework/formats/matrix.h"
#include "mediapipe/framework/timestamp.h"

static const NSUInteger kMicrosecondsPerMillisecond = 1000;

namespace {
using ::mediapipe::Adopt;
using ::mediapipe::Matrix;
using ::mediapipe::Packet;
using ::mediapipe::Timestamp;
}  // namespace

@implementation MPPAudioPacketCreator

+ (Packet)createPacketWithAudioData:(MPPAudioData *)audioData error:(NSError **)error {
  std::unique_ptr<Matrix> matrix = [MPPAudioPacketCreator createMatrixWithAudioData:audioData
                                                                              error:error];
  if (!matrix) {
    return Packet();
  }

  return mediapipe::Adopt(matrix.release());
}

+ (Packet)createPacketWithAudioData:(MPPAudioData *)audioData
            timestampInMilliseconds:(NSInteger)timestampInMilliseconds
                              error:(NSError **)error {
  std::unique_ptr<Matrix> matrix = [MPPAudioPacketCreator createMatrixWithAudioData:audioData
                                                                              error:error];
  if (!matrix) {
    return Packet();
  }
  return Adopt(matrix.release())
      .At(Timestamp(int64_t(timestampInMilliseconds * kMicrosecondsPerMillisecond)));
}

+ (std::unique_ptr<Matrix>)createMatrixWithAudioData:(MPPAudioData *)audioData
                                               error:(NSError **)error {
  MPPFloatBuffer *audioDataBuffer = audioData.buffer;
  if (!audioDataBuffer.data) {
    [MPPCommonUtils createCustomError:error
                             withCode:MPPTasksErrorCodeInvalidArgumentError
                          description:@"Audio data buffer cannot be nil."];
    return nullptr;
  }

  NSUInteger rowCount = audioData.format.channelCount;
  NSUInteger colCount = audioData.bufferLength;

  std::unique_ptr<mediapipe::Matrix> matrix(new mediapipe::Matrix(rowCount, colCount));
  // iOS is always little-endian. Hence, data can be copied directly.
  memcpy(matrix->data(), audioDataBuffer.data, rowCount * colCount * sizeof(float));

  return matrix;
}

@end
