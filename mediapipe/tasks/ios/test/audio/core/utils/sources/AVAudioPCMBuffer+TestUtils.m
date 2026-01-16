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

#import "mediapipe/tasks/ios/test/audio/core/utils/sources/AVAudioPCMBuffer+TestUtils.h"

#define AudioFileWithInfo(fileInfo) \
  [[AVAudioFile alloc] initForReading:[NSURL fileURLWithPath:fileInfo.path] error:nil]

@implementation AVAudioPCMBuffer (TestUtils)

+ (nullable AVAudioPCMBuffer *)interleavedFloat32BufferFromAudioFileWithInfo:
    (MPPFileInfo *)fileInfo {
  AVAudioFile *audioFile = AudioFileWithInfo(fileInfo);

  AVAudioFormat *outputProcessingFormat =
      [[AVAudioFormat alloc] initWithCommonFormat:AVAudioPCMFormatFloat32
                                       sampleRate:audioFile.processingFormat.sampleRate
                                         channels:audioFile.processingFormat.channelCount
                                      interleaved:YES];
  return [AVAudioPCMBuffer bufferFromAudioFile:audioFile processingFormat:outputProcessingFormat];
}

+ (nullable AVAudioPCMBuffer *)bufferFromAudioFileWithInfo:(MPPFileInfo *)fileInfo
                                          processingFormat:(AVAudioFormat *)processingFormat {
  AVAudioFile *audioFile = AudioFileWithInfo(fileInfo);

  return [AVAudioPCMBuffer bufferFromAudioFile:audioFile processingFormat:processingFormat];
}

+ (nullable AVAudioPCMBuffer *)bufferFromAudioFile:(AVAudioFile *)audioFile
                                  processingFormat:(AVAudioFormat *)processingFormat {
  AVAudioPCMBuffer *buffer =
      [[AVAudioPCMBuffer alloc] initWithPCMFormat:audioFile.processingFormat
                                    frameCapacity:(AVAudioFrameCount)audioFile.length];

  [audioFile readIntoBuffer:buffer error:nil];

  return [buffer bufferWithProcessingFormat:processingFormat];
}

- (nullable MPPFloatBuffer *)floatBuffer {
  if (self.format.commonFormat != AVAudioPCMFormatFloat32) {
    return nil;
  }

  return [[MPPFloatBuffer alloc] initWithData:self.floatChannelData[0] length:self.frameLength];
}

- (AVAudioPCMBuffer *)bufferWithProcessingFormat:(AVAudioFormat *)processingFormat {
  if ([self.format isEqual:processingFormat]) {
    return self;
  }

  AVAudioConverter *audioConverter = [[AVAudioConverter alloc] initFromFormat:self.format
                                                                     toFormat:processingFormat];

  return [self bufferUsingAudioConverter:audioConverter];
}

- (AVAudioPCMBuffer *)bufferUsingAudioConverter:(AVAudioConverter *)audioConverter {
  if (!audioConverter) {
    return nil;
  }
  // Capacity of converted PCM buffer is calculated in order to maintain the same
  // latency as the input pcmBuffer.
  AVAudioFrameCount capacity = ceil(self.frameLength * audioConverter.outputFormat.sampleRate /
                                    audioConverter.inputFormat.sampleRate);
  AVAudioPCMBuffer *outPCMBuffer = [[AVAudioPCMBuffer alloc]
      initWithPCMFormat:audioConverter.outputFormat
          frameCapacity:capacity * (AVAudioFrameCount)audioConverter.outputFormat.channelCount];

  AVAudioConverterInputBlock inputBlock = ^AVAudioBuffer *_Nullable(
      AVAudioPacketCount inNumberOfPackets, AVAudioConverterInputStatus *_Nonnull outStatus) {
    *outStatus = AVAudioConverterInputStatus_HaveData;
    return self;
  };

  AVAudioConverterOutputStatus converterStatus = [audioConverter convertToBuffer:outPCMBuffer
                                                                           error:nil
                                                              withInputFromBlock:inputBlock];
  switch (converterStatus) {
    case AVAudioConverterOutputStatus_HaveData: {
      return outPCMBuffer;
    }
    case AVAudioConverterOutputStatus_InputRanDry:
    case AVAudioConverterOutputStatus_EndOfStream:
    case AVAudioConverterOutputStatus_Error: {
      // Conversion failed so returning a nil. Reason of the error isn't
      // important for tests.
      break;
    }
  }

  return nil;
}

@end
