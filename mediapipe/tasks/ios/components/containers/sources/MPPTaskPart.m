// Copyright 2026 The MediaPipe Authors. All Rights Reserved.
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

#import "mediapipe/tasks/ios/components/containers/sources/MPPTaskPart.h"

@implementation MPPTaskPart

@end

@implementation MPPTextPart

- (instancetype)initWithText:(NSString *)text {
  self = [super init];
  if (self) {
    _text = [text copy];
  }
  return self;
}

@end

@implementation MPPImagePart

- (instancetype)initWithImageBytes:(NSData *)imageBytes {
  return [self initWithImageBytes:imageBytes filePath:nil];
}

- (instancetype)initWithImageBytes:(NSData *)imageBytes filePath:(nullable NSString *)filePath {
  self = [super init];
  if (self) {
    _imageBytes = [imageBytes copy];
    _filePath = [filePath copy];
  }
  return self;
}

@end

@implementation MPPAudioPart

- (instancetype)initWithAudioData:(MPPFloatBuffer *)audioData {
  return [self initWithAudioData:audioData filePath:nil];
}

- (instancetype)initWithAudioData:(MPPFloatBuffer *)audioData
                         filePath:(nullable NSString *)filePath {
  self = [super init];
  if (self) {
    _audioData = audioData;
    _filePath = [filePath copy];
  }
  return self;
}

@end
