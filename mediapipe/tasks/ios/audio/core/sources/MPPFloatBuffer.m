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

#import "mediapipe/tasks/ios/audio/core/sources/MPPFloatBuffer.h"

@implementation MPPFloatBuffer

- (instancetype)initWithLength:(NSUInteger)length {
  return [self initWithData:NULL length:length];
}

- (instancetype)initWithData:(nullable const float *)data length:(NSUInteger)length {
  self = [super init];
  if (self) {
    _length = length;
    _data = calloc(length, sizeof(float));
    if (data) {
      memcpy(_data, data, sizeof(float) * length);
    }
  }
  return self;
}

- (id)copyWithZone:(NSZone *)zone {
  return [[MPPFloatBuffer alloc] initWithData:_data length:_length];
}

- (void)clear {
  memset(_data, 0, sizeof(float) * _length);
}

- (void)dealloc {
  free(_data);
}

@end
