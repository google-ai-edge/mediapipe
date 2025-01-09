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

#import "mediapipe/tasks/ios/vision/core/sources/MPPMask.h"
#import "mediapipe/tasks/ios/common/sources/MPPCommon.h"
#import "mediapipe/tasks/ios/common/utils/sources/MPPCommonUtils.h"

@interface MPPMask () {
  const UInt8 *_uint8Data;
  const float *_float32Data;
  std::unique_ptr<UInt8[]> _uint8DataPtr;
  std::unique_ptr<float[]> _float32DataPtr;
}
@end

@implementation MPPMask

- (nullable instancetype)initWithUInt8Data:(const UInt8 *)uint8Data
                                     width:(NSInteger)width
                                    height:(NSInteger)height
                                shouldCopy:(BOOL)shouldCopy {
  self = [super init];
  if (self) {
    _width = width;
    _height = height;
    _dataType = MPPMaskDataTypeUInt8;

    if (shouldCopy) {
      size_t length = _width * _height;
      _uint8DataPtr = std::unique_ptr<UInt8[]>(new UInt8[length]);
      _uint8Data = _uint8DataPtr.get();
      memcpy((UInt8 *)_uint8Data, uint8Data, length * sizeof(UInt8));
    } else {
      _uint8Data = uint8Data;
    }
  }
  return self;
}

- (nullable instancetype)initWithFloat32Data:(const float *)float32Data
                                       width:(NSInteger)width
                                      height:(NSInteger)height
                                  shouldCopy:(BOOL)shouldCopy {
  self = [super init];
  if (self) {
    _width = width;
    _height = height;
    _dataType = MPPMaskDataTypeFloat32;

    if (shouldCopy) {
      size_t length = _width * _height;
      _float32DataPtr = std::unique_ptr<float[]>(new float[length]);
      _float32Data = _float32DataPtr.get();
      memcpy((float *)_float32Data, float32Data, length * sizeof(float));
    } else {
      _float32Data = float32Data;
    }
  }
  return self;
}

- (const UInt8 *)uint8Data {
  switch (_dataType) {
    case MPPMaskDataTypeUInt8: {
      return _uint8Data;
    }
    case MPPMaskDataTypeFloat32: {
      if (_uint8DataPtr) {
        return _uint8DataPtr.get();
      }

      size_t length = _width * _height;
      _uint8DataPtr = std::unique_ptr<UInt8[]>(new UInt8[length]);
      UInt8 *data = _uint8DataPtr.get();
      for (int i = 0; i < length; i++) {
        data[i] = _float32Data[i] * 255;
      }
      return data;
    }
    default:
      return NULL;
  }
}

- (const float *)float32Data {
  switch (_dataType) {
    case MPPMaskDataTypeUInt8: {
      if (_float32DataPtr) {
        return _float32DataPtr.get();
      }

      size_t length = _width * _height;
      _float32DataPtr = std::unique_ptr<float[]>(new float[length]);
      float *data = _float32DataPtr.get();
      for (int i = 0; i < length; i++) {
        data[i] = (float)_uint8Data[i] / 255;
      }
      return data;
    }
    case MPPMaskDataTypeFloat32: {
      return _float32Data;
    }
    default:
      return NULL;
  }
}

- (id)copyWithZone:(NSZone *)zone {
  switch (_dataType) {
    case MPPMaskDataTypeUInt8:
      return [[MPPMask alloc] initWithUInt8Data:self.uint8Data
                                          width:self.width
                                         height:self.height
                                     shouldCopy:YES];
    case MPPMaskDataTypeFloat32:
      return [[MPPMask alloc] initWithFloat32Data:self.float32Data
                                            width:self.width
                                           height:self.height
                                       shouldCopy:YES];
  }
}

@end
