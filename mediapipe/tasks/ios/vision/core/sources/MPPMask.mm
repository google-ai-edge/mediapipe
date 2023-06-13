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

namespace {
template <typename T>
T *allocateDataPtr(std::unique_ptr<T[]> &data, size_t length) {
  data = std::unique_ptr<T[]>(new T[length]);
  return data.get();
}

template <typename T>
void copyData(const T *destination, const T *source, size_t length) {
  memcpy((void *)destination, source, length * sizeof(T));
}
}  // namespace

@interface MPPMask () {
  const UInt8 *_uint8Data;
  const float *_float32Data;
  std::unique_ptr<UInt8[]> _allocatedUInt8Data;
  std::unique_ptr<float[]> _allocatedFloat32Data;
}
@end

@implementation MPPMask

- (nullable instancetype)initWithWidth:(NSInteger)width
                                height:(NSInteger)height
                              dataType:(MPPMaskDataType)dataType
                                 error:(NSError **)error {
  if (dataType < MPPMaskDataTypeUInt8 || dataType > MPPMaskDataTypeFloat32) {
    [MPPCommonUtils createCustomError:error
                             withCode:MPPTasksErrorCodeInvalidArgumentError
                          description:@"Invalid value for data type."];
    return nil;
  }

  self = [super init];
  if (self) {
    _width = width;
    _height = height;
    _dataType = dataType;
  }
  return self;
}

- (nullable instancetype)initWithUInt8Data:(const UInt8 *)uint8Data
                                     width:(NSInteger)width
                                    height:(NSInteger)height {
  self = [self initWithWidth:width height:height dataType:MPPMaskDataTypeUInt8 error:nil];
  if (self) {
    _uint8Data = uint8Data;
  }
  return self;
}

- (nullable instancetype)initWithFloat32Data:(const float *)float32Data
                                       width:(NSInteger)width
                                      height:(NSInteger)height {
  self = [self initWithWidth:width height:height dataType:MPPMaskDataTypeFloat32 error:nil];
  if (self) {
    _float32Data = float32Data;
  }
  return self;
}

- (instancetype)initWithUInt8DataToCopy:(const UInt8 *)uint8DataToCopy
                                  width:(NSInteger)width
                                 height:(NSInteger)height {
  self = [self initWithWidth:width height:height dataType:MPPMaskDataTypeUInt8 error:nil];
  if (self) {
    _uint8Data = allocateDataPtr(_allocatedUInt8Data, _width * _height);
    copyData(_uint8Data, uint8DataToCopy, _width * _height);
  }
  return self;
}

- (instancetype)initWithFloat32DataToCopy:(const float *)float32DataToCopy
                                    width:(NSInteger)width
                                   height:(NSInteger)height {
  self = [self initWithWidth:width height:height dataType:MPPMaskDataTypeFloat32 error:nil];
  if (self) {
    _float32Data = allocateDataPtr(_allocatedFloat32Data, _width * _height);
    copyData(_float32Data, float32DataToCopy, _width * _height);
  }
  return self;
}

- (const UInt8 *)uint8Data {
  switch (_dataType) {
    case MPPMaskDataTypeUInt8: {
      return _uint8Data;
    }
    case MPPMaskDataTypeFloat32: {
      if (_allocatedUInt8Data) {
        return _allocatedUInt8Data.get();
      }
      UInt8 *data = allocateDataPtr(_allocatedUInt8Data, _width * _height);
      for (int i = 0; i < _width * _height; i++) {
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
      if (_allocatedFloat32Data) {
        return _allocatedFloat32Data.get();
      }
      float *data = allocateDataPtr(_allocatedFloat32Data, _width * _height);
      for (int i = 0; i < _width * _height; i++) {
        data[i] = _uint8Data[i] / 255;
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
      return [[MPPMask alloc] initWithUInt8DataToCopy:self.uint8Data
                                                width:self.width
                                               height:self.height];
    case MPPMaskDataTypeFloat32:
      return [[MPPMask alloc] initWithFloat32DataToCopy:self.float32Data
                                                  width:self.width
                                                 height:self.height];
  }
}

@end
