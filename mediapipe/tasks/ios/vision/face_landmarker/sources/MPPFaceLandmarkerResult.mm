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

#import <Foundation/Foundation.h>

#include <vector>

#import "mediapipe/tasks/ios/vision/face_landmarker/sources/MPPFaceLandmarkerResult.h"

@interface MPPTransformMatrix () {
  std::vector<float> _data;
}
@end

@implementation MPPTransformMatrix

- (instancetype)initWithData:(const float *)data rows:(NSInteger)rows columns:(NSInteger)columns {
  self = [super init];
  if (self) {
    _rows = rows;
    _columns = columns;
    _data = std::vector<float>(rows * columns);
    memcpy(_data.data(), data, rows * columns * sizeof(float));
  }
  return self;
}

- (float *)data {
  return _data.data();
}

- (float)valueAtRow:(NSUInteger)row column:(NSUInteger)column {
  if (row < 0 || row >= self.rows) {
    @throw [NSException exceptionWithName:NSRangeException
                                   reason:@"Row is outside of matrix range."
                                 userInfo:nil];
  }
  if (column < 0 || column >= self.columns) {
    @throw [NSException exceptionWithName:NSRangeException
                                   reason:@"Column is outside of matrix range."
                                 userInfo:nil];
  }
  return _data[row * _rows + column];
}

@end

@implementation MPPFaceLandmarkerResult

- (instancetype)initWithFaceLandmarks:(NSArray<NSArray<MPPNormalizedLandmark *> *> *)faceLandmarks
                      faceBlendshapes:(NSArray<MPPClassifications *> *)faceBlendshapes
         facialTransformationMatrixes:(NSArray<NSArray<NSNumber *> *> *)facialTransformationMatrixes
              timestampInMilliseconds:(NSInteger)timestampInMilliseconds {
  self = [super initWithTimestampInMilliseconds:timestampInMilliseconds];
  if (self) {
    _faceLandmarks = [faceLandmarks copy];
    _faceBlendshapes = [faceBlendshapes copy];
    _facialTransformationMatrixes = [facialTransformationMatrixes copy];
  }
  return self;
}

@end
