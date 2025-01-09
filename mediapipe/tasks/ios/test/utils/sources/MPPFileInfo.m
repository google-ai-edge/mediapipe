// Copyright 2023 The TensorFlow Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#import "mediapipe/tasks/ios/test/utils/sources/MPPFileInfo.h"

@implementation MPPFileInfo

- (instancetype)initWithName:(NSString *)name type:(NSString *)type {
  self = [super init];
  if (self) {
    _name = name;
    _type = type;
  }

  return self;
}

- (NSString *)path {
  return [[NSBundle bundleForClass:self.class] pathForResource:self.name ofType:self.type];
}

@end
