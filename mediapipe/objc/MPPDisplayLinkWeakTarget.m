// Copyright 2019 The MediaPipe Authors.
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

#import "mediapipe/objc/MPPDisplayLinkWeakTarget.h"

@implementation MPPDisplayLinkWeakTarget {
  __weak id _target;
  SEL _selector;
}

#pragma mark - Init

- (instancetype)initWithTarget:(id)target selector:(SEL)sel {
  self = [super init];
  if (self) {
    _target = target;
    _selector = sel;
  }
  return self;
}

#pragma mark - Public

- (void)displayLinkCallback:(CADisplayLink *)sender {
  void (*display)(id, SEL, CADisplayLink *) = (void *)[_target methodForSelector:_selector];
  display(_target, _selector, sender);
}

@end
