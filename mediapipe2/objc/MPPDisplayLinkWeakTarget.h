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

#import <AVFoundation/AVFoundation.h>
#import <Foundation/Foundation.h>

/// A generic target/callback holder. Useful for indirectly using DisplayLink and allowing the
/// complete deletion of displaylink reference holders.
@interface MPPDisplayLinkWeakTarget : NSObject

- (instancetype)initWithTarget:(id)target selector:(SEL)sel;

- (void)displayLinkCallback:(CADisplayLink *)sender;

@end
