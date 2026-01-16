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

#include "mediapipe/tasks/cc/core/proto/base_options.pb.h"
#import "mediapipe/tasks/ios/core/sources/MPPBaseOptions.h"

NS_ASSUME_NONNULL_BEGIN

@interface MPPBaseOptions (Helpers)

- (void)copyToProto:(mediapipe::tasks::core::proto::BaseOptions *)baseOptionsProto;
- (void)copyToProto:(mediapipe::tasks::core::proto::BaseOptions *)baseOptionsProto
    withUseStreamMode:(BOOL)useStreamMode;

@end

NS_ASSUME_NONNULL_END
