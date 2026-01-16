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

#include "mediapipe/tasks/cc/components/processors/proto/classifier_options.pb.h"
#import "mediapipe/tasks/ios/components/processors/sources/MPPClassifierOptions.h"

NS_ASSUME_NONNULL_BEGIN

@interface MPPClassifierOptions (Helpers)

- (void)copyToProto:
    (mediapipe::tasks::components::processors::proto::ClassifierOptions *)classifierOptionsProto;

@end

NS_ASSUME_NONNULL_END
