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

#import "mediapipe/tasks/ios/vision/image_embedder/utils/sources/MPPImageEmbedderOptions+Helpers.h"

#import "mediapipe/tasks/ios/core/utils/sources/MPPBaseOptions+Helpers.h"

#include "mediapipe/tasks/c/vision/image_embedder/image_embedder.h"

@implementation MPPImageEmbedderOptions (Helpers)

- (void)copyToCOptions:(struct ImageEmbedderOptions *)cOptions {
  [self.baseOptions copyToCOptions:&cOptions->base_options];
  switch (self.runningMode) {
    case MPPRunningModeImage:
      cOptions->running_mode = MP_RUNNING_MODE_IMAGE;
      break;
    case MPPRunningModeVideo:
      cOptions->running_mode = MP_RUNNING_MODE_VIDEO;
      break;
    case MPPRunningModeLiveStream:
      cOptions->running_mode = MP_RUNNING_MODE_LIVE_STREAM;
      break;
  }
  cOptions->embedder_options.l2_normalize = self.l2Normalize ? true : false;
  cOptions->embedder_options.quantize = self.quantize ? true : false;
  cOptions->result_callback = nullptr;
}

+ (void)freeCOptions:(struct ImageEmbedderOptions *)cOptions {
  if (cOptions) {
    [MPPBaseOptions freeCOptions:&cOptions->base_options];
  }
}

@end
