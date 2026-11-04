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

#import "mediapipe/tasks/ios/text/text_embedder/utils/sources/MPPTextEmbedderOptions+Helpers.h"

#import "mediapipe/tasks/ios/core/utils/sources/MPPBaseOptions+Helpers.h"

@implementation MPPTextEmbedderOptions (Helpers)

- (void)copyToCOptions:(MpTextEmbedderOptions *)cOptions {
  [self.baseOptions copyToCOptions:&(cOptions->base_options)];
  cOptions->embedder_options.l2_normalize = self.l2Normalize ? true : false;
  cOptions->embedder_options.quantize = self.quantize ? true : false;
}

+ (void)freeCOptions:(MpTextEmbedderOptions *)cOptions {
  if (cOptions) {
    [MPPBaseOptions freeCOptions:&(cOptions->base_options)];
  }
}

@end
