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

#include "mediapipe/tasks/cc/components/containers/proto/embeddings.pb.h"
#import "mediapipe/tasks/ios/components/containers/sources/MPPEmbeddingResult.h"

NS_ASSUME_NONNULL_BEGIN

@interface MPPEmbeddingResult (Helpers)

+ (MPPEmbeddingResult *)embeddingResultWithProto:
    (const ::mediapipe::tasks::components::containers::proto::EmbeddingResult &)
        embeddingResultProto;

@end

NS_ASSUME_NONNULL_END
