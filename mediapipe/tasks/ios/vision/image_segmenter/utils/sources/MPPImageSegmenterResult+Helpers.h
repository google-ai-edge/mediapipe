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

#import "mediapipe/tasks/ios/vision/image_segmenter/sources/MPPImageSegmenterResult.h"

struct MpImageSegmenterResult;

NS_ASSUME_NONNULL_BEGIN

@interface MPPImageSegmenterResult (Helpers)

/**
 * Creates an `MPPImageSegmenterResult` from a C `MpImageSegmenterResult` struct.
 *
 * If `shouldCopyMaskPacketData` is set to `YES`, the confidence and category masks of the newly
 * created `MPPImageSegmenterResult` hold references to deep copied pixel data of the output masks.
 */
+ (nullable MPPImageSegmenterResult *)
    imageSegmenterResultWithCResult:(const struct MpImageSegmenterResult &)cResult
            timestampInMilliseconds:(NSInteger)timestampInMilliseconds
           shouldCopyMaskPacketData:(BOOL)shouldCopyMaskPacketData;

@end

NS_ASSUME_NONNULL_END
