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

#import "mediapipe/tasks/ios/components/containers/sources/MPPClassificationResult.h"

struct MpClassifications;
struct MpClassificationResult;

NS_ASSUME_NONNULL_BEGIN

@interface MPPClassifications (Helpers)

/** Creates a new `MPPClassifications` object from the C API `MpClassifications` struct. */
+ (MPPClassifications *)classificationsWithCClassifications:
    (const struct MpClassifications &)cClassifications;

@end

@interface MPPClassificationResult (Helpers)

/** Creates a new `MPPClassificationResult` object from the C API `MpClassificationResult` struct.
 */
+ (MPPClassificationResult *)classificationResultWithCResult:
    (const struct MpClassificationResult &)cResult;

@end

NS_ASSUME_NONNULL_END
