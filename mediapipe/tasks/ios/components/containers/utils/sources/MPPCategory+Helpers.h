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

#include "mediapipe/framework/formats/classification.pb.h"
#import "mediapipe/tasks/ios/components/containers/sources/MPPCategory.h"

NS_ASSUME_NONNULL_BEGIN

@interface MPPCategory (Helpers)

/**
 * Creates an `MPPCategory` with the given MediaPipe `Classification` proto.
 *
 * @param classificationProto A MediaPipe `Classification` proto.
 * @return  An `MPPCategory` object that with the given MediaPipe `Classification` proto.
 */
+ (MPPCategory *)categoryWithProto:(const ::mediapipe::Classification &)classificationProto;

/**
 * Creates an `MPPCategory` with the given MediaPipe `Classification` proto and the given category
 * index. The resulting `MPPCategory` is created with the given category index instead of the
 * category index specified in the `Classification` proto. This method is useful for tasks like
 * gesture recognizer which always returns a default index for the recognized gestures.
 *
 * @param classificationProto A MediaPipe `Classification` proto.
 * @param index The index to be used for creating the `MPPCategory` instead of the category index
 * specified in the `Classification` proto.
 *
 * @return  An `MPPGestureRecognizerResult` object that contains the hand gesture recognition
 * results.
 */
+ (MPPCategory *)categoryWithProto:(const ::mediapipe::Classification &)classificationProto
                             index:(NSInteger)index;

@end

NS_ASSUME_NONNULL_END
