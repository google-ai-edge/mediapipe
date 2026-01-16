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
#import "mediapipe/tasks/ios/vision/gesture_recognizer/sources/MPPGestureRecognizerResult.h"

NS_ASSUME_NONNULL_BEGIN
@interface MPPGestureRecognizerResult (ProtobufHelpers)

+ (MPPGestureRecognizerResult *)
    gestureRecognizerResultsFromProtobufFileWithName:(NSString *)fileName
                                        gestureLabel:(NSString *)gestureLabel
                               shouldRemoveZPosition:(BOOL)removeZPosition;

@end

NS_ASSUME_NONNULL_END
