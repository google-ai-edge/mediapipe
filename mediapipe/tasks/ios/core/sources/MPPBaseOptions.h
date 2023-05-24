// Copyright 2022 The MediaPipe Authors.
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

NS_ASSUME_NONNULL_BEGIN

/**
 * Holds the base options that is used for creation of any type of task. It has fields with
 * important information acceleration configuration, TFLite model source etc.
 */
NS_SWIFT_NAME(BaseOptions)
@interface MPPBaseOptions : NSObject <NSCopying>

/** The path to the model asset to open and mmap in memory. */
@property(nonatomic, copy) NSString *modelAssetPath;

@end

NS_ASSUME_NONNULL_END
