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

NS_ASSUME_NONNULL_BEGIN

@interface MPPFileInfo : NSObject

/** The name of the file. */
@property(nonatomic, readonly) NSString *name;

/** The type of the file. */
@property(nonatomic, readonly) NSString *type;

/** The path to file in the app bundle. */
@property(nonatomic, readonly, nullable) NSString *path;

/**
 * Initializes an `MPPFileInfo` using the given name and type of file.
 *
 * @param name The name of the file.
 * @param type The type of the file.
 *
 * @return The `MPPFileInfo` with the given name and type of file.
 */
- (instancetype)initWithName:(NSString *)name type:(NSString *)type;

@end

NS_ASSUME_NONNULL_END
