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

#include "mediapipe/tasks/cc/common.h"

NS_ASSUME_NONNULL_BEGIN

/** Error domain of MediaPipe Task related errors. */
extern NSString *const MPPTasksErrorDomain;

/** Helper utility for the all tasks which encapsulates common functionality. */
@interface MPPCommonUtils : NSObject

/**
 * Creates and saves an NSError in the MediPipe task library domain, with the given code and
 * description.
 *
 * @param code Error code.
 * @param description Error description.
 * @param error Pointer to the memory location where the created error should be saved. If `nil`,
 * no error will be saved.
 */
+ (void)createCustomError:(NSError **)error
                 withCode:(NSUInteger)code
              description:(NSString *)description;

/**
 * Creates and saves an NSError with the given domain, code and description.
 *
 * @param error Pointer to the memory location where the created error should be saved. If `nil`,
 * no error will be saved.
 * @param domain Error domain.
 * @param code Error code.
 * @param description Error description.
 */
+ (void)createCustomError:(NSError **)error
               withDomain:(NSString *)domain
                     code:(NSUInteger)code
              description:(NSString *)description;

/**
 * Converts an absl::Status to an NSError.
 *
 * @param status absl::Status.
 * @param error Pointer to the memory location where the created error should be saved. If `nil`,
 * no error will be saved.
 * @return YES when there is no error, NO otherwise.
 */
+ (BOOL)checkCppError:(const absl::Status &)status toError:(NSError **)error;

/**
 * Allocates a block of memory with the specified size and returns a pointer to it. If memory
 * cannot be allocated because of an invalid `memSize`, it saves an error. In other cases, it
 * terminates program execution.
 *
 * @param memSize size of memory to be allocated
 * @param error Pointer to the memory location where errors if any should be saved. If `nil`, no
 * error will be saved.
 *
 * @return Pointer to the allocated block of memory on successful allocation. `nil` in case as
 * error is encountered because of invalid `memSize`. If failure is due to any other reason, method
 * terminates program execution.
 */
+ (void *)mallocWithSize:(size_t)memSize error:(NSError **)error;
@end

NS_ASSUME_NONNULL_END
