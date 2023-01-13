// Copyright 2022 The MediaPipe Authors. All Rights Reserved.
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
 * @enum MPPTasksErrorCode
 * This enum specifies  error codes for MediaPipe Task Library.
 * It maintains a 1:1 mapping to MediaPipeTasksStatus of the C ++libray.
 */
typedef NS_ENUM(NSUInteger, MPPTasksErrorCode) {

  // Generic error codes.

  /** Indicates the operation was cancelled, typically by the caller. */
  MPPTasksErrorCodeCancelledError = 1,
  /** Indicates an unknown error occurred. */
  MPPTasksErrorCodeUnknownError = 2,
  /** Indicates the caller specified an invalid argument, such as a malformed filename. */
  MPPTasksErrorCodeInvalidArgumentError = 3,
  /** Indicates a deadline expired before the operation could complete. */
  MPPTasksErrorCodeDeadlineExceededError = 4,
  /** Indicates some requested entity (such as a file or directory) was not found. */
  MPPTasksErrorCodeNotFoundError = 5,
  /** Indicates that the entity a caller attempted to create (such as a file or directory) is already present. */
  MPPTasksErrorCodeAlreadyExistsError = 6,
  /** Indicates that the caller does not have permission to execute the specified operation. */
  MPPTasksErrorCodePermissionDeniedError = 7,

  MPPTasksErrorCodeResourceExhaustedError = 8,

  MPPTasksErrorCodeFailedPreconditionError = 9,

  MPPTasksErrorCodeAbortedError = 10,

  MPPTasksErrorCodeOutOfRangeError = 11,

  MPPTasksErrorCodeUnimplementedError = 12,

  MPPTasksErrorCodeInternalError = 13,

  MPPTasksErrorCodeUnavailableError = 14,

  MPPTasksErrorCodeDataLossError = 15,

  MPPTasksErrorCodeUnauthenticatedError = 16,

  // The first error code in MPPTasksErrorCode (for internal use only).
  MPPTasksErrorCodeFirst = MPPTasksErrorCodeCancelledError,

  // The last error code in MPPTasksErrorCode (for internal use only).
  MPPTasksErrorCodeLast = MPPTasksErrorCodeUnauthenticatedError,

} NS_SWIFT_NAME(TasksErrorCode);

NS_ASSUME_NONNULL_END
