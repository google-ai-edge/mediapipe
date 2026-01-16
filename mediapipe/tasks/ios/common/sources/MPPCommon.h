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
 * @enum MPPTasksErrorCode
 * This enum specifies  error codes for errors thrown by iOS MediaPipe Task Library.
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

  /**
   * Indicates that the entity a caller attempted to create (such as a file or directory) is
   * already present.
   */
  MPPTasksErrorCodeAlreadyExistsError = 6,

  /** Indicates that the caller does not have permission to execute the specified operation. */
  MPPTasksErrorCodePermissionDeniedError = 7,

  /**
   * Indicates some resource has been exhausted, perhaps a per-user quota, or perhaps the entire
   * file system is out of space.
   */
  MPPTasksErrorCodeResourceExhaustedError = 8,

  /**
   * Indicates that the operation was rejected because the system is not in a state required for
   * the operation's execution. For example, a directory to be deleted may be non-empty, an "rmdir"
   * operation is applied to a non-directory, etc.
   */
  MPPTasksErrorCodeFailedPreconditionError = 9,

  /**
   * Indicates the operation was aborted, typically due to a concurrency issue such as a sequencer
   * check failure or a failed transaction.
   */
  MPPTasksErrorCodeAbortedError = 10,

  /**
   * Indicates the operation was attempted past the valid range, such as seeking or reading past an
   * end-of-file.
   */
  MPPTasksErrorCodeOutOfRangeError = 11,

  /**
   * Indicates the operation is not implemented or supported in this service. In this case, the
   * operation should not be re-attempted.
   */
  MPPTasksErrorCodeUnimplementedError = 12,

  /**
   * Indicates an internal error has occurred and some invariants expected by the underlying system
   * have not been satisfied. This error code is reserved for serious errors.
   */
  MPPTasksErrorCodeInternalError = 13,

  /**
   * Indicates the service is currently unavailable and that this is most likely a transient
   * condition.
   */
  MPPTasksErrorCodeUnavailableError = 14,

  /** Indicates that unrecoverable data loss or corruption has occurred. */
  MPPTasksErrorCodeDataLossError = 15,

  /**
   * Indicates that the request does not have valid authentication credentials for the operation.
   */
  MPPTasksErrorCodeUnauthenticatedError = 16,

  /** Indicates that audio record permissions were denied by the user. */
  MPPTasksErrorCodeAudioRecordPermissionDeniedError = 17,

  /**
   * Audio record permissions cannot be determined. If this error is returned by Audio, the caller
   * has to acquire permissions using AVFoundation.
   */
  MPPTasksErrorCodeAudioRecordPermissionUndeterminedError = 18,

  /** Indicates that `AudioRecord` is waiting for new microphone input. */
  MPPTasksErrorCodeAudioRecordWaitingForNewMicInputError = 19,

  /**
   * Indicates that `AudioRecord` is not tapping the microphone. Operations permitted only while the
   * microphone is being actively tapped can return this error.
   */
  MPPTasksErrorCodeAudioRecordNotTappingMicError = 20,

  /** The first error code in MPPTasksErrorCode (for internal use only). */
  MPPTasksErrorCodeFirst = MPPTasksErrorCodeCancelledError,

  /** The last error code in MPPTasksErrorCode (for internal use only). */
  MPPTasksErrorCodeLast = MPPTasksErrorCodeAudioRecordNotTappingMicError,

} NS_SWIFT_NAME(TasksErrorCode);

NS_ASSUME_NONNULL_END
