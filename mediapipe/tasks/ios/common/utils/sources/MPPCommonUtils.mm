// Copyright 2022 The TensorFlow Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#import "mediapipe/tasks/ios/common/utils/sources/MPPCommonUtils.h"

#import "mediapipe/tasks/ios/common/sources/MPPCommon.h"

#include <string>

#include "absl/status/status.h"  // from @com_google_absl
#include "absl/strings/cord.h"   // from @com_google_absl
#include "mediapipe/tasks/cc/common.h"

/** Error domain of MediaPipe task library errors. */
NSString *const MPPTasksErrorDomain = @"com.google.mediapipe.tasks";

namespace {
using absl::StatusCode;
}

@implementation MPPCommonUtils

+ (void)createCustomError:(NSError **)error
                 withCode:(NSUInteger)code
              description:(NSString *)description {
  [MPPCommonUtils createCustomError:error
                         withDomain:MPPTasksErrorDomain
                               code:code
                        description:description];
}

+ (void)createCustomError:(NSError **)error
               withDomain:(NSString *)domain
                     code:(NSUInteger)code
              description:(NSString *)description {
  if (error) {
    *error = [NSError errorWithDomain:domain
                                 code:code
                             userInfo:@{NSLocalizedDescriptionKey : description}];
  }
}

+ (void *)mallocWithSize:(size_t)memSize error:(NSError **)error {
  if (!memSize) {
    [MPPCommonUtils createCustomError:error
                             withCode:MPPTasksErrorCodeInvalidArgumentError
                          description:@"memSize cannot be zero."];
    return NULL;
  }

  void *allocatedMemory = malloc(memSize);
  if (!allocatedMemory) {
    exit(-1);
  }

  return allocatedMemory;
}

+ (BOOL)checkCppError:(const absl::Status &)status toError:(NSError *_Nullable *)error {
  if (status.ok()) {
    return YES;
  }

  // Converts the absl status message to an NSString.
  NSString *description = [NSString
      stringWithCString:status.ToString(absl::StatusToStringMode::kWithNoExtraData).c_str()
               encoding:NSUTF8StringEncoding];

  MPPTasksErrorCode errorCode = MPPTasksErrorCodeUnknownError;

  // Maps the absl::StatusCode to the appropriate MPPTasksErrorCode. Note: MPPTasksErrorCode omits
  // absl::StatusCode::kOk.
  switch (status.code()) {
    case StatusCode::kCancelled:
      errorCode = MPPTasksErrorCodeCancelledError;
      break;
    case StatusCode::kUnknown:
      errorCode = MPPTasksErrorCodeUnknownError;
      break;
    case StatusCode::kInvalidArgument:
      errorCode = MPPTasksErrorCodeInvalidArgumentError;
      break;
    case StatusCode::kDeadlineExceeded:
      errorCode = MPPTasksErrorCodeDeadlineExceededError;
      break;
    case StatusCode::kNotFound:
      errorCode = MPPTasksErrorCodeNotFoundError;
      break;
    case StatusCode::kAlreadyExists:
      errorCode = MPPTasksErrorCodeAlreadyExistsError;
      break;
    case StatusCode::kPermissionDenied:
      errorCode = MPPTasksErrorCodePermissionDeniedError;
      break;
    case StatusCode::kResourceExhausted:
      errorCode = MPPTasksErrorCodeResourceExhaustedError;
      break;
    case StatusCode::kFailedPrecondition:
      errorCode = MPPTasksErrorCodeFailedPreconditionError;
      break;
    case StatusCode::kAborted:
      errorCode = MPPTasksErrorCodeAbortedError;
      break;
    case StatusCode::kOutOfRange:
      errorCode = MPPTasksErrorCodeOutOfRangeError;
      break;
    case StatusCode::kUnimplemented:
      errorCode = MPPTasksErrorCodeUnimplementedError;
      break;
    case StatusCode::kInternal:
      errorCode = MPPTasksErrorCodeInternalError;
      break;
    case StatusCode::kUnavailable:
      errorCode = MPPTasksErrorCodeUnavailableError;
      break;
    case StatusCode::kDataLoss:
      errorCode = MPPTasksErrorCodeDataLossError;
      break;
    case StatusCode::kUnauthenticated:
      errorCode = MPPTasksErrorCodeUnauthenticatedError;
      break;
    default:
      break;
  }

  [MPPCommonUtils createCustomError:error withCode:errorCode description:description];
  return NO;
}

@end
