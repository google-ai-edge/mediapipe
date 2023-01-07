// Copyright 2022 The TensorFlow Authors. All Rights Reserved.
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

  void *allocedMemory = malloc(memSize);
  if (!allocedMemory) {
    exit(-1);
  }

  return allocedMemory;
}

+ (BOOL)checkCppError:(const absl::Status &)status toError:(NSError *_Nullable *)error {
  if (status.ok()) {
    return YES;
  }
  // Payload of absl::Status created by the MediaPipe task library stores an appropriate value of
  // the enum MediaPipeTasksStatus. The integer value corresponding to the MediaPipeTasksStatus enum
  // stored in the payload is extracted here to later map to the appropriate error code to be
  // returned. In cases where the enum is not stored in (payload is NULL or the payload string
  // cannot be converted to an integer), we set the error code value to be 1
  // (MPPTasksErrorCodeError of MPPTasksErrorCode used in the iOS library to signify
  // any errors not falling into other categories.) Since payload is of type absl::Cord that can be
  // type cast into an absl::optional<std::string>, we use the std::stoi function to convert it into
  // an integer code if possible.
  NSUInteger genericErrorCode = MPPTasksErrorCodeError;
  NSUInteger errorCode;
  try {
    // Try converting payload to integer if payload is not empty. Otherwise convert a string
    // signifying generic error code MPPTasksErrorCodeError to integer.
    errorCode =
        (NSUInteger)std::stoi(static_cast<absl::optional<std::string>>(
                                  status.GetPayload(mediapipe::tasks::kMediaPipeTasksPayload))
                                  .value_or(std::to_string(genericErrorCode)));
  } catch (std::invalid_argument &e) {
    // If non empty payload string cannot be converted to an integer. Set error code to 1(kError).
    errorCode = MPPTasksErrorCodeError;
  }

  // If errorCode is outside the range of enum values possible or is
  // MPPTasksErrorCodeError, we try to map the absl::Status::code() to assign
  // appropriate MPPTasksErrorCode in default cases. Note:
  // The mapping to absl::Status::code() is done to generate a more specific error code than
  // MPPTasksErrorCodeError in cases when the payload can't be mapped to
  // MPPTasksErrorCode. This can happen when absl::Status returned by TFLite library are in turn
  // returned without modification by MediaPipe cc library methods.
  if (errorCode > MPPTasksErrorCodeLast || errorCode <= MPPTasksErrorCodeFirst) {
    switch (status.code()) {
      case absl::StatusCode::kInternal:
        errorCode = MPPTasksErrorCodeError;
        break;
      case absl::StatusCode::kInvalidArgument:
        errorCode = MPPTasksErrorCodeInvalidArgumentError;
        break;
      case absl::StatusCode::kNotFound:
        errorCode = MPPTasksErrorCodeError;
        break;
      default:
        errorCode = MPPTasksErrorCodeError;
        break;
    }
  }

  // Creates the NSEror with the appropriate error
  // MPPTasksErrorCode and message. MPPTasksErrorCode has a one to one
  // mapping with MediaPipeTasksStatus starting from the value 1(MPPTasksErrorCodeError)
  // and hence will be correctly initialized if directly cast from the integer code derived from
  // MediaPipeTasksStatus stored in its payload. MPPTasksErrorCode omits kOk = 0 of
  // MediaPipeTasksStatusx.
  //
  // Stores a string including absl status code and message(if non empty) as the
  // error message See
  // https://github.com/abseil/abseil-cpp/blob/master/absl/status/status.h#L514
  // for explanation. absl::Status::message() can also be used but not always
  // guaranteed to be non empty.
  NSString *description = [NSString
      stringWithCString:status.ToString(absl::StatusToStringMode::kWithNoExtraData).c_str()
               encoding:NSUTF8StringEncoding];
  [MPPCommonUtils createCustomError:error withCode:errorCode description:description];
  return NO;
}

@end
