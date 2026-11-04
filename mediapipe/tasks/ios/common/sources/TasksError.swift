// Copyright 2026 The MediaPipe Authors.
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

import Foundation
package import third_party_mediapipe_tasks_c_core_common
package import third_party_mediapipe_tasks_c_core_mp_status

/// Error codes thrown by the MediaPipe Tasks library.
public enum TasksErrorCode: Int, Sendable, Equatable, Hashable, CaseIterable {
  /// Indicates the operation was cancelled, typically by the caller.
  case cancelled = 1

  /// Indicates an unknown error occurred.
  case unknown = 2

  /// Indicates the caller specified an invalid argument, such as a malformed filename.
  case invalidArgument = 3

  /// Indicates a deadline expired before the operation could complete.
  case deadlineExceeded = 4

  /// Indicates some requested entity (such as a file or directory) was not found.
  case notFound = 5

  /// Indicates that the entity a caller attempted to create is already present.
  case alreadyExists = 6

  /// Indicates that the caller does not have permission to execute the specified operation.
  case permissionDenied = 7

  /// Indicates some resource has been exhausted, such as a per-user quota or disk space.
  case resourceExhausted = 8

  /// Indicates that the operation was rejected because the system is not in a required state.
  case failedPrecondition = 9

  /// Indicates the operation was aborted, typically due to a concurrency or transaction failure.
  case aborted = 10

  /// Indicates the operation was attempted past the valid range.
  case outOfRange = 11

  /// Indicates the operation is not implemented or supported.
  case unimplemented = 12

  /// Indicates an internal error has occurred and underlying system invariants were violated.
  case internalError = 13

  /// Indicates the service is currently unavailable, typically as a transient condition.
  case unavailable = 14

  /// Indicates that unrecoverable data loss or corruption has occurred.
  case dataLoss = 15

  /// Indicates that the request does not have valid authentication credentials.
  case unauthenticated = 16

  /// Indicates that audio record permissions were denied by the user.
  case audioRecordPermissionDenied = 17

  /// Indicates that audio record permissions cannot be determined.
  case audioRecordPermissionUndetermined = 18

  /// Indicates that `AudioRecord` is waiting for new microphone input.
  case audioRecordWaitingForNewMicInput = 19

  /// Indicates that `AudioRecord` is not tapping the microphone.
  case audioRecordNotTappingMic = 20

  /// Fallback human-readable description when an explicit error message is unavailable.
  public var defaultDescription: String {
    switch self {
    case .cancelled:
      return "The operation was cancelled."
    case .unknown:
      return "An unknown error occurred."
    case .invalidArgument:
      return "An invalid argument was provided."
    case .deadlineExceeded:
      return "The operation deadline was exceeded."
    case .notFound:
      return "The requested resource was not found."
    case .alreadyExists:
      return "The resource already exists."
    case .permissionDenied:
      return "Permission was denied for the operation."
    case .resourceExhausted:
      return "A required resource was exhausted."
    case .failedPrecondition:
      return "A required precondition was not met."
    case .aborted:
      return "The operation was aborted."
    case .outOfRange:
      return "A value or index was out of range."
    case .unimplemented:
      return "The requested operation is not implemented."
    case .internalError:
      return "An internal MediaPipe Tasks error occurred."
    case .unavailable:
      return "The requested service is currently unavailable."
    case .dataLoss:
      return "Unrecoverable data loss or corruption occurred."
    case .unauthenticated:
      return "The request lacks valid authentication credentials."
    case .audioRecordPermissionDenied:
      return "Microphone recording permission was denied."
    case .audioRecordPermissionUndetermined:
      return "Microphone recording permission has not been determined."
    case .audioRecordWaitingForNewMicInput:
      return "AudioRecord is waiting for new microphone input."
    case .audioRecordNotTappingMic:
      return "AudioRecord is not actively tapping the microphone."
    }
  }

  /// Creates a `TasksErrorCode` corresponding to the non-OK `MpStatus` value.
  package init(cStatus: MpStatus) {
    switch cStatus {
    case kMpCancelled:
      self = .cancelled
    case kMpUnknown:
      self = .unknown
    case kMpInvalidArgument:
      self = .invalidArgument
    case kMpDeadlineExceeded:
      self = .deadlineExceeded
    case kMpNotFound:
      self = .notFound
    case kMpAlreadyExists:
      self = .alreadyExists
    case kMpPermissionDenied:
      self = .permissionDenied
    case kMpResourceExhausted:
      self = .resourceExhausted
    case kMpFailedPrecondition:
      self = .failedPrecondition
    case kMpAborted:
      self = .aborted
    case kMpOutOfRange:
      self = .outOfRange
    case kMpUnimplemented:
      self = .unimplemented
    case kMpInternal:
      self = .internalError
    case kMpUnavailable:
      self = .unavailable
    case kMpDataLoss:
      self = .dataLoss
    case kMpUnauthenticated:
      self = .unauthenticated
    default:
      self = .unknown
    }
  }
}

/// Structured error thrown by MediaPipe Tasks operations.
public struct TasksError: Error, Sendable, Equatable, Hashable {
  /// The categorized error code.
  public let code: TasksErrorCode

  /// Detailed description of the failure.
  public let message: String

  /// Creates a `TasksError` with the specified `code` and optional `message`.
  public init(code: TasksErrorCode, message: String? = nil) {
    self.code = code
    if let message, !message.isEmpty {
      self.message = message
    } else {
      self.message = code.defaultDescription
    }
  }

  /// Creates an invalid-argument `TasksError`.
  public static func invalidArgument(_ message: String) -> TasksError {
    TasksError(code: .invalidArgument, message: message)
  }

  /// Creates an internal `TasksError`.
  public static func internalError(_ message: String) -> TasksError {
    TasksError(code: .internalError, message: message)
  }

  /// Creates a failed-precondition `TasksError`.
  public static func failedPrecondition(_ message: String) -> TasksError {
    TasksError(code: .failedPrecondition, message: message)
  }

  /// Creates a not-found `TasksError`.
  public static func notFound(_ message: String) -> TasksError {
    TasksError(code: .notFound, message: message)
  }

  /// Creates an unimplemented `TasksError`.
  public static func unimplemented(_ message: String) -> TasksError {
    TasksError(code: .unimplemented, message: message)
  }
}

extension TasksError: LocalizedError {
  public var errorDescription: String? {
    message
  }
}

extension TasksError: CustomNSError {
  public static let errorDomain = "com.google.mediapipe.tasks"

  public var errorCode: Int {
    code.rawValue
  }

  public var errorUserInfo: [String: Any] {
    [NSLocalizedDescriptionKey: message]
  }
}

extension MpStatus {
  /// Invokes `body` with a managed C error-message out-pointer, freeing any allocated C string
  /// and throwing a `TasksError` when the returned `MpStatus` is not `kMpOk`.
  package static func check(
    defaultMessage: String? = nil,
    _ body: (inout UnsafeMutablePointer<CChar>?) -> MpStatus
  ) throws {
    var errorMessage: UnsafeMutablePointer<CChar>?
    let status = body(&errorMessage)
    try status.check(errorMessage: errorMessage, defaultMessage: defaultMessage)
  }

  /// Validates that the C status is `kMpOk`, or frees `errorMessage` and throws a `TasksError`.
  package func check(
    errorMessage: UnsafeMutablePointer<CChar>?,
    defaultMessage: String? = nil
  ) throws {
    defer {
      if let errorMessage {
        MpErrorFree(errorMessage)
      }
    }
    guard self == kMpOk else {
      let errorCode = TasksErrorCode(cStatus: self)
      let cDescription = errorMessage.map { String(cString: $0) }
      let description =
        if let cDescription, !cDescription.isEmpty {
          cDescription
        } else {
          defaultMessage
        }
      throw TasksError(code: errorCode, message: description)
    }
  }
}
