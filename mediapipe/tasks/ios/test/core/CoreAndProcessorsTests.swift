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
import MediaPipeTasksCore
import Testing
import third_party_mediapipe_tasks_c_core_base_options
import third_party_mediapipe_tasks_c_core_common
import third_party_mediapipe_tasks_c_core_mp_status

struct CoreAndProcessorsTests {
  @Test(arguments: TasksErrorCode.allCases)
  func `TasksErrorCode provides non-empty default description`(code: TasksErrorCode) {
    #expect(code.defaultDescription.isEmpty == false)
  }

  @Test(
    arguments: zip(
      [
        kMpOk, kMpCancelled, kMpUnknown, kMpInvalidArgument, kMpDeadlineExceeded,
        kMpNotFound, kMpAlreadyExists, kMpPermissionDenied, kMpResourceExhausted,
        kMpFailedPrecondition, kMpAborted, kMpOutOfRange, kMpUnimplemented,
        kMpInternal, kMpUnavailable, kMpDataLoss, kMpUnauthenticated,
      ],
      [
        TasksErrorCode.unknown, .cancelled, .unknown, .invalidArgument, .deadlineExceeded,
        .notFound, .alreadyExists, .permissionDenied, .resourceExhausted,
        .failedPrecondition, .aborted, .outOfRange, .unimplemented,
        .internalError, .unavailable, .dataLoss, .unauthenticated,
      ]
    )
  )
  func `MpStatus maps to expected TasksErrorCode`(
    cStatus: MpStatus,
    expectedCode: TasksErrorCode
  ) {
    #expect(TasksErrorCode(cStatus: cStatus) == expectedCode)
  }

  @Test
  func `TasksError conforms to LocalizedError and CustomNSError`() {
    let error = TasksError.invalidArgument("Invalid model path.")
    #expect(error.code == .invalidArgument)
    #expect(error.message == "Invalid model path.")
    #expect(error.errorDescription == "Invalid model path.")
    #expect(TasksError.errorDomain == "com.google.mediapipe.tasks")
    #expect(error.errorCode == TasksErrorCode.invalidArgument.rawValue)
    #expect(
      error.errorUserInfo[NSLocalizedDescriptionKey] as? String == "Invalid model path."
    )

    let fallbackError = TasksError(code: .notFound, message: "")
    #expect(fallbackError.message == TasksErrorCode.notFound.defaultDescription)
  }

  @Test
  func `MpStatus check succeeds on kMpOk and throws TasksError on failure`() throws {
    try MpStatus.check { _ in kMpOk }

    #expect(throws: TasksError(code: .notFound, message: "Failed to open model file.")) {
      try MpStatus.check { errorMessage in
        errorMessage = strdup("Failed to open model file.")
        return kMpNotFound
      }
    }

    #expect(
      throws: TasksError(
        code: .internalError,
        message: TasksErrorCode.internalError.defaultDescription
      )
    ) {
      try MpStatus.check { _ in kMpInternal }
    }

    #expect(
      throws: TasksError(
        code: .internalError,
        message: "Custom fallback error."
      )
    ) {
      try MpStatus.check(defaultMessage: "Custom fallback error.") { _ in kMpInternal }
    }
  }

  @Test
  func `Delegate maps to expected MpDelegate constants`() {
    #expect(Delegate.default.rawValue == 0)
    #expect(Delegate.cpu.rawValue == 1)
    #expect(Delegate.gpu.rawValue == 2)
    #expect(Delegate.default.cDelegate == MP_DELEGATE_DEFAULT)
    #expect(Delegate.default.cDelegate.rawValue == 0)
    #expect(Delegate.cpu.cDelegate == MP_DELEGATE_CPU)
    #expect(Delegate.cpu.cDelegate.rawValue == 1)
    #expect(Delegate.gpu.cDelegate == MP_DELEGATE_GPU)
    #expect(Delegate.gpu.cDelegate.rawValue == 2)
  }

  @Test
  func `BaseOptions bridges path, buffer, and delegate to MpBaseOptions`() throws {
    let modelBytes = Data([0x4D, 0x50, 0x30, 0x31])
    let options = BaseOptions(
      modelAssetPath: "/tmp/model.tflite",
      modelAssetBuffer: modelBytes,
      delegate: .gpu
    )

    try options.withCOptions { cOptions in
      let bridgedPath = cOptions.model_asset_path.map { String(cString: $0) }
      #expect(bridgedPath == "/tmp/model.tflite")
      #expect(cOptions.model_asset_buffer_count == UInt32(modelBytes.count))
      let bufferPtr = try #require(cOptions.model_asset_buffer)
      let bridgedData = Data(bytes: bufferPtr, count: Int(cOptions.model_asset_buffer_count))
      #expect(bridgedData == modelBytes)
      #expect(cOptions.delegate == MP_DELEGATE_GPU)
      #expect(cOptions.file_descriptor == -1)
      #if os(iOS)
        #expect(cOptions.host_environment == MP_HOST_ENVIRONMENT_IOS)
        #expect(cOptions.host_system == MP_HOST_SYSTEM_IOS)
      #elseif os(macOS)
        #expect(cOptions.host_environment == MP_HOST_ENVIRONMENT_IOS)
        #expect(cOptions.host_system == MP_HOST_SYSTEM_MAC)
      #endif
      let hostVersion = try #require(cOptions.host_version.map { String(cString: $0) })
      #expect(hostVersion.split(separator: ".").count == 3)
      let infoAppVersion =
        Bundle.main.object(forInfoDictionaryKey: "CFBundleShortVersionString") as? String
      #expect(cOptions.app_id.map { String(cString: $0) } == Bundle.main.bundleIdentifier ?? "")
      #expect(cOptions.app_version.map { String(cString: $0) } == infoAppVersion ?? "")
    }
  }

  @Test
  func `BaseOptions omits empty path and nil buffer in MpBaseOptions`() {
    let options = BaseOptions()
    options.withCOptions { cOptions in
      #expect(cOptions.model_asset_path == nil)
      #expect(cOptions.model_asset_buffer == nil)
      #expect(cOptions.model_asset_buffer_count == 0)
      #expect(cOptions.delegate == MP_DELEGATE_CPU)
    }
  }
}
