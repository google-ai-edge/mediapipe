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
package import third_party_mediapipe_tasks_c_core_base_options

/// Hardware acceleration delegates supported by MediaPipe Tasks.
public enum Delegate: Int, Sendable, Equatable, Hashable {
  /// Uses the default hardware delegate configured by the MediaPipe runtime.
  case `default` = 0

  /// Executes inference on the CPU.
  case cpu = 1

  /// Executes inference on the GPU.
  case gpu = 2

  /// The underlying C delegate representation.
  package var cDelegate: MpDelegate {
    switch self {
    case .default:
      return MP_DELEGATE_DEFAULT
    case .cpu:
      return MP_DELEGATE_CPU
    case .gpu:
      return MP_DELEGATE_GPU
    }
  }
}

/// Base options shared across MediaPipe Tasks for model loading and hardware delegation.
public struct BaseOptions: Sendable, Equatable, Hashable {
  private enum HostTelemetry {
    static let hostVersion: String = {
      let version = ProcessInfo.processInfo.operatingSystemVersion
      return "\(version.majorVersion).\(version.minorVersion).\(version.patchVersion)"
    }()

    static let appID: String = Bundle.main.bundleIdentifier ?? ""

    static let appVersion: String =
      (Bundle.main.object(forInfoDictionaryKey: "CFBundleShortVersionString") as? String) ?? ""
  }

  /// Path to the model asset file on disk to open and memory-map.
  public var modelAssetPath: String

  /// In-memory model asset contents, if loading directly from a byte buffer.
  public var modelAssetBuffer: Data?

  /// Hardware backend delegate used to execute the MediaPipe graph.
  public var delegate: Delegate

  /// Creates `BaseOptions` with the given model path, optional in-memory buffer, and delegate.
  public init(
    modelAssetPath: String = "",
    modelAssetBuffer: Data? = nil,
    delegate: Delegate = .cpu
  ) {
    self.modelAssetPath = modelAssetPath
    self.modelAssetBuffer = modelAssetBuffer
    self.delegate = delegate
  }

  /// Invokes `body` with a C `MpBaseOptions` struct whose pointer fields are valid for the
  /// duration of the closure call.
  package func withCOptions<Result>(
    _ body: (MpBaseOptions) throws -> Result
  ) rethrows -> Result {
    let optionalPath = modelAssetPath.isEmpty ? nil : modelAssetPath
    return try optionalPath.withCString { pathPtr in
      try HostTelemetry.hostVersion.withCString { hostVersionPtr in
        try HostTelemetry.appID.withCString { appIDPtr in
          try HostTelemetry.appVersion.withCString { appVersionPtr in
            try withModelAssetBuffer { bufferPtr, bufferCount in
              var cOptions = MpBaseOptions()
              cOptions.model_asset_path = pathPtr
              cOptions.model_asset_buffer = bufferPtr
              cOptions.model_asset_buffer_count = bufferCount
              cOptions.file_descriptor = -1
              cOptions.delegate = delegate.cDelegate
              #if os(iOS)
                cOptions.host_environment = MP_HOST_ENVIRONMENT_IOS
                cOptions.host_system = MP_HOST_SYSTEM_IOS
              #elseif os(macOS)
                cOptions.host_environment = MP_HOST_ENVIRONMENT_IOS
                cOptions.host_system = MP_HOST_SYSTEM_MAC
              #endif
              cOptions.host_version = hostVersionPtr
              cOptions.app_id = appIDPtr
              cOptions.app_version = appVersionPtr
              return try body(cOptions)
            }
          }
        }
      }
    }
  }

  private func withModelAssetBuffer<Result>(
    _ body: (UnsafePointer<CChar>?, UInt32) throws -> Result
  ) rethrows -> Result {
    guard let modelAssetBuffer, !modelAssetBuffer.isEmpty else {
      return try body(nil, 0)
    }
    return try modelAssetBuffer.withUnsafeBytes { rawBuffer in
      try rawBuffer.withMemoryRebound(to: CChar.self) { charBuffer in
        try body(charBuffer.baseAddress, UInt32(clamping: charBuffer.count))
      }
    }
  }
}

extension String? {
  /// Invokes `body` with a pointer to the UTF-8 C string representation of the wrapped string,
  /// or `nil` when the optional is `nil`.
  package func withCString<Result>(
    _ body: (UnsafePointer<CChar>?) throws -> Result
  ) rethrows -> Result {
    switch self {
    case .some(let value):
      return try value.withCString(body)
    case .none:
      return try body(nil)
    }
  }
}
