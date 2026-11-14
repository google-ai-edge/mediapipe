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

#if !MEDIAPIPE_SINGLE_MODULE
  @_exported import MediaPipeTasksCore
#endif

/// Options for configuring an `InteractiveSegmenter` task.
public struct InteractiveSegmenterOptions: Sendable, Equatable, Hashable {
  /// Base options for configuring model loading and hardware acceleration delegation.
  public var baseOptions: BaseOptions

  /// Creates options configured with `baseOptions`.
  public init(baseOptions: BaseOptions = BaseOptions()) {
    self.baseOptions = baseOptions
  }
}
