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
  @_exported import MediaPipeTasksImageSegmenter
#endif

/// Segmentation masks and quality scores produced by `InteractiveSegmenterLegacy`.
///
/// Shares `ImageSegmenterResult` because both tasks emit the same C `MpImageSegmenterResult`;
/// `timestampInMilliseconds` is always `0` since the legacy task only supports single images.
public typealias InteractiveSegmenterLegacyResult = ImageSegmenterResult
