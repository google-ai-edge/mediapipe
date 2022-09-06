// Copyright 2019 The MediaPipe Authors.
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

package com.google.mediapipe.framework;

/**
 * Represents a synchronization point for OpenGL operations. This can be needed when working with
 * multiple GL contexts.
 */
public interface GlSyncToken {
  /**
   * Waits until the GPU has executed all commands up to the sync point. This blocks the CPU, and
   * ensures the commands are complete from the point of view of all threads and contexts.
   */
  void waitOnCpu();

  /**
   * Ensures that the following commands on the current OpenGL context will not be executed until
   * the sync point has been reached. This does not block the CPU, and only affects the current
   * OpenGL context.
   */
  void waitOnGpu();

  /** Releases the underlying native object. */
  void release();

  /** Returns a handle to the underlying native object. For internal use. */
  long nativeToken();
}
