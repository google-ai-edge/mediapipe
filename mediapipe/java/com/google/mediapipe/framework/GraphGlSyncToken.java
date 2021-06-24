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
 * Represents a synchronization point for OpenGL operations. It can be used to wait until the GPU
 * has reached the specified point in the sequence of commands it is executing. This can be
 * necessary when working with multiple GL contexts.
 */
public final class GraphGlSyncToken implements GlSyncToken {
  private long token;

  @Override
  public void waitOnCpu() {
    if (token != 0) {
      nativeWaitOnCpu(token);
    }
  }

  @Override
  public void waitOnGpu() {
    if (token != 0) {
      nativeWaitOnGpu(token);
    }
  }

  @Override
  public void release() {
    if (token != 0) {
      nativeRelease(token);
      token = 0;
    }
  }

  public GraphGlSyncToken(long token) {
    this.token = token;
  }

  private static native void nativeWaitOnCpu(long token);

  private static native void nativeWaitOnGpu(long token);

  private static native void nativeRelease(long token);
}
