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
 * A callback that gets invoked when a texture is no longer in use.
 */
public interface TextureReleaseCallback {
  /**
   * Called when the texture has been released. The sync token can be used to ensure that the GPU is
   * done reading from it. Implementations of this interface should release the token once they are
   * done with it.
   */
  void release(GlSyncToken syncToken);
}
