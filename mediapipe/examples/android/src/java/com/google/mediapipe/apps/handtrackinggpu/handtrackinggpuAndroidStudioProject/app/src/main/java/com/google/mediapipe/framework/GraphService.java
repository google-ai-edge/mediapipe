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
 * Implement this interface to wrap a native GraphService.
 *
 * <p>T should be the Java class wrapping the native service object.
 */
public interface GraphService<T> {
  /**
   * Provides the native service object corresponding to the provided Java object. This must be
   * handled by calling mediapipe::android::GraphServiceHelper::SetServiceObject in native code,
   * passing the provided context argument. We do it this way to minimize the number of trips
   * through JNI and maintain more type safety in the native code.
   */
  public void installServiceObject(long context, T object);
}
