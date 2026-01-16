// Copyright 2020 The MediaPipe Authors.
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

import com.google.protobuf.MessageLite;

/**
 * Utility interface for retrieving the protobuf type name for a MessageLite class.
 */
interface TypeNameRegistry {

  /** Returns the protobuf type name for a Java Class. */
  public <T extends MessageLite> String getTypeName(Class<T> clazz);

  /** Records the protobuf type name for a Java Class. */
  public <T extends MessageLite> void registerTypeName(Class<T> clazz, String typeName);
}
