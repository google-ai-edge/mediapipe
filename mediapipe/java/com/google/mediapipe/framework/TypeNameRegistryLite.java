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
import java.util.HashMap;
import java.util.Map;

/**
 * Utility class for retrieving the protobuf type name for a MessageLite class. This implementation
 * uses the mediapipe protobuf type names registry.
 *
 * <p>This class is defined in separate source files for "full" or for "lite" dependencies.
 */
final class TypeNameRegistryConcrete implements TypeNameRegistry {

  TypeNameRegistryConcrete() {}

  /** Returns the protobuf type name for a Java Class. */
  @Override
  public <T extends MessageLite> String getTypeName(Class<T> javaClass) {
    return typeNames.get(javaClass);
  }

  /** Records the protobuf type name for a Java Class. */
  @Override
  public <T extends MessageLite> void registerTypeName(Class<T> clazz, String typeName) {
    if (typeNames.containsKey(clazz) && !typeNames.get(clazz).equals(typeName)) {
      throw new MediaPipeException(
          MediaPipeException.StatusCode.ALREADY_EXISTS.ordinal(),
          "Protobuf type name: " + typeName + " conflicts with: " + typeNames.get(clazz));
    }
    typeNames.put(clazz, typeName);
  }

  /** A mapping from java package names to proto package names. */
  private final Map<Class<? extends MessageLite>, String> typeNames = new HashMap<>();
}

/** Satisfies Java file name convention. */
@SuppressWarnings("TopLevel")
final class TypeNameRegistryLite {}
