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
import com.google.protobuf.contrib.MessageUtils;

/**
 * Utility class for retrieving the protobuf type name for a MessageLite class. This implementation
 * uses the full-protobuf Message and Descriptor library.
 *
 * <p>This class is defined in separate source files for "full" or for "lite" dependencies.
 */
final class TypeNameRegistryConcrete implements TypeNameRegistry {

  /** Returns the protobuf type name for a Java Class. */
  @Override
  public <T extends MessageLite> String getTypeName(Class<T> clazz) {
    return MessageUtils.getProtoTypeName(clazz);
  }

  /** Records the protobuf type name for a Java Class. */
  @Override
  public <T extends MessageLite> void registerTypeName(Class<T> clazz, String typeName) {}
}

/** Satisfies Java file name convention. */
@SuppressWarnings("TopLevel")
final class TypeNameRegistryFull {}
