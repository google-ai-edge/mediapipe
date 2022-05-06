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

import com.google.protobuf.ExtensionRegistryLite;
import com.google.protobuf.InvalidProtocolBufferException;
import com.google.protobuf.MessageLite;
import java.util.NoSuchElementException;

/** Utility functions for translating MediaPipe packet data between languages. */
public final class ProtoUtil {
  /** Records the protobuf type name for a Java Class. */
  public static <T extends MessageLite> void registerTypeName(Class<T> clazz, String typeName) {
    typeNameRegistry.registerTypeName(clazz, typeName);
  }

  /** Returns the protobuf type name for a Java Class. */
  public static <T extends MessageLite> String getTypeName(Class<T> clazz) {
    return typeNameRegistry.getTypeName(clazz);
  }

  /** Returns the best available ExtensionRegistry */
  public static ExtensionRegistryLite getExtensionRegistry() {
    return ExtensionRegistryLite.getEmptyRegistry();
  }

  /** Serializes a MessageLite into a SerializedMessage object. */
  public static <T extends MessageLite> SerializedMessage pack(T message) {
    SerializedMessage result = new SerializedMessage();
    result.typeName = getTypeName(message.getClass());
    if (result.typeName == null) {
      throw new NoSuchElementException(
          "Cannot determine the protobuf type name for class: "
              + message.getClass()
              + ". Have you called ProtoUtil.registerTypeName?");
    }
    result.value = message.toByteArray();
    return result;
  }

  /** Deserializes a MessageLite from a SerializedMessage object. */
  public static <T extends MessageLite> T unpack(SerializedMessage serialized, T defaultInstance)
      throws InvalidProtocolBufferException {
    String expectedType = ProtoUtil.getTypeName(defaultInstance.getClass());
    if (!serialized.typeName.equals(expectedType)) {
      throw new InvalidProtocolBufferException(
          "Message type does not match the expected type. Expected: "
              + expectedType
              + " Got: "
              + serialized.typeName);
    }
    // Specifying the ExtensionRegistry is recommended.  The ExtensionRegistry is
    // needed to deserialize any nested proto2 extension Messages.
    @SuppressWarnings("unchecked") // The type_url indicates type T.
    T result =
        (T)
            defaultInstance
                .getParserForType()
                .parseFrom(serialized.value, ProtoUtil.getExtensionRegistry());
    return result;
  }

  /** A singleton to find protobuf full type names. */
  static TypeNameRegistry typeNameRegistry = new TypeNameRegistryConcrete();

  private ProtoUtil() {}

  static class SerializedMessage {
    public String typeName;
    public byte[] value;
  }
}
