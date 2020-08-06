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
