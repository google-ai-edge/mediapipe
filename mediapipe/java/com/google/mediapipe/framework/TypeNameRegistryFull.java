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
