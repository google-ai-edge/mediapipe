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
