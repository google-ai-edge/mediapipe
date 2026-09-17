/* Copyright 2026 The MediaPipe Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

package com.google.mediapipe.tasks.retrieval.components;

import com.google.mediapipe.tasks.retrieval.semanticretriever.proto.VectorStoresProto.TableConfig;

/** SQLite table column configuration. */
public final class ColumnConfig {
  /** The type of key this column is. */
  public enum KeyType {
    DEFAULT_NOT_KEY,
    PRIMARY_KEY,
  }

  private final String name;
  private final String sqlType;
  private final KeyType keyType;
  private final boolean autoIncrement;
  private final boolean isNullable;

  private ColumnConfig(
      String name, String sqlType, KeyType keyType, boolean autoIncrement, boolean isNullable) {
    this.name = name;
    this.sqlType = sqlType;
    this.keyType = keyType;
    this.autoIncrement = autoIncrement;
    this.isNullable = isNullable;
  }

  public String getName() {
    return name;
  }

  public String getSqlType() {
    return sqlType;
  }

  public KeyType getKeyType() {
    return keyType;
  }

  public boolean getAutoIncrement() {
    return autoIncrement;
  }

  public boolean getIsNullable() {
    return isNullable;
  }

  public static ColumnConfig create(String name, String sqlType) {
    return new ColumnConfig(name, sqlType, KeyType.DEFAULT_NOT_KEY, false, false);
  }

  public static ColumnConfig create(
      String name, String sqlType, KeyType keyType, boolean autoIncrement, boolean isNullable) {
    return new ColumnConfig(name, sqlType, keyType, autoIncrement, isNullable);
  }

  TableConfig.ColumnConfig toProto() {
    TableConfig.ColumnConfig.KeyType protoKeyType;
    switch (keyType) {
      case PRIMARY_KEY:
        protoKeyType = TableConfig.ColumnConfig.KeyType.PRIMARY_KEY;
        break;
      case DEFAULT_NOT_KEY:
        protoKeyType = TableConfig.ColumnConfig.KeyType.DEFAULT_NOT_KEY;
        break;
      default:
        throw new IllegalArgumentException("Unknown key type: " + keyType);
    }
    return TableConfig.ColumnConfig.newBuilder()
        .setName(name)
        .setSqlType(sqlType)
        .setKeyType(protoKeyType)
        .setAutoIncrement(autoIncrement)
        .setIsNullable(isNullable)
        .build();
  }
}
