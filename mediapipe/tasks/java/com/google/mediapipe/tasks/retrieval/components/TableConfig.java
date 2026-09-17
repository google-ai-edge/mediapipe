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

import static com.google.mediapipe.tasks.retrieval.semanticretriever.proto.VectorStoresProto.TableConfig.newBuilder;

import com.google.common.collect.ImmutableList;
import com.google.mediapipe.tasks.retrieval.semanticretriever.proto.VectorStoresProto.TableConfig.Builder;

/** SQLite table configuration. */
public final class TableConfig {
  private final String name;
  private final ImmutableList<ColumnConfig> columns;

  private TableConfig(String name, ImmutableList<ColumnConfig> columns) {
    this.name = name;
    this.columns = columns;
  }

  public String getName() {
    return name;
  }

  public ImmutableList<ColumnConfig> getColumns() {
    return columns;
  }

  public static TableConfig create(String name, ImmutableList<ColumnConfig> columns) {
    return new TableConfig(name, columns);
  }

  byte[] toProtoBytes() {
    Builder builder = newBuilder().setName(name);
    for (ColumnConfig columnConfig : columns) {
      builder.addColumns(columnConfig.toProto());
    }
    return builder.build().toByteArray();
  }
}
