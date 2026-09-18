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

package com.google.mediapipe.tasks.retrieval.chunking;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * A default text chunking implementation using native C++ word-based chunking with overlap or
 * character sliding window.
 */
public final class DefaultTextChunker implements TextChunker {
  /** Mode for text chunking. */
  public enum ChunkingMode {
    CHARACTER(0),
    WORD(1);

    private final int value;

    ChunkingMode(int value) {
      this.value = value;
    }

    public int getValue() {
      return value;
    }
  }

  private final int chunkSize;
  private final int chunkOverlap;
  private final ChunkingMode mode;

  static {
    System.loadLibrary("sqlite_vector_store_jni");
  }

  public DefaultTextChunker(int chunkSize, int chunkOverlap, ChunkingMode mode) {
    if (chunkSize <= 0) {
      throw new IllegalArgumentException("Chunk size must be positive.");
    }
    if (chunkOverlap < 0 || chunkOverlap >= chunkSize) {
      throw new IllegalArgumentException(
          "Chunk overlap must be non-negative and less than chunk size.");
    }
    this.chunkSize = chunkSize;
    this.chunkOverlap = chunkOverlap;
    this.mode = mode;
  }

  @Override
  public List<String> chunk(String text) {
    if (text == null || text.trim().isEmpty()) {
      return new ArrayList<>();
    }
    String[] chunks = nativeChunkText(text, chunkSize, chunkOverlap, mode.getValue());
    return chunks != null ? Arrays.asList(chunks) : new ArrayList<>();
  }

  private static native String[] nativeChunkText(
      String text, int chunkSize, int chunkOverlap, int chunkingMode);
}
