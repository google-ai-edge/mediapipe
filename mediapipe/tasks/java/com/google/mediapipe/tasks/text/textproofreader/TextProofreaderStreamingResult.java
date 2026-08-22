// Copyright 2026 The MediaPipe Authors.
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

package com.google.mediapipe.tasks.text.textproofreader;

import com.google.mediapipe.tasks.text.textproofreader.TextProofreaderResult.Correction;
import java.util.List;

/** Represents the streaming proofreading result. */
public final class TextProofreaderStreamingResult {
  private final String chunk;
  private final List<Correction> corrections;
  private final boolean done;

  TextProofreaderStreamingResult(String chunk, List<Correction> corrections, boolean done) {
    this.chunk = chunk;
    this.corrections = corrections;
    this.done = done;
  }

  /** Returns the proofread text chunk. */
  public String getChunk() {
    return chunk;
  }

  /**
   * Returns the list of unchanged, deleted, and inserted text segments. Only populated when
   * done=true.
   */
  public List<Correction> getCorrections() {
    return corrections;
  }

  /** Returns whether the streaming is done. */
  public boolean isDone() {
    return done;
  }
}
