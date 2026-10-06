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

import java.util.List;

/** The result of a proofreading operation. */
public class TextProofreaderResult {
  private final String proofreadText;
  private final List<Correction> corrections;

  TextProofreaderResult(String proofreadText, List<Correction> corrections) {
    this.proofreadText = proofreadText;
    this.corrections = corrections;
  }

  /** Returns the proofread text with corrections applied. */
  public String getProofreadText() {
    return proofreadText;
  }

  /** Returns the list of unchanged, deleted, and inserted text segments. */
  public List<Correction> getCorrections() {
    return corrections;
  }

  /** The type of correction. */
  public enum CorrectionType {
    /** The text is the same in the original and proofread text. */
    SAME,
    /** The text was inserted in the proofread text. */
    INSERTION,
    /** The text was deleted from the original text. */
    DELETION
  }

  /** A single correction. */
  public static class Correction {
    private final CorrectionType type;
    private final String text;

    Correction(CorrectionType type, String text) {
      this.type = type;
      this.text = text;
    }

    /** Returns the type of correction. */
    public CorrectionType getType() {
      return type;
    }

    /** Returns the text associated with the correction. */
    public String getText() {
      return text;
    }
  }
}
