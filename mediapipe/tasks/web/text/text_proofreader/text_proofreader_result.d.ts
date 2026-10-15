/**
 * Copyright 2026 The MediaPipe Authors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/**
 * The type of a correction:
 * - 'SAME': The text is the same in the original and proofread text.
 * - 'INSERTION': The text was inserted in the proofread text.
 * - 'DELETION': The text was deleted from the original text.
 */
export type CorrectionType = 'SAME' | 'INSERTION' | 'DELETION';

/** A single correction. */
export declare interface Correction {
  /** The type of correction. */
  type: CorrectionType;
  /** The text associated with the correction. */
  text: string;
}

/** The result of text proofreading. */
export declare interface TextProofreaderResult {
  /** The generated proofread text. */
  proofreadText: string;
  /**
   * The list of corrections to transform the input text into the proofread
   * text.
   */
  corrections: Correction[];
}
