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
 * Result of evaluating a Boolean.
 */
export declare interface BooleanResult {
  /** The boolean decision based on threshold. */
  value: boolean;
  /** Probability of true in [0.0, 1.0]. */
  probabilityTrue: number;
  /** Normalized certainty confidence in [0.0, 1.0]. */
  confidence: number;
}

/**
 * Result of evaluating a Choice.
 */
export declare interface ChoiceResult {
  /** The winning choice key with highest probability. */
  selectedKey: string;
  /** Probability distribution across all candidate choice keys. */
  probabilities: Record<string, number>;
  /** Normalized certainty confidence in [0.0, 1.0]. */
  confidence: number;
  /**
   * Conformal prediction set of candidate choice keys covering >= 90%
   * probability mass (size == 1 to execute immediately; size > 1 to escalate
   * or disambiguate).
   */
  predictionSet?: string[];
}

/**
 * Result of evaluating an Score.
 */
export declare interface ScoreResult {
  /**
   * Expected score computed as dot product of probabilities and rubric
   * indices (or numeric option labels when provided).
   */
  expectedScore: number;
  /** Probability distribution across each ordinal grade in rubric. */
  probabilities: number[];
  /** Normalized certainty confidence in [0.0, 1.0]. */
  confidence: number;
  /** Winning rubric level label/key with highest probability. */
  selectedKey?: string;
}

/**
 * Calibrated probability entry for a single candidate option.
 */
export declare interface ClassifierOptionProbability {
  label: string;
  probability: number;
}

/**
 * Unified decision output for a single ClassifierQuestion.
 */
export declare interface ClassifierDecision {
  /** Question identifier matching `ClassifierQuestion.id`. */
  id: string;
  /** Winning option label (`"true"`/`"false"` for binary, winning label for categorical/ordinal). */
  label: string;
  /** Normalized certainty confidence in [0.0, 1.0]. */
  confidence: number;
  /** Binary only: calibrated P(true) in [0.0, 1.0]. */
  probability?: number;
  /** Ordinal only: continuous expected score across levels, sum(level_i * p_i). */
  expectedScore?: number;
  /** Full calibrated probability distribution over candidate options. */
  probabilities: ClassifierOptionProbability[];
  /** Optional conformal prediction set covering >= 90% probability mass. */
  predictionSet?: string[];
}

/**
 * Record mapping each question `id` in a `ClassifierSchema` to its `ClassifierDecision`.
 */
export type ClassifierResult = Record<string, ClassifierDecision>;
