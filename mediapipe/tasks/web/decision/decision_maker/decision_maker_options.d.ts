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

import {TaskRunnerOptions} from '../../../../tasks/web/core/task_runner_options';

/**
 * A single candidate option for binary, categorical, or ordinal questions.
 */
export declare interface ClassifierOption {
  /** Short identifier or level label (e.g. "billing", "1", "true"). */
  label: string;
  /** Optional natural language description of the option criteria. */
  description?: string;
}

/**
 * Supported question types in a ClassifierSchema.
 */
export type ClassifierQuestionType =
  | 'binary'
  | 'categorical'
  | 'ordinal'
  | 'boolean'
  | 'choice'
  | 'score';

/**
 * A single question within a multi-question ClassifierSchema.
 */
export declare interface ClassifierQuestion {
  /** Unique identifier key in the returned ClassifierResult record. */
  id: string;
  /** Question evaluation type. */
  type: ClassifierQuestionType;
  /** Task instruction or natural language condition prompt. */
  prompt: string;
  /** Candidate options (required for categorical and ordinal; optional for binary). */
  options?: ClassifierOption[];
  /** Decision threshold in [0.0, 1.0] for binary questions. Defaults to 0.5. */
  threshold?: number;
  /**
   * Softmax temperature override (> 0.0). When unset or <= 0, the model's
   * calibrated temperature is used.
   */
  temperature?: number;
  /** Whether to normalize against the empty-context ("[None]") prior. */
  normalizePrior?: boolean;
}

/**
 * Expected input modality and language configuration.
 */
export declare interface ClassifierExpectedInput {
  type: 'text' | 'image' | 'audio';
  languages?: string[];
}

/**
 * Multi-question decision schema compatible with the Web Classifier API.
 */
export declare interface ClassifierSchema {
  /** Optional domain or system context prepended to question instructions. */
  context?: string;
  /** Optional expected input modalities and languages. */
  expectedInputs?: ClassifierExpectedInput[];
  /** Ordered list of questions to evaluate against the input. */
  questions?: ClassifierQuestion[];
}

/**
 * Per-call options for `evaluate()`, `evaluateBatch()`, and `classify()`.
 */
export declare interface ClassifierEvaluateOptions {
  /** Optional per-call context prepended to the input text. */
  context?: string;
  /** Optional AbortSignal to cancel an in-flight evaluation. */
  signal?: AbortSignal;
}

/**
 * Question configuration for a Boolean Predicate evaluation.
 */
export declare interface BooleanQuestion {
  /** The natural language boolean condition to evaluate. */
  condition: string;
  /** Probability threshold for true. Defaults to 0.5. */
  threshold?: number;
  /**
   * Softmax temperature override (> 0.0). When unset or <= 0, the model's
   * calibrated temperature is used.
   */
  temperature?: number;
  /**
   * Subtracts the memoized empty-context ("[None]") prior distribution in
   * log-probability space to eliminate positional or label bias.
   */
  normalizePrior?: boolean;
  /**
   * Optional custom false/true option descriptions (defaults to Laya's
   * "false: no, the statement does not hold" and
   * "true: yes, the statement holds").
   */
  options?: ClassifierOption[];
  /** Optional domain or schema context prepended to the condition. */
  context?: string;
}

/**
 * Question configuration for a Categorical Choice evaluation.
 */
export declare interface ChoiceQuestion {
  /** Map of choice keys to descriptive phrases or criteria. */
  criteria?: Record<string, string>;
  /**
   * Ordered list of candidate options with `{ label, description }`.
   * Can be provided instead of or alongside `criteria`.
   */
  options?: ClassifierOption[];
  /**
   * Softmax temperature override (> 0.0). When unset or <= 0, the model's
   * calibrated temperature is used.
   */
  temperature?: number;
  /** Optional task instruction prompt for the model. */
  instructions?: string;
  /** Optional domain or schema context prepended to instructions. */
  context?: string;
  /**
   * Scoring mode (0=SCORING_AUTO, 1=SCORING_SINGLE_TOKEN,
   * 2=SCORING_FULL_OPTION). Defaults to 0 (SCORING_AUTO).
   */
  scoringMode?: number;
  /**
   * Subtracts the memoized empty-context ("[None]") prior distribution in
   * log-probability space to eliminate positional or label bias.
   */
  normalizePrior?: boolean;
}

/**
 * Question configuration for an Ordinal Score evaluation.
 */
export declare interface ScoreQuestion {
  /** List of ordered rubric grade descriptors from lowest to highest. */
  rubric?: string[];
  /**
   * Ordered list of rubric levels with `{ label, description }`. When all
   * `label` values parse as finite numbers (e.g. "1".."5"), `expectedScore`
   * is computed directly over those numeric level values.
   */
  options?: ClassifierOption[];
  /**
   * Softmax temperature override (> 0.0). When unset or <= 0, the model's
   * calibrated temperature is used.
   */
  temperature?: number;
  /** Optional task instruction prompt for the model. */
  instructions?: string;
  /** Optional domain or schema context prepended to instructions. */
  context?: string;
}

/**
 * Options to configure the MediaPipe Semantic Decision Maker Task.
 */
export declare interface DecisionMakerOptions extends TaskRunnerOptions {
  /** Backend type (0=AUTO, 1=ENCODER, 2=DIRECT_LOGIT, 3=EMBEDDING). */
  backendType?: number;
  /** The maximum number of tokens in prompt context. */
  maxNumTokens?: number;
  /** Optional session-bound ClassifierSchema for `classify()`. */
  schema?: ClassifierSchema;
  /** Optional session-level context prepended to question prompts. */
  context?: string;
  /** Optional expected input modalities and languages. */
  expectedInputs?: ClassifierExpectedInput[];
  /** Optional session-bound questions for `classify()`. */
  questions?: ClassifierQuestion[];
  /**
   * Optional byte length of `baseOptions.modelAssetBuffer` when provided as a
   * `ReadableStreamDefaultReader`, enabling single-pass zero-copy streaming
   * allocation directly in the Wasm heap.
   */
  modelAssetSize?: number;
  /**
   * Optional URL or path to a companion per-layer embedder `.tflite` model
   * (e.g. `*_per_layer_embedder.tflite` for Gemma 4 E2B).
   */
  companionPleAssetPath?: string;
  /**
   * Optional buffer or stream reader for a companion per-layer embedder
   * `.tflite` model.
   */
  companionPleAssetBuffer?: Uint8Array | ReadableStreamDefaultReader;
  /**
   * Optional byte length of `companionPleAssetBuffer` when provided as a
   * `ReadableStreamDefaultReader`.
   */
  companionPleAssetSize?: number;
}
