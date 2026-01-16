/**
 * Copyright 2022 The MediaPipe Authors.
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

/** Options to configure a MediaPipe Classifier Task. */
export declare interface ClassifierOptions {
  /**
   * The locale to use for display names specified through the TFLite Model
   * Metadata, if any. Defaults to English.
   */
  displayNamesLocale?: string|undefined;

  /** The maximum number of top-scored detection results to return. */
  maxResults?: number|undefined;

  /**
   * Overrides the value provided in the model metadata. Results below this
   * value are rejected.
   */
  scoreThreshold?: number|undefined;

  /**
   * Allowlist of category names. If non-empty, detection results whose category
   * name is not in this set will be filtered out. Duplicate or unknown category
   * names are ignored. Mutually exclusive with `categoryDenylist`.
   */
  categoryAllowlist?: string[]|undefined;

  /**
   * Denylist of category names. If non-empty, detection results whose category
   * name is in this set will be filtered out. Duplicate or unknown category
   * names are ignored. Mutually exclusive with `categoryAllowlist`.
   */
  categoryDenylist?: string[]|undefined;
}
