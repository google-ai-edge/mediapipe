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

import {Category} from '../../../../tasks/web/components/containers/category';

/** Classification results for a given classifier head. */
export declare interface Classifications {
  /**
   * The array of predicted categories, usually sorted by descending scores,
   * e.g., from high to low probability.
   */
  categories: Category[];

  /**
   * The index of the classifier head these categories refer to. This is
   * useful for multi-head models.
   */
  headIndex: number;

  /**
   * The name of the classifier head, which is the corresponding tensor
   * metadata name. Defaults to an empty string if there is no such metadata.
   */
  headName: string;
}

/** Classification results of a model. */
export declare interface ClassificationResult {
  /** The classification results for each head of the model. */
  classifications: Classifications[];

  /**
   * The optional timestamp (in milliseconds) of the start of the chunk of data
   * corresponding to these results.
   *
   * This is only used for classification on time series (e.g. audio
   * classification). In these use cases, the amount of data to process might
   * exceed the maximum size that the model can process: to solve this, the
   * input data is split into multiple chunks starting at different timestamps.
   */
  timestampMs?: number;
}
