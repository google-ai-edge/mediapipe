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

import {ContentPart} from '../../../../tasks/web/core/embedding_provider';

/** Represents a retrieved document or multimedia chunk. */
export declare interface RetrievalResult {
  /** The unique identifier of the retrieved record. */
  readonly id: string;

  /** The list of multi-modal parts in the retrieved record. */
  readonly content: readonly ContentPart[];

  /** The metadata associated with the retrieved record. */
  readonly metadata: Record<string, string>;

  /** The similarity/relevance score of this retrieved result. */
  readonly score: number;
}
