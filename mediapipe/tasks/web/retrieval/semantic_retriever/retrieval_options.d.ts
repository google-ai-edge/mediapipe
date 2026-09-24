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

/** Options for configuring retrieval queries in SemanticRetriever. */
export declare interface RetrievalOptions {
  /** The maximum number of results to return. Defaults to 5. */
  readonly limit?: number;

  /** Key-value metadata filter to restrict matching records. */
  readonly metadataFilter?: Record<string, string>;
}
