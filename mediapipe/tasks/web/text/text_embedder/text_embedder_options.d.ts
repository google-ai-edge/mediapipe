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

import {EmbedderOptions} from '../../../../tasks/web/core/embedder_options';
import {TaskRunnerOptions} from '../../../../tasks/web/core/task_runner_options';

/**
 * The type of embedding to generate. This can be used to specify different
 * embedding tasks for models that support it (e.g. Gecko).
 *
 * Supported embedding types are:
 * - 'RETRIEVAL_QUERY': Embeds a query for document retrieval. The input text
 *   will be formatted as "task: search result | query: ${text}".
 * - 'RETRIEVAL_DOCUMENT': Embeds a document for semantic retrieval. The input
 *   text will be formatted as "title: none | text: ${text}".
 * - 'SEMANTIC_SIMILARITY': Embeds text for semantic similarity. The input text
 *   will be formatted as "task: sentence similarity | query: ${text}".
 * - 'CLASSIFICATION': Embeds text for classification. The input text will be
 *   formatted as "task: classification | query: ${text}".
 * - 'QUESTION_ANSWERING': Embeds text for question answering. The input text
 *   will be formatted as "task: question answering | query: ${text}".
 * - 'CLUSTERING': Embeds text for clustering. The input text will be formatted
 *   as "task: clustering | query: ${text}".
 * - 'FACT_CHECKING': Embeds text for fact checking. The input text will be
 *   formatted as "task: fact checking | query: ${text}".
 * - 'CODE_RETRIEVAL': Embeds text for code retrieval. The input text will be
 *   formatted as "task: code retrieval | query: ${text}".
 */
export type TextEmbeddingType =
  | 'RETRIEVAL_QUERY'
  | 'RETRIEVAL_DOCUMENT'
  | 'SEMANTIC_SIMILARITY'
  | 'CLASSIFICATION'
  | 'QUESTION_ANSWERING'
  | 'CLUSTERING'
  | 'FACT_CHECKING'
  | 'CODE_RETRIEVAL';

/**
 * The role of the text to be embedded.
 *
 * Supported roles are:
 * - 'QUERY': The text is a query.
 * - 'DOCUMENT': The text is a document.
 */
export type TextRole = 'QUERY' | 'DOCUMENT';

/**
 * Options for formatting a text embedding request.
 */
export declare interface TextFormatOptions {
  /** The embedding type to generate for the given text. */
  type: TextEmbeddingType;
  /** The title of the text to be embedded. Only used for document embeddings. */
  title?: string;
  /** The role of the text to be embedded. */
  textRole?: TextRole;
}

/** Options to configure the MediaPipe Text Embedder Task */
export declare interface TextEmbedderOptions
  extends EmbedderOptions,
    TaskRunnerOptions {}
