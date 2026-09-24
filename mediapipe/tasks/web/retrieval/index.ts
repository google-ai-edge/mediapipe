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

import {FilesetResolver as FilesetResolverImpl} from '../../../tasks/web/core/fileset_resolver';
import {
  DefaultTextChunker as DefaultTextChunkerImpl,
  MemoryVectorStore as MemoryVectorStoreImpl,
  SemanticRetrieverComponents as SemanticRetrieverComponentsImpl,
  SemanticRetriever as SemanticRetrieverImpl,
} from '../../../tasks/web/retrieval/semantic_retriever/semantic_retriever';
import {UniversalEmbedder as UniversalEmbedderImpl} from '../../../tasks/web/retrieval/universal_embedder/universal_embedder';

// tslint:disable:enforce-comments-on-exported-symbols

// Declare and export the variables inline so that Rollup in OSS
// explicitly retains the bindings and avoids dead-code elimination bugs.
export const DefaultTextChunker = DefaultTextChunkerImpl;
export const FilesetResolver = FilesetResolverImpl;
export const MemoryVectorStore = MemoryVectorStoreImpl;
export const SemanticRetriever = SemanticRetrieverImpl;
export const SemanticRetrieverComponents = SemanticRetrieverComponentsImpl;
export const UniversalEmbedder = UniversalEmbedderImpl;
