# Copyright 2026 The MediaPipe Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""MediaPipe Tasks Retrieval API."""

import mediapipe.tasks.python.retrieval.semantic_retriever
import mediapipe.tasks.python.retrieval.universal_embedder

ChunkingMode = semantic_retriever.ChunkingMode
RetrievalRecord = semantic_retriever.RetrievalRecord
RetrievalResult = semantic_retriever.RetrievalResult
SemanticRetriever = semantic_retriever.SemanticRetriever
SemanticRetrieverOptions = semantic_retriever.SemanticRetrieverOptions
TaskPartKind = semantic_retriever.TaskPartKind
UniversalEmbedder = universal_embedder.UniversalEmbedder
UniversalEmbedderOptions = universal_embedder.UniversalEmbedderOptions
UniversalEmbedderResult = universal_embedder.UniversalEmbedderResult

# Remove unnecessary modules to avoid duplication in API docs.
del mediapipe
del semantic_retriever
del universal_embedder
