/* Copyright 2026 The MediaPipe Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#ifndef __EMSCRIPTEN__
#error "This is web-only code, but was built for a non-web target platform."
#endif  // __EMSCRIPTEN__

#include <emscripten.h>
#include <emscripten/bind.h>
#include <emscripten/val.h>

#include <cstddef>
#include <string>
#include <vector>

#include "mediapipe/tasks/cc/retrieval/semantic_retriever/text_chunker.h"

namespace mediapipe::tasks::retrieval {
namespace {

enum WasmChunkingMode {
  CHUNK_MODE_CHARACTER = 0,
  CHUNK_MODE_WORD = 1,
};

emscripten::val NativeChunkText(const std::string& text, int chunk_size,
                                int chunk_overlap, int chunking_mode) {
  ChunkingMode mode = chunking_mode == CHUNK_MODE_WORD
                          ? ChunkingMode::kWord
                          : ChunkingMode::kCharacter;
  DefaultTextChunker chunker(chunk_size, chunk_overlap, mode);
  auto chunks_or = chunker.Chunk(text);
  emscripten::val js_chunks = emscripten::val::array();
  if (!chunks_or.ok()) {
    return js_chunks;
  }
  const auto& chunks = chunks_or.value();
  for (size_t i = 0; i < chunks.size(); ++i) {
    js_chunks.set(i, chunks[i]);
  }
  return js_chunks;
}

}  // namespace

EMSCRIPTEN_BINDINGS(text_chunker_api) {
  emscripten::function("defaultTextChunker_nativeChunkText", &NativeChunkText);
}

}  // namespace mediapipe::tasks::retrieval
