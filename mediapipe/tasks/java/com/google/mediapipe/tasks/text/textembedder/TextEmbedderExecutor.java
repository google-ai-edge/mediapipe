// Copyright 2026 The MediaPipe Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package com.google.mediapipe.tasks.text.textembedder;

import com.google.mediapipe.tasks.text.textembedder.TextEmbedder.TextFormatContext;

/** Executor interface to isolate and branch Text Embedder executions. */
public interface TextEmbedderExecutor extends AutoCloseable {
  TextEmbedderResult embed(String inputText);

  TextEmbedderResult embed(String inputText, TextFormatContext formatContext);

  @Override
  void close();
}
