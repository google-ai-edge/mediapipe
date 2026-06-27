/**
 * Copyright 2025 The MediaPipe Authors.
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

/**
 * Optimized model loading utilities for LLM inference.
 */

/**
 * Creates a streaming model loader with proper resource management.
 */
export async function createModelStream(
  modelAssetPath: string,
  signal?: AbortSignal
): Promise<ReadableStreamDefaultReader<Uint8Array>> {
  const response = await fetch(modelAssetPath, { signal });
  
  if (!response.ok) {
    throw new Error(
      `Failed to fetch model: ${modelAssetPath} (${response.status})`
    );
  }
  
  if (!response.body) {
    throw new Error(
      `Failed to fetch model: ${modelAssetPath} (no body)`
    );
  }
  
  return response.body.getReader();
}

/**
 * Model loader with cancellation support.
 */
export class ModelLoader {
  private abortController?: AbortController;

  async loadModel(
    modelAssetPath: string
  ): Promise<ReadableStreamDefaultReader<Uint8Array>> {
    this.cancel();
    this.abortController = new AbortController();
    
    return createModelStream(modelAssetPath, this.abortController.signal);
  }

  cancel(): void {
    this.abortController?.abort();
    this.abortController = undefined;
  }

  isLoading(): boolean {
    return !!this.abortController;
  }
}