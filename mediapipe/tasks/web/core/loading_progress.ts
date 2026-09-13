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

/**
 * Reads a fetch Response into an ArrayBuffer, invoking `onProgress` as bytes
 * arrive when `response.body` is a ReadableStream.
 *
 * If the body is unavailable (older fetch mocks / environments), falls back to
 * `arrayBuffer()` and reports a single completed event.
 */
export async function readResponseArrayBufferWithProgress(
  response: Response,
  onProgress?: (loaded: number, total: number) => void,
): Promise<ArrayBuffer> {
  const total = Number(response.headers?.get?.('content-length')) || 0;

  if (!onProgress || !response.body) {
    const buffer = await response.arrayBuffer();
    onProgress?.(buffer.byteLength, total || buffer.byteLength);
    return buffer;
  }

  const reader = response.body.getReader();
  const chunks: Uint8Array[] = [];
  let loaded = 0;

  while (true) {
    const {done, value} = await reader.read();
    if (done) {
      break;
    }
    chunks.push(value);
    loaded += value.byteLength;
    onProgress(loaded, total);
  }

  const result = new Uint8Array(loaded);
  let offset = 0;
  for (const chunk of chunks) {
    result.set(chunk, offset);
    offset += chunk.byteLength;
  }
  return result.buffer;
}

/**
 * Fetches `url` as an ArrayBuffer. Throws if the response is not OK.
 *
 * @param url The resource to download.
 * @param onProgress Optional byte-progress callback.
 * @param errorLabel Prefix used in the thrown Error, e.g. `model: foo.tflite`.
 */
export async function fetchArrayBufferWithProgress(
  url: string,
  onProgress?: (loaded: number, total: number) => void,
  errorLabel?: string,
): Promise<ArrayBuffer> {
  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(
      `Failed to fetch ${errorLabel ?? url} (${response.status})`,
    );
  }
  return readResponseArrayBufferWithProgress(response, onProgress);
}
