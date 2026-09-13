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

// Placeholder for internal dependency on trusted resource url

/** Options to configure MediaPipe model loading and processing. */
export declare interface BaseOptions {
  /**
   * The model path to the model asset file. Only one of `modelAssetPath` or
   * `modelAssetBuffer` can be set.
   */
  modelAssetPath?: string | undefined;

  /**
   * A buffer or stream reader containing the model asset. Only one of
   * `modelAssetPath` or `modelAssetBuffer` can be set.
   */
  modelAssetBuffer?: Uint8Array | ReadableStreamDefaultReader | undefined;

  /** Overrides the default backend to use for the provided model. */
  delegate?: 'CPU' | 'GPU' | undefined;
}

/**
 * Which download `onLoadingProgress` is reporting.
 *
 * - `'wasm'`: the task's WebAssembly binary
 * - `'asset'`: the optional Emscripten `.data` package (tasks that ship extra
 *   assets)
 * - `'model'`: the TFLite / task model from `baseOptions.modelAssetPath`
 */
export type LoadingResourceType = 'wasm'|'asset'|'model';

/** Progress of a Wasm, asset, or model download. Use this to drive a loading bar. */
export declare interface LoadingProgressEvent {
  /** The resource currently being downloaded. */
  type: LoadingResourceType;
  /** Bytes received so far. */
  loaded: number;
  /**
   * Total size in bytes when the server sends `Content-Length`.
   * `0` if the size is not known (show an indeterminate bar).
   */
  total: number;
}

/** Options to configure MediaPipe Tasks in general. */
export declare interface TaskRunnerOptions {
  /** Options to configure the loading of the model assets. */
  baseOptions?: BaseOptions;

  /**
   * Called as the Wasm binary, optional `.data` assets, and model file
   * download. Apps can use `loaded / total` (when `total > 0`) to show a
   * loading bar. Downloads of ~10MB each otherwise leave the UI stalled with
   * no feedback.
   *
   * Omitted by default. Pass `undefined` in a later `setOptions()` call to
   * stop receiving events.
   */
  onLoadingProgress?: ((event: LoadingProgressEvent) => void)|undefined;
}
