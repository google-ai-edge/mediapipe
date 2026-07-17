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

import {GraphRunner} from './graph_runner';

// tslint:disable-next-line:no-any
type LibConstructor = new (...args: any[]) => GraphRunner;

/**
 * Declarations for Emscripten's WebAssembly Module behavior for Logging.
 */
export declare interface WasmLoggingModule {
  _decodeBase64?: (ptr: number) => string;
  _mediapipeLoggerGetEncodedApiKey?: () => number;
}

/**
 * An implementation of GraphRunner that supports MediaPipe API Key retrieval.
 */
// tslint:disable-next-line:enforce-name-casing
export function SupportLogging<TBase extends LibConstructor>(Base: TBase) {
  return class extends Base {
    get wasmLoggingModule(): WasmLoggingModule {
      return this.wasmModule as unknown as WasmLoggingModule;
    }

    /** Fetches the MediaPippe API key. */
    getMediapipeApiKey(): string | undefined {
      if (
        typeof this.wasmLoggingModule._mediapipeLoggerGetEncodedApiKey ===
        'function'
      ) {
        const ptr = this.wasmLoggingModule._mediapipeLoggerGetEncodedApiKey();
        return this.wasmLoggingModule._decodeBase64!(ptr);
      }
      return undefined;
    }
  };
}
