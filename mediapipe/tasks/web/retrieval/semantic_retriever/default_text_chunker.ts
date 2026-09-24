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

import {WasmFileset} from '../../../../tasks/web/core/wasm_fileset';
import {
  createMediaPipeLib,
  FileLocator,
  WasmModule,
} from '../../../../web/graph_runner/graph_runner';

import type {ChunkingMode, TextChunker} from './text_chunker';

const CHUNKING_MODE_VALUE: Record<ChunkingMode, number> = {
  'CHARACTER': 0,
  'WORD': 1,
};

const DEFAULT_CHUNK_SIZE = 512;
const DEFAULT_CHUNK_OVERLAP = 100;
const DEFAULT_CHUNKING_MODE: ChunkingMode = 'CHARACTER';

/**
 * Declarations for the native DefaultTextChunker WebAssembly methods.
 */
// tslint:disable:name-casing
/* eslint-disable @typescript-eslint/naming-convention */
export declare interface TextChunkerWasmModule extends WasmModule {
  defaultTextChunker_nativeChunkText(
    text: string,
    chunkSize: number,
    chunkOverlap: number,
    chunkingMode: number,
  ): string[];
}

/**
 * A default text chunking implementation using native C++ Wasm methods.
 */
export class DefaultTextChunker implements TextChunker {
  private readonly wasmModule: TextChunkerWasmModule;
  private chunkSize = DEFAULT_CHUNK_SIZE;
  private chunkOverlap = DEFAULT_CHUNK_OVERLAP;
  private mode: ChunkingMode = DEFAULT_CHUNKING_MODE;

  /**
   * Initializes the Wasm runtime and creates a new DefaultTextChunker instance.
   * @export
   */
  static async create(
    wasmFileset: WasmFileset,
    chunkSize = DEFAULT_CHUNK_SIZE,
    chunkOverlap = DEFAULT_CHUNK_OVERLAP,
    mode: ChunkingMode = DEFAULT_CHUNKING_MODE,
  ): Promise<DefaultTextChunker> {
    DefaultTextChunker.validateOptions(chunkSize, chunkOverlap);

    const fileLocator: FileLocator = {
      locateFile(file: string): string {
        if (file.endsWith('.wasm')) {
          return wasmFileset.wasmBinaryPath.toString();
        }
        return file;
      },
    };

    const instance = await createMediaPipeLib(
      DefaultTextChunker,
      wasmFileset.wasmLoaderPath,
      wasmFileset.assetLoaderPath,
      /* glCanvas= */ null,
      fileLocator,
    );
    instance.configure(chunkSize, chunkOverlap, mode);
    return instance;
  }

  private static validateOptions(
    chunkSize: number,
    chunkOverlap: number,
  ): void {
    if (chunkSize <= 0) {
      throw new Error('Chunk size must be positive.');
    }
    if (chunkOverlap < 0 || chunkOverlap >= chunkSize) {
      throw new Error(
        'Chunk overlap must be non-negative and less than chunk size.',
      );
    }
  }

  /**
   * Creates a DefaultTextChunker from an already initialized Wasm module.
   * @export
   */
  static createFromModule(
    wasmModule: WasmModule,
    chunkSize = DEFAULT_CHUNK_SIZE,
    chunkOverlap = DEFAULT_CHUNK_OVERLAP,
    mode: ChunkingMode = DEFAULT_CHUNKING_MODE,
  ): DefaultTextChunker {
    DefaultTextChunker.validateOptions(chunkSize, chunkOverlap);
    const instance = new DefaultTextChunker(wasmModule);
    instance.configure(chunkSize, chunkOverlap, mode);
    return instance;
  }

  /** @hideconstructor */
  constructor(wasmModule: WasmModule) {
    this.wasmModule = wasmModule as TextChunkerWasmModule;
  }

  /**
   * Configures chunking parameters on this instance.
   */
  private configure(
    chunkSize: number,
    chunkOverlap: number,
    mode: ChunkingMode = DEFAULT_CHUNKING_MODE,
  ): void {
    DefaultTextChunker.validateOptions(chunkSize, chunkOverlap);
    this.chunkSize = chunkSize;
    this.chunkOverlap = chunkOverlap;
    this.mode = mode;
  }

  chunk(text: string): string[] {
    if (!text || text.trim().length === 0) {
      return [];
    }
    if (
      !this.wasmModule ||
      typeof this.wasmModule.defaultTextChunker_nativeChunkText !== 'function'
    ) {
      throw new Error('DefaultTextChunker Wasm module is not initialized.');
    }
    const modeValue = CHUNKING_MODE_VALUE[this.mode] ?? 0;
    const chunks = this.wasmModule.defaultTextChunker_nativeChunkText(
      text,
      this.chunkSize,
      this.chunkOverlap,
      modeValue,
    );
    return chunks ? Array.from(chunks) : [];
  }
}


