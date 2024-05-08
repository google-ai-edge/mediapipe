/**
 * Copyright 2024 The MediaPipe Authors.
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
import {LlmInference} from '../../../../tasks/web/genai/llm_inference/llm_inference';
import {
  FileLocator,
  WasmMediaPipeConstructor,
  WasmModule,
  createMediaPipeLib,
} from '../../../../web/graph_runner/graph_runner';
import {WasmFileReference} from '../../../../web/graph_runner/graph_runner_wasm_file_reference';
// Placeholder for internal dependency on trusted resource url

// The OSS JS API does not support the builder pattern.
// tslint:disable:jspb-use-builder-pattern

declare interface RagWasmModule extends WasmModule {
  HEAPF32: Float32Array;
  HEAPU8: Uint8Array;

  _addStringVectorEntry(vecPtr: number, strPtr: number): void;
  _allocateStringVector(size: number): number;
  _release(pointer: number): void;
  ccall(
    ident: string,
    returnType: string,
    argTypes: string[],
    args: unknown[],
    opts?: {async?: boolean},
  ): number;
  UTF8ToString(encoded: number): string;
  stringToNewUTF8(decoded: string): number;
}

const PROMPT_TEMPLATE = `<start_of_turn>system
You are an assistant for question-answering tasks. You are given facts and you need to answer a question only using the facts provided.
<end_of_turn>
<start_of_turn>context
Here are the facts:
{memory}
<end_of_turn>
<start_of_turn>user
Use the facts to answer questions from the User.
User query:{query}
<end_of_turn>
<start_of_turn>model
`;

type ProgressListener = (partial: string, done: boolean) => void;

/**
 * RAG (Retrieval-Augmented Generation) Pipeline API for MediaPipe.
 *
 * This API is highly experimental and will change.
 */
export class RagPipeline {
  /**
   * Initializes the Wasm runtime and creates a new `RagPipeline` using the
   * provided `LLMInference` task.
   *
   * @export
   * @param wasmFileset A configuration object that provides the location of the
   *     Wasm binary and its loader.
   * @param llmInference The LLM Inference Task to use with this RAG pipeline.
   * @param embeddingModel Either the buffer or url to the embedding model that
   * will be used in the RAG pipeline to embed texts.
   */
  static async createWithEmbeddingModel(
    wasmFileset: WasmFileset,
    llmInference: LlmInference,
    embeddingModel: string | Uint8Array,
  ): Promise<RagPipeline> {
    const fileLocator: FileLocator = {
      locateFile(file): string {
        // We currently only use a single .wasm file and a single .data file (for
        // the tasks that have to load assets). We need to revisit how we
        // initialize the file locator if we ever need to differentiate between
        // diffferent files.
        if (file.endsWith('.wasm')) {
          return wasmFileset.wasmBinaryPath.toString();
        } else if (wasmFileset.assetBinaryPath && file.endsWith('.data')) {
          return wasmFileset.assetBinaryPath.toString();
        }
        return file;
      },
    };

    const ragPipeline = await createMediaPipeLib(
      RagPipeline.bind(
        null,
        llmInference,
      ) as unknown as WasmMediaPipeConstructor<RagPipeline>,
      wasmFileset.wasmLoaderPath,
      wasmFileset.assetLoaderPath,
      /*  glCanvas= */ null,
      fileLocator,
    );

    let wasmFileRef: WasmFileReference;
    if (embeddingModel instanceof Uint8Array) {
      wasmFileRef = WasmFileReference.loadFromArray(
        ragPipeline.ragModule,
        embeddingModel,
      );
    } else {
      wasmFileRef = await WasmFileReference.loadFromUrl(
        ragPipeline.ragModule,
        embeddingModel,
      );
    }
    await ragPipeline.wrapStringPtr(PROMPT_TEMPLATE, (promptStrPtr) =>
      ragPipeline.ragModule.ccall(
        'initializeChain',
        'void',
        ['number', 'number', 'number'],
        [wasmFileRef.offset, wasmFileRef.size, promptStrPtr],
        {async: true},
      ),
    );

    wasmFileRef.free();
    return ragPipeline;
  }

  /** @hideconstructor */
  constructor(
    private readonly llmInference: LlmInference,
    private readonly ragModule: RagWasmModule,
  ) {}

  /**
   * Instructs the RAG pipeline to memorize the records in the array.
   *
   * @export
   * @param data The array of records to be remembered by RAG pipeline.
   */
  recordBatchedMemory(data: string[]) {
    const vecPtr = this.ragModule._allocateStringVector(data.length);
    if (!vecPtr) {
      throw new Error('Unable to allocate new string vector on heap.');
    }
    for (const entry of data) {
      this.wrapStringPtr(entry, (entryStringPtr) => {
        this.ragModule._addStringVectorEntry(vecPtr, entryStringPtr);
      });
    }
    return this.ragModule.ccall(
      'recordBatchedMemory',
      'void',
      ['number'],
      [vecPtr],
      {async: true},
    );
  }

  /**
   * Uses the RAG pipeline to augment the query.
   *
   * @param query The users' query.
   * @param topK The number of top related entries to be accounted in.
   * @return RAG's augmented query.
   */
  private async buildPrompt(query: string, topK = 2): Promise<string> {
    const result = await this.wrapStringPtr(query, (queryStrPtr) =>
      this.ragModule.ccall(
        'invoke',
        'number',
        ['number', 'number'],
        [queryStrPtr, topK],
        {async: true},
      ),
    );
    return this.ragModule.UTF8ToString(result);
  }

  /**
   * Uses RAG to augment the query and run LLM Inference. `topK` defaults to 2.
   *
   * @export
   * @param query The users' query.
   * @return The generated text result.
   */
  generateResponse(query: string): Promise<string>;
  /**
   * Uses RAG to augment the query and run LLM Inference.
   *
   * @export
   * @param query The users' query.
   * @param topK The number of top related entries to be accounted in.
   * @return The generated text result.
   */
  generateResponse(query: string, topK: number): Promise<string>;
  /**
   * Uses RAG to augment the query and run LLM Inference.
   *
   * @export
   * @param query The users' query.
   * @param progressListener A listener that will be triggered when the task has
   *     new partial response generated.
   * @return The generated text result.
   */
  generateResponse(
    query: string,
    progressListener: ProgressListener,
  ): Promise<string>;
  /**
   * Uses RAG to augment the query and run LLM Inference. `topK` defaults to 2.
   *
   * @export
   * @param query The users' query.
   * @param topK The number of top related entries to be accounted in.
   * @param progressListener A listener that will be triggered when the task has
   *     new partial response generated.
   * @return The generated text result.
   */
  generateResponse(
    query: string,
    topK: number,
    progressListener: ProgressListener,
  ): Promise<string>;
  /** @export */
  generateResponse(
    query: string,
    topKOrProgressListener?: number | ProgressListener,
    progressListener?: ProgressListener,
  ): Promise<string> {
    const topK =
      typeof topKOrProgressListener === 'number' ? topKOrProgressListener : 2;
    progressListener =
      typeof topKOrProgressListener === 'function'
        ? topKOrProgressListener
        : progressListener;
    return this.buildPrompt(query, topK).then((prompt) => {
      if (progressListener) {
        return this.llmInference.generateResponse(prompt, progressListener);
      } else {
        return this.llmInference.generateResponse(prompt);
      }
    });
  }

  private wrapStringPtr<T>(
    stringData: string,
    userFunction: (strPtr: number) => T,
  ): T {
    const stringDataPtr = this.ragModule.stringToNewUTF8(stringData);
    const res = userFunction(stringDataPtr);
    this.ragModule._release(stringDataPtr);
    return res;
  }

  /** @export */
  close() {}
}


