/**
 * Copyright 2022 The MediaPipe Authors. All Rights Reserved.
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

import {BaseOptions as BaseOptionsProto} from '../../../tasks/cc/core/proto/base_options_pb';
import {convertBaseOptionsToProto} from '../../../tasks/web/components/processors/base_options';
import {TaskRunnerOptions} from '../../../tasks/web/core/task_runner_options';
import {createMediaPipeLib, FileLocator, GraphRunner, WasmMediaPipeConstructor, WasmModule} from '../../../web/graph_runner/graph_runner';
import {SupportImage} from '../../../web/graph_runner/graph_runner_image_lib';
import {SupportModelResourcesGraphService} from '../../../web/graph_runner/register_model_resources_graph_service';

import {WasmFileset} from './wasm_fileset';

// None of the MP Tasks ship bundle assets.
const NO_ASSETS = undefined;

// tslint:disable-next-line:enforce-name-casing
const GraphRunnerImageLibType =
    SupportModelResourcesGraphService(SupportImage(GraphRunner));
/** An implementation of the GraphRunner that supports image operations */
export class GraphRunnerImageLib extends GraphRunnerImageLibType {}

/**
 * Creates a new instance of a Mediapipe Task. Determines if SIMD is
 * supported and loads the relevant WASM binary.
 * @return A fully instantiated instance of `T`.
 */
export async function createTaskRunner<T extends TaskRunner>(
    type: WasmMediaPipeConstructor<T>, initializeCanvas: boolean,
    fileset: WasmFileset, options: TaskRunnerOptions): Promise<T> {
  const fileLocator: FileLocator = {
    locateFile() {
      // The only file loaded with this mechanism is the Wasm binary
      return fileset.wasmBinaryPath.toString();
    }
  };

  // Initialize a canvas if requested. If OffscreenCanvas is availble, we
  // let the graph runner initialize it by passing `undefined`.
  const canvas = initializeCanvas ? (typeof OffscreenCanvas === 'undefined' ?
                                         document.createElement('canvas') :
                                         undefined) :
                                    null;
  const instance = await createMediaPipeLib(
      type, fileset.wasmLoaderPath, NO_ASSETS, canvas, fileLocator);
  await instance.setOptions(options);
  return instance;
}

/** Base class for all MediaPipe Tasks. */
export abstract class TaskRunner {
  protected abstract baseOptions: BaseOptionsProto;
  protected graphRunner: GraphRunnerImageLib;
  private processingErrors: Error[] = [];

  /**
   * Creates a new instance of a Mediapipe Task. Determines if SIMD is
   * supported and loads the relevant WASM binary.
   * @return A fully instantiated instance of `T`.
   */
  protected static async createInstance<T extends TaskRunner>(
      type: WasmMediaPipeConstructor<T>, initializeCanvas: boolean,
      fileset: WasmFileset, options: TaskRunnerOptions): Promise<T> {
    return createTaskRunner(type, initializeCanvas, fileset, options);
  }

  /** @hideconstructor protected */
  constructor(
      wasmModule: WasmModule, glCanvas?: HTMLCanvasElement|OffscreenCanvas|null,
      graphRunner?: GraphRunnerImageLib) {
    this.graphRunner =
        graphRunner ?? new GraphRunnerImageLib(wasmModule, glCanvas);

    // Disables the automatic render-to-screen code, which allows for pure
    // CPU processing.
    this.graphRunner.setAutoRenderToScreen(false);

    // Enables use of our model resource caching graph service.
    this.graphRunner.registerModelResourcesGraphService();
  }

  /** Configures the shared options of a MediaPipe Task. */
  async setOptions(options: TaskRunnerOptions): Promise<void> {
    if (options.baseOptions) {
      this.baseOptions = await convertBaseOptionsToProto(
          options.baseOptions, this.baseOptions);
    }
  }

  /**
   * Takes the raw data from a MediaPipe graph, and passes it to C++ to be run
   * over the video stream. Will replace the previously running MediaPipe graph,
   * if there is one.
   * @param graphData The raw MediaPipe graph data, either in binary
   *     protobuffer format (.binarypb), or else in raw text format (.pbtxt or
   *     .textproto).
   * @param isBinary This should be set to true if the graph is in
   *     binary format, and false if it is in human-readable text format.
   */
  protected setGraph(graphData: Uint8Array, isBinary: boolean): void {
    this.graphRunner.attachErrorListener((code, message) => {
      this.processingErrors.push(new Error(message));
    });
    this.graphRunner.setGraph(graphData, isBinary);
    this.handleErrors();
  }

  /**
   * Forces all queued-up packets to be pushed through the MediaPipe graph as
   * far as possible, performing all processing until no more processing can be
   * done.
   */
  protected finishProcessing(): void {
    this.graphRunner.finishProcessing();
    this.handleErrors();
  }

  /** Throws the error from the error listener if an error was raised. */
  private handleErrors() {
    const errorCount = this.processingErrors.length;
    if (errorCount === 1) {
      // Re-throw error to get a more meaningful stacktrace
      throw new Error(this.processingErrors[0].message);
    } else if (errorCount > 1) {
      throw new Error(
          'Encountered multiple errors: ' +
          this.processingErrors.map(e => e.message).join(', '));
    }
    this.processingErrors = [];
  }
}


