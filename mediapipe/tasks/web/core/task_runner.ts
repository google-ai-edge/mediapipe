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

import {createMediaPipeLib, FileLocator, GraphRunner, WasmMediaPipeConstructor, WasmModule} from '../../../web/graph_runner/graph_runner';
import {SupportImage} from '../../../web/graph_runner/graph_runner_image_lib';
import {SupportModelResourcesGraphService} from '../../../web/graph_runner/register_model_resources_graph_service';

import {WasmFileset} from './wasm_fileset';

// None of the MP Tasks ship bundle assets.
const NO_ASSETS = undefined;

// tslint:disable-next-line:enforce-name-casing
const WasmMediaPipeImageLib =
    SupportModelResourcesGraphService(SupportImage(GraphRunner));

/** Base class for all MediaPipe Tasks. */
export abstract class TaskRunner extends WasmMediaPipeImageLib {
  private processingErrors: Error[] = [];

  /**
   * Creates a new instance of a Mediapipe Task. Determines if SIMD is
   * supported and loads the relevant WASM binary.
   * @return A fully instantiated instance of `T`.
   */
  protected static async createInstance<T extends TaskRunner>(
      type: WasmMediaPipeConstructor<T>, initializeCanvas: boolean,
      fileset: WasmFileset): Promise<T> {
    const fileLocator: FileLocator = {
      locateFile() {
        // The only file loaded with this mechanism is the Wasm binary
        return fileset.wasmBinaryPath.toString();
      }
    };

    if (initializeCanvas) {
      // Fall back to an OffscreenCanvas created by the GraphRunner if
      // OffscreenCanvas is available
      const canvas = typeof OffscreenCanvas === 'undefined' ?
          document.createElement('canvas') :
          undefined;
      return createMediaPipeLib(
          type, fileset.wasmLoaderPath, NO_ASSETS, canvas, fileLocator);
    } else {
      return createMediaPipeLib(
          type, fileset.wasmLoaderPath, NO_ASSETS, /* glCanvas= */ null,
          fileLocator);
    }
  }

  constructor(
      wasmModule: WasmModule,
      glCanvas?: HTMLCanvasElement|OffscreenCanvas|null) {
    super(wasmModule, glCanvas);

    // Disables the automatic render-to-screen code, which allows for pure
    // CPU processing.
    this.setAutoRenderToScreen(false);

    // Enables use of our model resource caching graph service.
    this.registerModelResourcesGraphService();
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
  override setGraph(graphData: Uint8Array, isBinary: boolean): void {
    this.attachErrorListener((code, message) => {
      this.processingErrors.push(new Error(message));
    });
    super.setGraph(graphData, isBinary);
    this.handleErrors();
  }

  /**
   * Forces all queued-up packets to be pushed through the MediaPipe graph as
   * far as possible, performing all processing until no more processing can be
   * done.
   */
  override finishProcessing(): void {
    super.finishProcessing();
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


