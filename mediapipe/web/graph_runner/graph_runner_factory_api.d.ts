// Placeholder for internal dependency on trusted resource url

import {GraphRunnerApi} from './graph_runner_api';
import {WasmModule} from './wasm_module';

/**
 * Simple interface for allowing users to set the directory where internal
 * wasm-loading and asset-loading code looks (e.g. for .wasm and .data file
 * locations).
 */
export declare interface FileLocator {
  locateFile: (filename: string) => string;
  mainScriptUrlOrBlob?: string;
}

/**
 * Internal type of constructors used for initializing GraphRunner and
 * subclasses.
 */
export type WasmMediaPipeConstructor<LibType> =
    (new (
         module: WasmModule, canvas?: HTMLCanvasElement|OffscreenCanvas|null) =>
         LibType);

/**
 * Global function interface to initialize Wasm blob and load runtime assets for
 *     a specialized MediaPipe library. This allows us to create a requested
 *     subclass inheriting from GraphRunner. Standard implementation is
 *     `createMediaPipeLib<LibType>`.
 * @param constructorFcn The name of the class to instantiate via "new".
 * @param wasmLoaderScript Url for the wasm-runner script; produced by the build
 *     process.
 * @param assetLoaderScript Url for the asset-loading script; produced by the
 *     build process.
 * @param fileLocator A function to override the file locations for assets
 *     loaded by the MediaPipe library.
 * @return promise A promise which will resolve when initialization has
 *     completed successfully.
 */
export interface CreateMediaPipeLibApi {
  <LibType>(
      constructorFcn: WasmMediaPipeConstructor<LibType>,
      wasmLoaderScript?: string|null,
      assetLoaderScript?: string|null,
      glCanvas?: HTMLCanvasElement|OffscreenCanvas|null,
      fileLocator?: FileLocator): Promise<LibType>;
}

/**
 * Global function interface to initialize Wasm blob and load runtime assets for
 *      a generic MediaPipe library. Standard implementation is
 *      `createGraphRunner`.
 * @param wasmLoaderScript Url for the wasm-runner script; produced by the build
 *     process.
 * @param assetLoaderScript Url for the asset-loading script; produced by the
 *     build process.
 * @param fileLocator A function to override the file locations for assets
 *     loaded by the MediaPipe library.
 * @return promise A promise which will resolve when initialization has
 *     completed successfully.
 */
export interface CreateGraphRunnerApi {
  (wasmLoaderScript?: string,
   assetLoaderScript?: string,
   glCanvas?: HTMLCanvasElement|OffscreenCanvas|null,
   fileLocator?: FileLocator): Promise<GraphRunnerApi>;
}
