import {WasmMediaPipeLib} from './wasm_mediapipe_lib';

/**
 * We extend from a WasmMediaPipeLib constructor. This ensures our mixin has
 * access to the wasmModule, among other things. The `any` type is required for
 * mixin constructors.
 */
// tslint:disable-next-line:no-any
type LibConstructor = new (...args: any[]) => WasmMediaPipeLib;

/**
 * Declarations for Emscripten's WebAssembly Module behavior, so TS compiler
 * doesn't break our JS/C++ bridge.
 */
export declare interface WasmModuleRegisterModelResources {
  _registerModelResourcesGraphService: () => void;
}

/**
 * An implementation of WasmMediaPipeLib that supports registering model
 * resources to a cache, in the form of a GraphService C++-side. We implement as
 * a proper TS mixin, to allow for effective multiple inheritance. Sample usage:
 * `const WasmMediaPipeImageLib = SupportModelResourcesGraphService(
 *     WasmMediaPipeLib);`
 */
// tslint:disable:enforce-name-casing
export function SupportModelResourcesGraphService<TBase extends LibConstructor>(
    Base: TBase) {
  return class extends Base {
    // tslint:enable:enforce-name-casing
    /**
     * Instructs the graph runner to use the model resource caching graph
     * service for both graph expansion/inintialization, as well as for graph
     * run.
     */
    registerModelResourcesGraphService(): void {
      (this.wasmModule as unknown as WasmModuleRegisterModelResources)
          ._registerModelResourcesGraphService();
    }
  };
}
