import {GraphRunner} from '../../web/graph_runner/graph_runner';

/**
 * We extend from a GraphRunner constructor. This ensures our mixin has
 * access to the wasmModule, among other things. The `any` type is required for
 * mixin constructors.
 */
// tslint:disable-next-line:no-any
type LibConstructor = new (...args: any[]) => GraphRunner;

/**
 * Declarations for Emscripten's WebAssembly Module behavior, so TS compiler
 * doesn't break our JS/C++ bridge.
 */
export declare interface WasmWebGpuModule {
  preinitializedWebGPUDevice: GPUDevice;
}

// TODO: Remove once adapter info is piped through
// Emscripten.
/**
 * This interface is used to pipe adapter info through WebGPU device object to
 * our inference engine.
 */
export declare interface GPUDeviceWithAdapterInfo extends GPUDevice {
  adapterInfo: GPUAdapterInfo;
}

/**
 * Internal-only interface for calling `closeGraph` in an ASYNCIFY-friendly
 * manner.
 */
export declare interface WasmAsyncCloseModule {
  ccall: (name: string, type: string, inParams: unknown, outParams: unknown,
      options: unknown) => Promise<void>;
}

/**
 * An implementation of GraphRunner that integrates WebGpu functionality.
 * Example usage:
 * `const MediaPipeLib = SupportWebGpu(GraphRunner);`
 */
// tslint:disable-next-line:enforce-name-casing
export function SupportWebGpu<TBase extends LibConstructor>(Base: TBase) {
  return class WebGpuSupportedGraphRunner extends Base {
    /*
     * Requests and returns a GPUDevice.
     * @param deviceDescriptor The deviceDescriptor to request GPUDevice.
     * @param adapter The adapter to request GPUDevice.
     */
    static async requestWebGpuDevice(
        deviceDescriptor?: GPUDeviceDescriptor,
        adapter?: GPUAdapter): Promise<GPUDevice> {
      if (!adapter) {
        adapter = await WebGpuSupportedGraphRunner.requestWebGpuAdapter();
      }
      let device: GPUDevice;
      const supportedFeatures: GPUFeatureName[] = [];
      for (const feature of deviceDescriptor?.requiredFeatures ?? []) {
        if (adapter.features.has(feature)) {
          supportedFeatures.push(feature);
        } else {
          console.warn(`WebGPU feature ${feature} is not supported.`);
        }
      }
      const updatedDescriptor: GPUDeviceDescriptor = {
        ...deviceDescriptor,
        requiredFeatures: supportedFeatures
      };
      try {
        device = await adapter.requestDevice(updatedDescriptor);
      } catch (e: unknown) {
        console.error(
            'Unable to initialize WebGPU with the requested features.');
        // Rethrow original error.
        throw e;
      }

      // TODO: Remove once adapter info is piped through
      // Emscripten.
      // Our inference engines can utilize the adapter info to optimize WebGPU
      // shader performance. Therefore, we attempt to attach that information to
      // our internal GPUDevice reference.
      const adapterInfo = await adapter.requestAdapterInfo();
      (device as unknown as GPUDeviceWithAdapterInfo).adapterInfo = adapterInfo;

      return device;
    }

    /*
     * Requests and returns a GPUAdapter.
     * @param adapterDescriptor The adapterDescriptor to request GPUAdapter.
     */
    static async requestWebGpuAdapter(
        adapterDescriptor?: GPURequestAdapterOptions): Promise<GPUAdapter> {
      const adapter = await navigator.gpu.requestAdapter(adapterDescriptor);
      if (!adapter) {
        throw new Error(
            'Unable to request adapter from navigator.gpu; Ensure WebGPU is enabled.');
      }
      return adapter;
    }

    /*
     * Initializes the GraphRunner for WebGPU support, given the target canvas
     * and GPUDevice which it should use for internal WebGPU commands. Note that
     * currently when an OffscreenCanvas is used, render-to-display
     * functionality will not be available.
     * @param device The device to be used.
     * @param canvas The on- or offscreen canvas. If not provided, an offscreen
     * canvas will be created.
     */
    initializeForWebGpu(
        device: GPUDevice, canvas?: HTMLCanvasElement|OffscreenCanvas) {
      if (!canvas) {
        canvas = new OffscreenCanvas(1, 1);
      } else if (
          typeof HTMLCanvasElement !== 'undefined' &&
          canvas instanceof HTMLCanvasElement) {
        // TODO b/327324051 - Stop using a hard-coded `canvas_webgpu` selector.
        canvas.id = 'canvas_webgpu';  // id used as default for WebGPU code
      }
      const context = canvas.getContext('webgpu') as GPUCanvasContext;
      context.configure({
        device,
        format: navigator.gpu.getPreferredCanvasFormat(),
      });
      const wasmModule = this.wasmModule as unknown as WasmWebGpuModule;
      wasmModule.preinitializedWebGPUDevice = device;
    }

    /**
     * Special variant of `closeGraph` for use with WebGPU demos built with
     * ASYNCIFY=1, to allow us to ensure proper synchronization.
     * TODO: Remove when we have a better solution for async calls.
     */
    closeGraphAsync(): Promise<void> {
      return (this.wasmModule as unknown as WasmAsyncCloseModule).ccall(
          "closeGraph", "void", [], [], {async: true});
    }
  };
}
