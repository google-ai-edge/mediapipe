/**
 * @fileoverview Shared WebGPU device setup for LiteRT-LM WebAssembly tasks.
 *
 * The Wasm code does not request a device itself: it uses the one it finds in
 * `preinitializedWebGPUDevice` on the Emscripten module.
 *
 * @license
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

import {WasmModule} from '../../../web/graph_runner/graph_runner';

/** The module property that the Wasm code reads its WebGPU device from. */
const DEVICE_PROPERTY = 'preinitializedWebGPUDevice';

/** Features that make the models faster, requested where available. */
const OPTIONAL_FEATURES: readonly GPUFeatureName[] = [
  'shader-f16',
  'subgroups',
];

/**
 * Limits that are raised to what the adapter supports: the models need larger
 * buffers than the WebGPU defaults allow.
 */
const RAISED_LIMITS = [
  'maxBufferSize',
  'maxStorageBufferBindingSize',
  'maxStorageBuffersPerShaderStage',
  'maxComputeWorkgroupStorageSize',
  'maxTextureDimension2D',
];

function deviceHolders(wasmModule: WasmModule): Array<Record<string, unknown>> {
  // Emscripten reads `preinitializedWebGPUDevice` outside the typed
  // `WasmModule` interface (and on `globalThis.Module` in legacy loaders).
  const holders = [wasmModule as unknown as Record<string, unknown>];
  const globalModule = (globalThis as unknown as Record<string, unknown>)[
    'Module'
  ];
  if (typeof globalModule === 'object' && globalModule != null) {
    holders.push(globalModule as Record<string, unknown>);
  }
  return holders;
}

/**
 * Legacy Chromium `GPUAdapter` extension exposing `isFallbackAdapter` outside
 * standard WebGPU lib types (browser DOM interop; not a proto/JSPB payload).
 */
declare interface LegacyGpuAdapter {
  readonly isFallbackAdapter?: boolean;
}

/**
 * Emscripten Dawn shim extension on `GPUDevice` (browser/Wasm interop; not a
 * proto/JSPB payload).
 */
declare interface GpuDeviceWithAdapterInfo {
  adapterInfo?: GPUAdapterInfo;
}

function isSoftwareAdapter(adapter: GPUAdapter): boolean {
  const withFallbackFlag = adapter as unknown as LegacyGpuAdapter;
  if (withFallbackFlag.isFallbackAdapter) {
    return true;
  }
  // `adapter.info` can be undefined on older WebGPU implementations.
  const info = adapter.info as GPUAdapterInfo | undefined;
  return /swiftshader|llvmpipe|software|lavapipe/i.test(
    `${info?.vendor ?? ''} ${info?.architecture ?? ''} ${info?.description ?? ''}`,
  );
}

/**
 * Gives the Wasm code in `wasmModule` a WebGPU device on a hardware GPU,
 * unless it already has one. Returns false when WebGPU or a hardware adapter is
 * unavailable.
 */
export async function provideWebGpuDevice(
  wasmModule: WasmModule,
  gpu: GPU | null | undefined = typeof navigator !== 'undefined'
    ? navigator.gpu
    : undefined,
): Promise<boolean> {
  const holders = deviceHolders(wasmModule);
  // Read any previously attached `GPUDevice` from the Emscripten module slot.
  let device = holders[0][DEVICE_PROPERTY] as GPUDevice | undefined;
  if (!device) {
    if (!gpu) {
      return false;
    }
    try {
      const adapter = await gpu.requestAdapter({
        powerPreference: 'high-performance',
      });
      if (!adapter || isSoftwareAdapter(adapter)) {
        return false;
      }
      const requiredLimits: Record<string, number> = {};
      // Index `GPUSupportedLimits` dynamically by the `RAISED_LIMITS` keys.
      const adapterLimits = adapter.limits as unknown as Record<
        string,
        number | undefined
      >;
      for (const limit of RAISED_LIMITS) {
        const value = adapterLimits[limit];
        if (typeof value === 'number') {
          requiredLimits[limit] = value;
        }
      }
      device = await adapter.requestDevice({
        requiredFeatures: OPTIONAL_FEATURES.filter((feature) =>
          adapter.features.has(feature),
        ),
        requiredLimits,
      });
      const deviceWithInfo = device as unknown as GpuDeviceWithAdapterInfo;
      if (!deviceWithInfo.adapterInfo && adapter.info) {
        try {
          Object.defineProperty(device, 'adapterInfo', {
            value: adapter.info,
            writable: true,
            configurable: true,
          });
        } catch {
          // Ignore if `adapterInfo` is non-configurable on this browser's
          // `GPUDevice` prototype.
        }
      }
    } catch (e) {
      console.warn('WebGPU device initialization failed:', e);
      return false;
    }
  }
  for (const holder of holders) {
    holder[DEVICE_PROPERTY] = device;
  }
  return true;
}

/** Clears `preinitializedWebGPUDevice` on `wasmModule` when running CPU-only. */
export function removeWebGpuDevice(wasmModule: WasmModule): void {
  for (const holder of deviceHolders(wasmModule)) {
    holder[DEVICE_PROPERTY] = undefined;
  }
}
