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

import {WasmModule} from '../../../web/graph_runner/graph_runner';
import 'jasmine';
import {provideWebGpuDevice, removeWebGpuDevice} from './webgpu_device';

interface FakeAdapter {
  adapter: GPUAdapter;
  device: Record<string, unknown>;
  requestDevice: jasmine.Spy<GPUAdapter['requestDevice']>;
}

function fakeAdapter(
  overrides: {
    features?: string[];
    limits?: Record<string, number>;
    info?: Partial<GPUAdapterInfo>;
    isFallbackAdapter?: boolean;
  } = {},
): FakeAdapter {
  const device: Record<string, unknown> = {};
  const requestDevice = jasmine
    .createSpy<GPUAdapter['requestDevice']>('requestDevice')
    .and.resolveTo(device as unknown as GPUDevice);
  const adapter = {
    features: new Set<string>(overrides.features ?? []),
    limits: overrides.limits ?? {},
    info: {
      vendor: 'acme',
      architecture: 'rocket',
      description: 'Acme Rocket GPU',
      ...overrides.info,
    },
    isFallbackAdapter: overrides.isFallbackAdapter ?? false,
    requestDevice,
  } as unknown as GPUAdapter;
  return {adapter, device, requestDevice};
}

function fakeGpu(adapter: GPUAdapter | null): GPU & {
  requestAdapter: jasmine.Spy<GPU['requestAdapter']>;
} {
  const requestAdapter = jasmine
    .createSpy<GPU['requestAdapter']>('requestAdapter')
    .and.resolveTo(adapter);
  return {
    requestAdapter,
  } as unknown as GPU & {requestAdapter: jasmine.Spy<GPU['requestAdapter']>};
}

describe('provideWebGpuDevice', () => {
  let module: Record<string, unknown>;
  let wasmModule: WasmModule;

  beforeEach(() => {
    module = {};
    wasmModule = module as unknown as WasmModule;
  });

  afterEach(() => {
    delete (globalThis as unknown as Record<string, unknown>)['Module'];
  });

  it('provides a device with raised limits and optional features on hardware', async () => {
    const globalHolder: Record<string, unknown> = {};
    (globalThis as unknown as Record<string, unknown>)['Module'] = globalHolder;
    const {adapter, device, requestDevice} = fakeAdapter({
      features: ['shader-f16'],
      limits: {'maxBufferSize': 1073741824},
    });
    const gpu = fakeGpu(adapter);

    expect(await provideWebGpuDevice(wasmModule, gpu)).toBeTrue();
    expect(gpu.requestAdapter).toHaveBeenCalledOnceWith({
      powerPreference: 'high-performance',
    });
    expect(requestDevice).toHaveBeenCalledOnceWith({
      requiredFeatures: ['shader-f16'],
      requiredLimits: {'maxBufferSize': 1073741824},
    });
    expect(module['preinitializedWebGPUDevice']).toBe(device);
    expect(globalHolder['preinitializedWebGPUDevice']).toBe(device);
    expect(device['adapterInfo']).toEqual(
      jasmine.objectContaining({vendor: 'acme'}),
    );
  });

  it('reuses an existing device without requesting a new one', async () => {
    const existingDevice: Record<string, unknown> = {};
    module['preinitializedWebGPUDevice'] = existingDevice;
    const gpu = fakeGpu(fakeAdapter().adapter);

    expect(await provideWebGpuDevice(wasmModule, gpu)).toBeTrue();
    expect(gpu.requestAdapter).not.toHaveBeenCalled();
    expect(module['preinitializedWebGPUDevice']).toBe(existingDevice);
  });

  it('returns false without warning when WebGPU is unavailable', async () => {
    const warnSpy = spyOn(console, 'warn');

    expect(await provideWebGpuDevice(wasmModule, null)).toBeFalse();
    expect(warnSpy).not.toHaveBeenCalled();
  });

  it('returns false without warning when requestAdapter returns null', async () => {
    const warnSpy = spyOn(console, 'warn');

    expect(await provideWebGpuDevice(wasmModule, fakeGpu(null))).toBeFalse();
    expect(warnSpy).not.toHaveBeenCalled();
  });

  it('returns false and warns when requestAdapter rejects', async () => {
    const warnSpy = spyOn(console, 'warn');
    const rejectingGpu = fakeGpu(null);
    rejectingGpu.requestAdapter.and.rejectWith(new Error('WebGPU blocked'));

    expect(await provideWebGpuDevice(wasmModule, rejectingGpu)).toBeFalse();
    expect(warnSpy).toHaveBeenCalledTimes(1);
  });

  it('returns false and warns when requestDevice rejects', async () => {
    const warnSpy = spyOn(console, 'warn');
    const {adapter, requestDevice} = fakeAdapter();
    requestDevice.and.rejectWith(new Error('Limits unsupported'));

    expect(await provideWebGpuDevice(wasmModule, fakeGpu(adapter))).toBeFalse();
    expect(warnSpy).toHaveBeenCalledTimes(1);
  });

  it('rejects fallback adapters', async () => {
    const fallback = fakeAdapter({isFallbackAdapter: true});

    expect(
      await provideWebGpuDevice(wasmModule, fakeGpu(fallback.adapter)),
    ).toBeFalse();
  });

  it('rejects software adapters', async () => {
    const software = fakeAdapter({
      info: {vendor: 'google', architecture: 'swiftshader', description: ''},
    });

    expect(
      await provideWebGpuDevice(wasmModule, fakeGpu(software.adapter)),
    ).toBeFalse();
  });

  it('removes the device from module and global holders when asked', () => {
    const globalHolder: Record<string, unknown> = {
      'preinitializedWebGPUDevice': {},
    };
    (globalThis as unknown as Record<string, unknown>)['Module'] = globalHolder;
    module['preinitializedWebGPUDevice'] = {};

    removeWebGpuDevice(wasmModule);

    expect(module['preinitializedWebGPUDevice']).toBeUndefined();
    expect(globalHolder['preinitializedWebGPUDevice']).toBeUndefined();
  });
});
