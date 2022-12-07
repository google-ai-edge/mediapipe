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

import 'jasmine';

// Placeholder for internal dependency on encodeByteArray
// Placeholder for internal dependency on trusted resource URL builder

import {convertBaseOptionsToProto} from './base_options';

describe('convertBaseOptionsToProto()', () => {
  const mockBytes = new Uint8Array([0, 1, 2, 3]);
  const mockBytesResult = {
    modelAsset: {
      fileContent: Buffer.from(mockBytes).toString('base64'),
      fileName: undefined,
      fileDescriptorMeta: undefined,
      filePointerMeta: undefined,
    },
    useStreamMode: false,
    acceleration: {
      xnnpack: undefined,
      gpu: undefined,
      tflite: {},
    },
  };

  let fetchSpy: jasmine.Spy;

  beforeEach(() => {
    fetchSpy = jasmine.createSpy().and.callFake(async url => {
      expect(url).toEqual('foo');
      return {
        arrayBuffer: () => mockBytes.buffer,
      } as unknown as Response;
    });
    global.fetch = fetchSpy;
  });

  it('verifies that at least one model asset option is provided', async () => {
    await expectAsync(convertBaseOptionsToProto({}))
        .toBeRejectedWithError(
            /Either baseOptions.modelAssetPath or baseOptions.modelAssetBuffer must be set/);
  });

  it('verifies that no more than one model asset option is provided', async () => {
    await expectAsync(convertBaseOptionsToProto({
      modelAssetPath: `foo`,
      modelAssetBuffer: new Uint8Array([])
    }))
        .toBeRejectedWithError(
            /Cannot set both baseOptions.modelAssetPath and baseOptions.modelAssetBuffer/);
  });

  it('downloads model', async () => {
    const baseOptionsProto = await convertBaseOptionsToProto({
      modelAssetPath: `foo`,
    });

    expect(fetchSpy).toHaveBeenCalled();
    expect(baseOptionsProto.toObject()).toEqual(mockBytesResult);
  });

  it('does not download model when bytes are provided', async () => {
    const baseOptionsProto = await convertBaseOptionsToProto({
      modelAssetBuffer: new Uint8Array(mockBytes),
    });

    expect(fetchSpy).not.toHaveBeenCalled();
    expect(baseOptionsProto.toObject()).toEqual(mockBytesResult);
  });

  it('can enable CPU delegate', async () => {
    const baseOptionsProto = await convertBaseOptionsToProto({
      modelAssetBuffer: new Uint8Array(mockBytes),
      delegate: 'cpu',
    });
    expect(baseOptionsProto.toObject()).toEqual(mockBytesResult);
  });

  it('can enable GPU delegate', async () => {
    const baseOptionsProto = await convertBaseOptionsToProto({
      modelAssetBuffer: new Uint8Array(mockBytes),
      delegate: 'gpu',
    });
    expect(baseOptionsProto.toObject()).toEqual({
      ...mockBytesResult,
      acceleration: {
        xnnpack: undefined,
        gpu: {
          useAdvancedGpuApi: false,
          api: 0,
          allowPrecisionLoss: true,
          cachedKernelPath: undefined,
          serializedModelDir: undefined,
          modelToken: undefined,
          usage: 2,
        },
        tflite: undefined,
      },
    });
  });

  it('can reset delegate', async () => {
    let baseOptionsProto = await convertBaseOptionsToProto({
      modelAssetBuffer: new Uint8Array(mockBytes),
      delegate: 'gpu',
    });
    // Clear backend
    baseOptionsProto =
        await convertBaseOptionsToProto({delegate: undefined}, baseOptionsProto);
    expect(baseOptionsProto.toObject()).toEqual(mockBytesResult);
  });
});
