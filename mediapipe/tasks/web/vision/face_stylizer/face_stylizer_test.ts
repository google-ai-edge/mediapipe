/**
 * Copyright 2022 The MediaPipe Authors.
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
import {CalculatorGraphConfig} from '../../../../framework/calculator_pb';
import {addJasmineCustomFloatEqualityTester, createSpyWasmModule, MediapipeTasksFake, SpyWasmModule, verifyGraph, verifyListenersRegistered} from '../../../../tasks/web/core/task_runner_test_utils';
import {WasmImage} from '../../../../web/graph_runner/graph_runner_image_lib';

import {FaceStylizer} from './face_stylizer';

class FaceStylizerFake extends FaceStylizer implements MediapipeTasksFake {
  calculatorName = 'mediapipe.tasks.vision.face_stylizer.FaceStylizerGraph';
  attachListenerSpies: jasmine.Spy[] = [];
  graph: CalculatorGraphConfig|undefined;

  fakeWasmModule: SpyWasmModule;
  imageListener: ((images: WasmImage, timestamp: number) => void)|undefined;
  emptyPacketListener: ((timestamp: number) => void)|undefined;

  constructor() {
    super(createSpyWasmModule(), /* glCanvas= */ null);
    this.fakeWasmModule =
        this.graphRunner.wasmModule as unknown as SpyWasmModule;

    this.attachListenerSpies[0] =
        spyOn(this.graphRunner, 'attachImageListener')
            .and.callFake((stream, listener) => {
              expect(stream).toEqual('stylized_image');
              this.imageListener = listener;
            });
    this.attachListenerSpies[1] =
        spyOn(this.graphRunner, 'attachEmptyPacketListener')
            .and.callFake((stream, listener) => {
              expect(stream).toEqual('stylized_image');
              this.emptyPacketListener = listener;
            });
    spyOn(this.graphRunner, 'setGraph').and.callFake(binaryGraph => {
      this.graph = CalculatorGraphConfig.deserializeBinary(binaryGraph);
    });
    spyOn(this.graphRunner, 'addGpuBufferAsImageToStream');
  }
}

describe('FaceStylizer', () => {
  let faceStylizer: FaceStylizerFake;

  beforeEach(async () => {
    addJasmineCustomFloatEqualityTester();
    faceStylizer = new FaceStylizerFake();
    await faceStylizer.setOptions(
        {baseOptions: {modelAssetBuffer: new Uint8Array([])}});
  });

  afterEach(() => {
    faceStylizer.close();
  });

  it('initializes graph', async () => {
    verifyGraph(faceStylizer);
    verifyListenersRegistered(faceStylizer);
  });

  it('can use custom models', async () => {
    const newModel = new Uint8Array([0, 1, 2, 3, 4]);
    const newModelBase64 = Buffer.from(newModel).toString('base64');
    await faceStylizer.setOptions({
      baseOptions: {
        modelAssetBuffer: newModel,
      }
    });

    verifyGraph(
        faceStylizer,
        /* expectedCalculatorOptions= */ undefined,
        /* expectedBaseOptions= */
        [
          'modelAsset', {
            fileContent: newModelBase64,
            fileName: undefined,
            fileDescriptorMeta: undefined,
            filePointerMeta: undefined
          }
        ]);
  });

  it('returns result', () => {
    if (typeof ImageData === 'undefined') {
      console.log('ImageData tests are not supported on Node');
      return;
    }

    // Pass the test data to our listener
    faceStylizer.fakeWasmModule._waitUntilIdle.and.callFake(() => {
      verifyListenersRegistered(faceStylizer);
      faceStylizer.imageListener!
          ({data: new Uint8Array([1, 1, 1, 1]), width: 1, height: 1},
           /* timestamp= */ 1337);
    });

    // Invoke the face stylizeer
    const image = faceStylizer.stylize({} as HTMLImageElement);
    expect(faceStylizer.fakeWasmModule._waitUntilIdle).toHaveBeenCalled();
    expect(image).not.toBeNull();
    expect(image!.hasImageData()).toBeTrue();
    expect(image!.width).toEqual(1);
    expect(image!.height).toEqual(1);
    image!.close();
  });

  it('invokes callback', (done) => {
    if (typeof ImageData === 'undefined') {
      console.log('ImageData tests are not supported on Node');
      done();
      return;
    }

    // Pass the test data to our listener
    faceStylizer.fakeWasmModule._waitUntilIdle.and.callFake(() => {
      verifyListenersRegistered(faceStylizer);
      faceStylizer.imageListener!
          ({data: new Uint8Array([1, 1, 1, 1]), width: 1, height: 1},
           /* timestamp= */ 1337);
    });

    // Invoke the face stylizeer
    faceStylizer.stylize({} as HTMLImageElement, image => {
      expect(faceStylizer.fakeWasmModule._waitUntilIdle).toHaveBeenCalled();
      expect(image).not.toBeNull();
      expect(image!.hasImageData()).toBeTrue();
      expect(image!.width).toEqual(1);
      expect(image!.height).toEqual(1);
      done();
    });
  });

  it('invokes callback even when no faces are detected', (done) => {
    // Pass the test data to our listener
    faceStylizer.fakeWasmModule._waitUntilIdle.and.callFake(() => {
      verifyListenersRegistered(faceStylizer);
      faceStylizer.emptyPacketListener!(/* timestamp= */ 1337);
    });

    // Invoke the face stylizeer
    faceStylizer.stylize({} as HTMLImageElement, image => {
      expect(faceStylizer.fakeWasmModule._waitUntilIdle).toHaveBeenCalled();
      expect(image).toBeNull();
      done();
    });
  });
});
