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
import {Embedding, EmbeddingResult, FloatEmbedding} from '../../../../tasks/cc/components/containers/proto/embeddings_pb';
import {addJasmineCustomFloatEqualityTester, createSpyWasmModule, MediapipeTasksFake, SpyWasmModule, verifyGraph, verifyListenersRegistered} from '../../../../tasks/web/core/task_runner_test_utils';

import {ImageEmbedder} from './image_embedder';

// The OSS JS API does not support the builder pattern.
// tslint:disable:jspb-use-builder-pattern

class ImageEmbedderFake extends ImageEmbedder implements MediapipeTasksFake {
  calculatorName = 'mediapipe.tasks.vision.image_embedder.ImageEmbedderGraph';
  graph: CalculatorGraphConfig|undefined;
  attachListenerSpies: jasmine.Spy[] = [];
  fakeWasmModule: SpyWasmModule;
  protoListener:
      ((binaryProtos: Uint8Array, timestamp: number) => void)|undefined;

  constructor() {
    super(createSpyWasmModule(), /* glCanvas= */ null);
    this.fakeWasmModule =
        this.graphRunner.wasmModule as unknown as SpyWasmModule;

    this.attachListenerSpies[0] =
        spyOn(this.graphRunner, 'attachProtoListener')
            .and.callFake((stream, listener) => {
              expect(stream).toEqual('embeddings_out');
              this.protoListener = listener;
            });
    spyOn(this.graphRunner, 'setGraph').and.callFake(binaryGraph => {
      this.graph = CalculatorGraphConfig.deserializeBinary(binaryGraph);
    });
    spyOn(this.graphRunner, 'addGpuBufferAsImageToStream');
  }
}

describe('ImageEmbedder', () => {
  let imageEmbedder: ImageEmbedderFake;

  beforeEach(async () => {
    addJasmineCustomFloatEqualityTester();
    imageEmbedder = new ImageEmbedderFake();
    await imageEmbedder.setOptions(
        {baseOptions: {modelAssetBuffer: new Uint8Array([])}});
  });

  afterEach(() => {
    imageEmbedder.close();
  });

  it('initializes graph', async () => {
    verifyGraph(imageEmbedder);
    verifyListenersRegistered(imageEmbedder);
  });

  it('reloads graph when settings are changed', async () => {
    verifyListenersRegistered(imageEmbedder);

    await imageEmbedder.setOptions({quantize: true});
    verifyGraph(imageEmbedder, [['embedderOptions', 'quantize'], true]);
    verifyListenersRegistered(imageEmbedder);

    await imageEmbedder.setOptions({quantize: undefined});
    verifyGraph(imageEmbedder, [['embedderOptions', 'quantize'], undefined]);
    verifyListenersRegistered(imageEmbedder);
  });

  it('can use custom models', async () => {
    const newModel = new Uint8Array([0, 1, 2, 3, 4]);
    const newModelBase64 = Buffer.from(newModel).toString('base64');
    await imageEmbedder.setOptions({
      baseOptions: {
        modelAssetBuffer: newModel,
      }
    });

    verifyGraph(
        imageEmbedder,
        /* expectedCalculatorOptions= */ undefined,
        /* expectedBaseOptions= */[
          'modelAsset', {
            fileContent: newModelBase64,
            fileName: undefined,
            fileDescriptorMeta: undefined,
            filePointerMeta: undefined
          }
        ]);
  });

  it('overrides options', async () => {
    await imageEmbedder.setOptions({quantize: true});
    await imageEmbedder.setOptions({l2Normalize: true});
    verifyGraph(
        imageEmbedder,
        ['embedderOptions', {'quantize': true, 'l2Normalize': true}]);
  });

  describe('transforms result', () => {
    beforeEach(() => {
      const floatEmbedding = new FloatEmbedding();
      floatEmbedding.setValuesList([0.1, 0.9]);

      const embedding = new Embedding();
      embedding.setHeadIndex(1);
      embedding.setHeadName('headName');
      embedding.setFloatEmbedding(floatEmbedding);

      const resultProto = new EmbeddingResult();
      resultProto.addEmbeddings(embedding);
      resultProto.setTimestampMs(42);

      // Pass the test data to our listener
      imageEmbedder.fakeWasmModule._waitUntilIdle.and.callFake(() => {
        verifyListenersRegistered(imageEmbedder);
        imageEmbedder.protoListener!(resultProto.serializeBinary(), 1337);
      });
    });

    it('for image mode', async () => {
      // Invoke the image embedder
      const embeddingResult = imageEmbedder.embed({} as HTMLImageElement);

      expect(imageEmbedder.fakeWasmModule._waitUntilIdle).toHaveBeenCalled();
      expect(embeddingResult).toEqual({
        embeddings:
            [{headIndex: 1, headName: 'headName', floatEmbedding: [0.1, 0.9]}],
        timestampMs: 42
      });
    });

    it('for video mode', async () => {
      await imageEmbedder.setOptions({runningMode: 'VIDEO'});

      // Invoke the video embedder
      const embeddingResult =
          imageEmbedder.embedForVideo({} as HTMLImageElement, 42);

      expect(imageEmbedder.fakeWasmModule._waitUntilIdle).toHaveBeenCalled();
      expect(embeddingResult).toEqual({
        embeddings:
            [{headIndex: 1, headName: 'headName', floatEmbedding: [0.1, 0.9]}],
        timestampMs: 42
      });
    });
  });
});
