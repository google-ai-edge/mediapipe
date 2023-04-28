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
import {Embedding, EmbeddingResult, FloatEmbedding, QuantizedEmbedding} from '../../../../tasks/cc/components/containers/proto/embeddings_pb';
import {addJasmineCustomFloatEqualityTester, createSpyWasmModule, MediapipeTasksFake, SpyWasmModule, verifyGraph, verifyListenersRegistered} from '../../../../tasks/web/core/task_runner_test_utils';

import {TextEmbedder} from './text_embedder';

// The OSS JS API does not support the builder pattern.
// tslint:disable:jspb-use-builder-pattern

class TextEmbedderFake extends TextEmbedder implements MediapipeTasksFake {
  calculatorName = 'mediapipe.tasks.text.text_embedder.TextEmbedderGraph';
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
  }
}

describe('TextEmbedder', () => {
  let textEmbedder: TextEmbedderFake;

  beforeEach(async () => {
    addJasmineCustomFloatEqualityTester();
    textEmbedder = new TextEmbedderFake();
    await textEmbedder.setOptions(
        {baseOptions: {modelAssetBuffer: new Uint8Array([])}});
  });

  afterEach(() => {
    textEmbedder.close();
  });

  it('initializes graph', async () => {
    verifyGraph(textEmbedder);
    verifyListenersRegistered(textEmbedder);
  });

  it('reloads graph when settings are changed', async () => {
    await textEmbedder.setOptions({quantize: true});
    verifyGraph(textEmbedder, [['embedderOptions', 'quantize'], true]);
    verifyListenersRegistered(textEmbedder);

    await textEmbedder.setOptions({quantize: undefined});
    verifyGraph(textEmbedder, [['embedderOptions', 'quantize'], undefined]);
    verifyListenersRegistered(textEmbedder);
  });

  it('can use custom models', async () => {
    const newModel = new Uint8Array([0, 1, 2, 3, 4]);
    const newModelBase64 = Buffer.from(newModel).toString('base64');
    await textEmbedder.setOptions({
      baseOptions: {
        modelAssetBuffer: newModel,
      }
    });

    verifyGraph(
        textEmbedder,
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

  it('combines options', async () => {
    await textEmbedder.setOptions({quantize: true});
    await textEmbedder.setOptions({l2Normalize: true});
    verifyGraph(
        textEmbedder,
        ['embedderOptions', {'quantize': true, 'l2Normalize': true}]);
  });

  it('transforms results', async () => {
    const embedding = new Embedding();
    embedding.setHeadIndex(1);
    embedding.setHeadName('headName');

    const floatEmbedding = new FloatEmbedding();
    floatEmbedding.setValuesList([0.1, 0.9]);

    embedding.setFloatEmbedding(floatEmbedding);
    const resultProto = new EmbeddingResult();
    resultProto.addEmbeddings(embedding);

    // Pass the test data to our listener
    textEmbedder.fakeWasmModule._waitUntilIdle.and.callFake(() => {
      verifyListenersRegistered(textEmbedder);
      textEmbedder.protoListener!(resultProto.serializeBinary(), 1337);
    });

    // Invoke the text embedder
    const embeddingResult = textEmbedder.embed('foo');

    expect(textEmbedder.fakeWasmModule._waitUntilIdle).toHaveBeenCalled();
    expect(embeddingResult.embeddings.length).toEqual(1);
    expect(embeddingResult.embeddings[0])
        .toEqual(
            {floatEmbedding: [0.1, 0.9], headIndex: 1, headName: 'headName'});
  });

  it('transforms custom quantized values', async () => {
    const embedding = new Embedding();
    embedding.setHeadIndex(1);
    embedding.setHeadName('headName');

    const quantizedEmbedding = new QuantizedEmbedding();
    const quantizedValues = new Uint8Array([1, 2, 3]);
    quantizedEmbedding.setValues(quantizedValues);

    embedding.setQuantizedEmbedding(quantizedEmbedding);
    const resultProto = new EmbeddingResult();
    resultProto.addEmbeddings(embedding);

    // Pass the test data to our listener
    textEmbedder.fakeWasmModule._waitUntilIdle.and.callFake(() => {
      verifyListenersRegistered(textEmbedder);
      textEmbedder.protoListener!(resultProto.serializeBinary(), 1337);
    });

    // Invoke the text embedder
    const embeddingsResult = textEmbedder.embed('foo');

    expect(textEmbedder.fakeWasmModule._waitUntilIdle).toHaveBeenCalled();
    expect(embeddingsResult.embeddings.length).toEqual(1);
    expect(embeddingsResult.embeddings[0]).toEqual({
      quantizedEmbedding: new Uint8Array([1, 2, 3]),
      headIndex: 1,
      headName: 'headName'
    });
  });
});
