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
import {Embedding, EmbeddingResult as EmbeddingResultProto, FloatEmbedding} from '../../../../tasks/cc/components/containers/proto/embeddings_pb';
import {addJasmineCustomFloatEqualityTester, createSpyWasmModule, MediapipeTasksFake, SpyWasmModule, verifyGraph, verifyListenersRegistered} from '../../../../tasks/web/core/task_runner_test_utils';

import {AudioEmbedder, AudioEmbedderResult} from './audio_embedder';


// The OSS JS API does not support the builder pattern.
// tslint:disable:jspb-use-builder-pattern

class AudioEmbedderFake extends AudioEmbedder implements MediapipeTasksFake {
  lastSampleRate: number|undefined;
  calculatorName = 'mediapipe.tasks.audio.audio_embedder.AudioEmbedderGraph';
  graph: CalculatorGraphConfig|undefined;
  attachListenerSpies: jasmine.Spy[] = [];
  fakeWasmModule: SpyWasmModule;

  protoListener:
      ((binaryProto: Uint8Array, timestamp: number) => void)|undefined;
  protoVectorListener:
      ((binaryProtos: Uint8Array[], timestamp: number) => void)|undefined;

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
    this.attachListenerSpies[1] =
        spyOn(this.graphRunner, 'attachProtoVectorListener')
            .and.callFake((stream, listener) => {
              expect(stream).toEqual('timestamped_embeddings_out');
              this.protoVectorListener = listener;
            });
    spyOn(this.graphRunner, 'setGraph').and.callFake(binaryGraph => {
      this.graph = CalculatorGraphConfig.deserializeBinary(binaryGraph);
    });
    spyOn(this.graphRunner, 'addDoubleToStream').and.callFake(sampleRate => {
      this.lastSampleRate = sampleRate;
    });
    spyOn(this.graphRunner, 'addAudioToStreamWithShape');
  }
}

describe('AudioEmbedder', () => {
  let audioEmbedder: AudioEmbedderFake;

  beforeEach(async () => {
    addJasmineCustomFloatEqualityTester();
    audioEmbedder = new AudioEmbedderFake();
    await audioEmbedder.setOptions(
        {baseOptions: {modelAssetBuffer: new Uint8Array([])}});
  });

  afterEach(() => {
    audioEmbedder.close();
  });

  it('initializes graph', () => {
    verifyGraph(audioEmbedder);
    verifyListenersRegistered(audioEmbedder);
  });

  it('reloads graph when settings are changed', async () => {
    await audioEmbedder.setOptions({quantize: true});
    verifyGraph(audioEmbedder, [['embedderOptions', 'quantize'], true]);
    verifyListenersRegistered(audioEmbedder);

    await audioEmbedder.setOptions({quantize: undefined});
    verifyGraph(audioEmbedder, [['embedderOptions', 'quantize'], undefined]);
    verifyListenersRegistered(audioEmbedder);
  });

  it('can use custom models', async () => {
    const newModel = new Uint8Array([0, 1, 2, 3, 4]);
    const newModelBase64 = Buffer.from(newModel).toString('base64');
    await audioEmbedder.setOptions({
      baseOptions: {
        modelAssetBuffer: newModel,
      }
    });

    verifyGraph(
        audioEmbedder,
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
    await audioEmbedder.setOptions({quantize: true});
    await audioEmbedder.setOptions({l2Normalize: true});
    verifyGraph(
        audioEmbedder,
        ['embedderOptions', {'quantize': true, 'l2Normalize': true}]);
  });

  it('uses a sample rate of 48000 by default', async () => {
    audioEmbedder.embed(new Float32Array([]));
    expect(audioEmbedder.lastSampleRate).toEqual(48000);
  });

  it('uses default sample rate if none provided', async () => {
    audioEmbedder.setDefaultSampleRate(16000);
    audioEmbedder.embed(new Float32Array([]));
    expect(audioEmbedder.lastSampleRate).toEqual(16000);
  });

  it('uses custom sample rate if provided', async () => {
    audioEmbedder.setDefaultSampleRate(16000);
    audioEmbedder.embed(new Float32Array([]), 44100);
    expect(audioEmbedder.lastSampleRate).toEqual(44100);
  });

  describe('transforms results', () => {
    const embedding = new Embedding();
    embedding.setHeadIndex(1);
    embedding.setHeadName('headName');

    const floatEmbedding = new FloatEmbedding();
    floatEmbedding.setValuesList([0.1, 0.9]);

    embedding.setFloatEmbedding(floatEmbedding);
    const resultProto = new EmbeddingResultProto();
    resultProto.addEmbeddings(embedding);

    function validateEmbeddingResult(
        expectedEmbeddignResult: AudioEmbedderResult[]) {
      expect(expectedEmbeddignResult.length).toEqual(1);

      const [embeddingResult] = expectedEmbeddignResult;
      expect(embeddingResult.embeddings.length).toEqual(1);
      expect(embeddingResult.embeddings[0])
          .toEqual(
              {floatEmbedding: [0.1, 0.9], headIndex: 1, headName: 'headName'});
    }

    it('from embeddings stream', async () => {
      audioEmbedder.fakeWasmModule._waitUntilIdle.and.callFake(() => {
        verifyListenersRegistered(audioEmbedder);
        // Pass the test data to our listener
        audioEmbedder.protoListener!(resultProto.serializeBinary(), 1337);
      });

      // Invoke the audio embedder
      const embeddingResults = audioEmbedder.embed(new Float32Array([]));
      validateEmbeddingResult(embeddingResults);
    });

    it('from timestamped embeddgins stream', async () => {
      audioEmbedder.fakeWasmModule._waitUntilIdle.and.callFake(() => {
        verifyListenersRegistered(audioEmbedder);
        // Pass the test data to our listener
        audioEmbedder.protoVectorListener!
            ([resultProto.serializeBinary()], 1337);
      });

      // Invoke the audio embedder
      const embeddingResults = audioEmbedder.embed(new Float32Array([]), 42);
      validateEmbeddingResult(embeddingResults);
    });
  });
});
