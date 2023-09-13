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

import {CalculatorGraphConfig} from '../../../../framework/calculator_pb';
import {Classification, ClassificationList} from '../../../../framework/formats/classification_pb';
import {ClassificationResult, Classifications} from '../../../../tasks/cc/components/containers/proto/classifications_pb';
import {addJasmineCustomFloatEqualityTester, createSpyWasmModule, MediapipeTasksFake, verifyGraph, verifyListenersRegistered} from '../../../../tasks/web/core/task_runner_test_utils';

import {AudioClassifier} from './audio_classifier';

// The OSS JS API does not support the builder pattern.
// tslint:disable:jspb-use-builder-pattern

class AudioClassifierFake extends AudioClassifier implements
    MediapipeTasksFake {
  lastSampleRate: number|undefined;
  calculatorName =
      'mediapipe.tasks.audio.audio_classifier.AudioClassifierGraph';
  attachListenerSpies: jasmine.Spy[] = [];
  graph: CalculatorGraphConfig|undefined;

  private protoVectorListener:
      ((binaryProtos: Uint8Array[], timestamp: number) => void)|undefined;
  private resultProtoVector: ClassificationResult[] = [];

  constructor() {
    super(createSpyWasmModule(), /* glCanvas= */ null);

    this.attachListenerSpies[0] =
        spyOn(this.graphRunner, 'attachProtoVectorListener')
            .and.callFake((stream, listener) => {
              expect(stream).toEqual('timestamped_classifications');
              this.protoVectorListener = listener;
            });
    spyOn(this.graphRunner, 'addDoubleToStream')
        .and.callFake((sampleRate, streamName, timestamp) => {
          if (streamName === 'sample_rate') {
            this.lastSampleRate = sampleRate;
          }
        });
    spyOn(this.graphRunner, 'addAudioToStreamWithShape')
        .and.callFake(
            (audioData, numChannels, numSamples, streamName, timestamp) => {
              expect(numChannels).toBe(1);
            });
    spyOn(this.graphRunner, 'finishProcessing').and.callFake(() => {
      if (!this.protoVectorListener) return;
      this.protoVectorListener(
          this.resultProtoVector.map(
              classificationResult => classificationResult.serializeBinary()),
          1337);
    });
    spyOn(this.graphRunner, 'setGraph').and.callFake(binaryGraph => {
      this.graph = CalculatorGraphConfig.deserializeBinary(binaryGraph);
    });
  }

  /** Sets the Protobuf that will be send to the API. */
  setResults(results: ClassificationResult[]): void {
    this.resultProtoVector = results;
  }
}

describe('AudioClassifier', () => {
  let audioClassifier: AudioClassifierFake;

  beforeEach(async () => {
    addJasmineCustomFloatEqualityTester();
    audioClassifier = new AudioClassifierFake();
    await audioClassifier.setOptions(
        {baseOptions: {modelAssetBuffer: new Uint8Array([])}});
  });

  afterEach(() => {
    audioClassifier.close();
  });

  it('initializes graph', async () => {
    verifyGraph(audioClassifier);
    verifyListenersRegistered(audioClassifier);
  });

  it('reloads graph when settings are changed', async () => {
    await audioClassifier.setOptions({maxResults: 1});
    verifyGraph(audioClassifier, [['classifierOptions', 'maxResults'], 1]);
    verifyListenersRegistered(audioClassifier);

    await audioClassifier.setOptions({maxResults: 5});
    verifyGraph(audioClassifier, [['classifierOptions', 'maxResults'], 5]);
    verifyListenersRegistered(audioClassifier);
  });

  it('merges options', async () => {
    await audioClassifier.setOptions({maxResults: 1});
    await audioClassifier.setOptions({displayNamesLocale: 'en'});
    verifyGraph(audioClassifier, [
      'classifierOptions', {
        maxResults: 1,
        displayNamesLocale: 'en',
        scoreThreshold: undefined,
        categoryAllowlistList: [],
        categoryDenylistList: []
      }
    ]);
  });

  it('uses a sample rate of 48000 by default', async () => {
    audioClassifier.classify(new Float32Array([]));
    expect(audioClassifier.lastSampleRate).toEqual(48000);
  });

  it('uses default sample rate if none provided', async () => {
    audioClassifier.setDefaultSampleRate(16000);
    audioClassifier.classify(new Float32Array([]));
    expect(audioClassifier.lastSampleRate).toEqual(16000);
  });

  it('uses custom sample rate if provided', async () => {
    audioClassifier.setDefaultSampleRate(16000);
    audioClassifier.classify(new Float32Array([]), 44100);
    expect(audioClassifier.lastSampleRate).toEqual(44100);
  });

  it('transforms results', async () => {
    const resultProtoVector: ClassificationResult[] = [];

    let classificationResult = new ClassificationResult();
    classificationResult.setTimestampMs(0);
    let classifcations = new Classifications();
    classifcations.setHeadIndex(1);
    classifcations.setHeadName('headName');
    let classificationList = new ClassificationList();
    let classification = new Classification();
    classification.setIndex(1);
    classification.setScore(0.2);
    classification.setDisplayName('displayName');
    classification.setLabel('categoryName');
    classificationList.addClassification(classification);
    classifcations.setClassificationList(classificationList);
    classificationResult.addClassifications(classifcations);
    resultProtoVector.push(classificationResult);

    classificationResult = new ClassificationResult();
    classificationResult.setTimestampMs(1);
    classifcations = new Classifications();
    classificationList = new ClassificationList();
    classification = new Classification();
    classification.setIndex(2);
    classification.setScore(0.3);
    classificationList.addClassification(classification);
    classifcations.setClassificationList(classificationList);
    classificationResult.addClassifications(classifcations);
    resultProtoVector.push(classificationResult);

    // Invoke the audio classifier
    audioClassifier.setResults(resultProtoVector);
    const results = audioClassifier.classify(new Float32Array([]));
    expect(results.length).toEqual(2);
    expect(results[0]).toEqual({
      classifications: [{
        categories: [{
          index: 1,
          score: 0.2,
          displayName: 'displayName',
          categoryName: 'categoryName'
        }],
        headIndex: 1,
        headName: 'headName'
      }],
      timestampMs: 0
    });
    expect(results[1]).toEqual({
      classifications: [{
        categories: [{index: 2, score: 0.3, displayName: '', categoryName: ''}],
        headIndex: 0,
        headName: ''
      }],
      timestampMs: 1
    });
  });

  it('clears results between invocations', async () => {
    const classificationResult = new ClassificationResult();
    const classifcations = new Classifications();
    const classificationList = new ClassificationList();
    const classification = new Classification();
    classificationList.addClassification(classification);
    classifcations.setClassificationList(classificationList);
    classificationResult.addClassifications(classifcations);

    audioClassifier.setResults([classificationResult]);

    // Invoke the gesture recognizer twice
    const classifications1 = audioClassifier.classify(new Float32Array([]));
    const classifications2 = audioClassifier.classify(new Float32Array([]));

    // Verify that gestures2 is not a concatenation of all previously returned
    // gestures.
    expect(classifications1).toEqual(classifications2);
  });
});
