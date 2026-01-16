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
import {Classification, ClassificationList} from '../../../../framework/formats/classification_pb';
import {ClassificationResult, Classifications} from '../../../../tasks/cc/components/containers/proto/classifications_pb';
import {addJasmineCustomFloatEqualityTester, createSpyWasmModule, MediapipeTasksFake, SpyWasmModule, verifyGraph, verifyListenersRegistered} from '../../../../tasks/web/core/task_runner_test_utils';

import {TextClassifier} from './text_classifier';

// The OSS JS API does not support the builder pattern.
// tslint:disable:jspb-use-builder-pattern

class TextClassifierFake extends TextClassifier implements MediapipeTasksFake {
  calculatorName = 'mediapipe.tasks.text.text_classifier.TextClassifierGraph';
  attachListenerSpies: jasmine.Spy[] = [];
  graph: CalculatorGraphConfig|undefined;
  fakeWasmModule: SpyWasmModule;
  protoListener:
      ((binaryProto: Uint8Array, timestamp: number) => void)|undefined;

  constructor() {
    super(createSpyWasmModule(), /* glCanvas= */ null);
    this.fakeWasmModule =
        this.graphRunner.wasmModule as unknown as SpyWasmModule;
    this.attachListenerSpies[0] =
        spyOn(this.graphRunner, 'attachProtoListener')
            .and.callFake((stream, listener) => {
              expect(stream).toEqual('classifications_out');
              this.protoListener = listener;
            });
    spyOn(this.graphRunner, 'setGraph').and.callFake(binaryGraph => {
      this.graph = CalculatorGraphConfig.deserializeBinary(binaryGraph);
    });
  }
}

describe('TextClassifier', () => {
  let textClassifier: TextClassifierFake;

  beforeEach(async () => {
    addJasmineCustomFloatEqualityTester();
    textClassifier = new TextClassifierFake();
    await textClassifier.setOptions(
        {baseOptions: {modelAssetBuffer: new Uint8Array([])}});
  });

  afterEach(() => {
    textClassifier.close();
  });

  it('initializes graph', async () => {
    verifyGraph(textClassifier);
    verifyListenersRegistered(textClassifier);
  });

  it('reloads graph when settings are changed', async () => {
    await textClassifier.setOptions({maxResults: 1});
    verifyGraph(textClassifier, [['classifierOptions', 'maxResults'], 1]);
    verifyListenersRegistered(textClassifier);

    await textClassifier.setOptions({maxResults: 5});
    verifyGraph(textClassifier, [['classifierOptions', 'maxResults'], 5]);
    verifyListenersRegistered(textClassifier);
  });

  it('can use custom models', async () => {
    const newModel = new Uint8Array([0, 1, 2, 3, 4]);
    const newModelBase64 = Buffer.from(newModel).toString('base64');
    await textClassifier.setOptions({
      baseOptions: {
        modelAssetBuffer: newModel,
      }
    });

    verifyGraph(
        textClassifier,
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

  it('merges options', async () => {
    await textClassifier.setOptions({maxResults: 1});
    await textClassifier.setOptions({displayNamesLocale: 'en'});
    verifyGraph(textClassifier, [
      'classifierOptions', {
        maxResults: 1,
        displayNamesLocale: 'en',
        scoreThreshold: undefined,
        categoryAllowlistList: [],
        categoryDenylistList: []
      }
    ]);
  });

  it('transforms results', async () => {
    const classificationResult = new ClassificationResult();
    const classifcations = new Classifications();
    classifcations.setHeadIndex(1);
    classifcations.setHeadName('headName');
    const classificationList = new ClassificationList();
    const classification = new Classification();
    classification.setIndex(1);
    classification.setScore(0.2);
    classification.setDisplayName('displayName');
    classification.setLabel('categoryName');
    classificationList.addClassification(classification);
    classifcations.setClassificationList(classificationList);
    classificationResult.addClassifications(classifcations);

    // Pass the test data to our listener
    textClassifier.fakeWasmModule._waitUntilIdle.and.callFake(() => {
      verifyListenersRegistered(textClassifier);
      textClassifier.protoListener!
          (classificationResult.serializeBinary(), 1337);
    });

    // Invoke the text classifier
    const result = textClassifier.classify('foo');

    expect(textClassifier.fakeWasmModule._waitUntilIdle).toHaveBeenCalled();
    expect(result).toEqual({
      classifications: [{
        categories: [{
          index: 1,
          score: 0.2,
          displayName: 'displayName',
          categoryName: 'categoryName'
        }],
        headIndex: 1,
        headName: 'headName'
      }]
    });
  });
});
