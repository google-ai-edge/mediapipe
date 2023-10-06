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

import {ImageClassifier} from './image_classifier';

// The OSS JS API does not support the builder pattern.
// tslint:disable:jspb-use-builder-pattern

class ImageClassifierFake extends ImageClassifier implements
    MediapipeTasksFake {
  calculatorName =
      'mediapipe.tasks.vision.image_classifier.ImageClassifierGraph';
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
              expect(stream).toEqual('classifications');
              this.protoListener = listener;
            });
    spyOn(this.graphRunner, 'setGraph').and.callFake(binaryGraph => {
      this.graph = CalculatorGraphConfig.deserializeBinary(binaryGraph);
    });
    spyOn(this.graphRunner, 'addGpuBufferAsImageToStream');
  }
}

describe('ImageClassifier', () => {
  let imageClassifier: ImageClassifierFake;

  beforeEach(async () => {
    addJasmineCustomFloatEqualityTester();
    imageClassifier = new ImageClassifierFake();
    await imageClassifier.setOptions(
        {baseOptions: {modelAssetBuffer: new Uint8Array([])}});
  });

  afterEach(() => {
    imageClassifier.close();
  });

  it('initializes graph', async () => {
    verifyGraph(imageClassifier);
    verifyListenersRegistered(imageClassifier);
  });

  it('reloads graph when settings are changed', async () => {
    await imageClassifier.setOptions({maxResults: 1});
    verifyGraph(imageClassifier, [['classifierOptions', 'maxResults'], 1]);
    verifyListenersRegistered(imageClassifier);

    await imageClassifier.setOptions({maxResults: 5});
    verifyGraph(imageClassifier, [['classifierOptions', 'maxResults'], 5]);
    verifyListenersRegistered(imageClassifier);
  });

  it('can use custom models', async () => {
    const newModel = new Uint8Array([0, 1, 2, 3, 4]);
    const newModelBase64 = Buffer.from(newModel).toString('base64');
    await imageClassifier.setOptions({
      baseOptions: {
        modelAssetBuffer: newModel,
      }
    });

    verifyGraph(
        imageClassifier,
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

  it('merges options', async () => {
    await imageClassifier.setOptions({maxResults: 1});
    await imageClassifier.setOptions({displayNamesLocale: 'en'});
    verifyGraph(imageClassifier, [['classifierOptions', 'maxResults'], 1]);
    verifyGraph(
        imageClassifier, [['classifierOptions', 'displayNamesLocale'], 'en']);
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
    imageClassifier.fakeWasmModule._waitUntilIdle.and.callFake(() => {
      verifyListenersRegistered(imageClassifier);
      imageClassifier.protoListener!
          (classificationResult.serializeBinary(), 1337);
    });

    // Invoke the image classifier
    const result = imageClassifier.classify({} as HTMLImageElement);

    expect(imageClassifier.fakeWasmModule._waitUntilIdle).toHaveBeenCalled();
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
