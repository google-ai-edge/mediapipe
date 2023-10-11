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

import {Classification, ClassificationList} from '../../../../framework/formats/classification_pb';
import {ClassificationResult, Classifications} from '../../../../tasks/cc/components/containers/proto/classifications_pb';

import {convertFromClassificationResultProto} from './classifier_result';

// The OSS JS API does not support the builder pattern.
// tslint:disable:jspb-use-builder-pattern

describe('convertFromClassificationResultProto()', () => {
  it('transforms custom values', () => {
    const classificationResult = new ClassificationResult();
    classificationResult.setTimestampMs(1);
    const classifcations = new Classifications();
    classifcations.setHeadIndex(1);
    classifcations.setHeadName('headName');
    const classificationList = new ClassificationList();
    const classification = new Classification();
    classification.setIndex(2);
    classification.setScore(0.3);
    classification.setDisplayName('displayName');
    classification.setLabel('categoryName');
    classificationList.addClassification(classification);
    classifcations.setClassificationList(classificationList);
    classificationResult.addClassifications(classifcations);

    const result = convertFromClassificationResultProto(classificationResult);

    expect(result).toEqual({
      classifications: [{
        categories: [{
          index: 2,
          score: 0.3,
          displayName: 'displayName',
          categoryName: 'categoryName'
        }],
        headIndex: 1,
        headName: 'headName'
      }],
      timestampMs: 1
    });
  });

  it('transforms default values', () => {
    const classificationResult = new ClassificationResult();
    const classifcations = new Classifications();
    const classificationList = new ClassificationList();
    const classification = new Classification();
    classificationList.addClassification(classification);
    classifcations.setClassificationList(classificationList);
    classificationResult.addClassifications(classifcations);

    const result = convertFromClassificationResultProto(classificationResult);

    expect(result).toEqual({
      classifications: [{
        categories: [{index: 0, score: 0, displayName: '', categoryName: ''}],
        headIndex: 0,
        headName: ''
      }],
    });
  });
});
