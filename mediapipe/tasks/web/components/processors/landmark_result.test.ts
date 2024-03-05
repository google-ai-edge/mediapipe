/**
 * Copyright 2023 The MediaPipe Authors.
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

import {convertToLandmarks, convertToWorldLandmarks} from '../../../../tasks/web/components/processors/landmark_result';
import {createLandmarks, createWorldLandmarks} from '../../../../tasks/web/components/processors/landmark_result_test_lib';


// The OSS JS API does not support the builder pattern.
// tslint:disable:jspb-use-builder-pattern

describe('convertToLandmarks()', () => {
  it('transforms custom values', () => {
    const landmarkListProto = createLandmarks(0.1, 0.2, 0.3);
    const result = convertToLandmarks(landmarkListProto);
    expect(result).toEqual([{x: 0.1, y: 0.2, z: 0.3, visibility: 0}]);
  });

  it('transforms default values', () => {
    const landmarkListProto = createLandmarks();
    const result = convertToLandmarks(landmarkListProto);
    expect(result).toEqual([{x: 0, y: 0, z: 0, visibility: 0}]);
  });
});

describe('convertToWorldLandmarks()', () => {
  it('transforms custom values', () => {
    const worldLandmarkListProto = createWorldLandmarks(10, 20, 30);
    const result = convertToWorldLandmarks(worldLandmarkListProto);
    expect(result).toEqual([{x: 10, y: 20, z: 30, visibility: 0}]);
  });

  it('transforms default values', () => {
    const worldLandmarkListProto = createWorldLandmarks();
    const result = convertToWorldLandmarks(worldLandmarkListProto);
    expect(result).toEqual([{x: 0, y: 0, z: 0, visibility: 0}]);
  });
});
