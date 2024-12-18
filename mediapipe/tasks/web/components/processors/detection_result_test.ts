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

import {Detection as DetectionProto} from '../../../../framework/formats/detection_pb';
import {LocationData} from '../../../../framework/formats/location_data_pb';

import {convertFromDetectionProto} from './detection_result';

// The OSS JS API does not support the builder pattern.
// tslint:disable:jspb-use-builder-pattern

describe('convertFromDetectionProto()', () => {
  it('transforms custom values', () => {
    const detection = new DetectionProto();
    detection.addScore(0.1);
    detection.addLabelId(1);
    detection.addLabel('foo');
    detection.addDisplayName('bar');

    const locationData = new LocationData();
    const boundingBox = new LocationData.BoundingBox();
    boundingBox.setXmin(1);
    boundingBox.setYmin(2);
    boundingBox.setWidth(3);
    boundingBox.setHeight(4);
    locationData.setBoundingBox(boundingBox);

    const keypoint = new LocationData.RelativeKeypoint();
    keypoint.setX(5);
    keypoint.setY(6);
    keypoint.setScore(0.7);
    keypoint.setKeypointLabel('bar');
    locationData.addRelativeKeypoints(new LocationData.RelativeKeypoint());

    detection.setLocationData(locationData);

    const result = convertFromDetectionProto(detection);

    expect(result).toEqual({
      categories: [{
        score: 0.1,
        index: 1,
        categoryName: 'foo',
        displayName: 'bar',
      }],
      boundingBox: {originX: 1, originY: 2, width: 3, height: 4, angle: 0},
      keypoints: [{
        x: 5,
        y: 6,
        score: 0.7,
        label: 'bar',
      }],
    });
  });

  it('transforms default values', () => {
    const detection = new DetectionProto();
    detection.addScore(0.2);
    const locationData = new LocationData();
    const boundingBox = new LocationData.BoundingBox();
    locationData.setBoundingBox(boundingBox);
    detection.setLocationData(locationData);

    const result = convertFromDetectionProto(detection);

    expect(result).toEqual({
      categories: [{
        score: 0.2,
        index: -1,
        categoryName: '',
        displayName: '',
      }],
      boundingBox: {originX: 0, originY: 0, width: 0, height: 0, angle: 0},
      keypoints: []
    });
  });
});
