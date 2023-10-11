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

import {Landmark as LandmarkProto, LandmarkList as LandmarkListProto, NormalizedLandmark as NormalizedLandmarkProto, NormalizedLandmarkList as NormalizedLandmarkListProto} from '../../../../framework/formats/landmark_pb';

// The OSS JS API does not support the builder pattern.
// tslint:disable:jspb-use-builder-pattern

/** Creates a normalized landmark list with one entrry. */
export function createLandmarks(
    x?: number, y?: number, z?: number): NormalizedLandmarkListProto {
  const landmarksProto = new NormalizedLandmarkListProto();
  const landmark = new NormalizedLandmarkProto();
  if (x !== undefined) landmark.setX(x);
  if (y !== undefined) landmark.setY(y);
  if (z !== undefined) landmark.setZ(z);
  landmarksProto.addLandmark(landmark);
  return landmarksProto;
}

/** Creates a world landmark list with one entry. */
export function createWorldLandmarks(
    x?: number, y?: number, z?: number): LandmarkListProto {
  const worldLandmarksProto = new LandmarkListProto();
  const landmark = new LandmarkProto();
  if (x !== undefined) landmark.setX(x);
  if (y !== undefined) landmark.setY(y);
  if (z !== undefined) landmark.setZ(z);
  worldLandmarksProto.addLandmark(landmark);
  return worldLandmarksProto;
}
