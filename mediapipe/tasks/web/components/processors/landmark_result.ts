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

import {LandmarkList as LandmarkListProto, NormalizedLandmarkList as NormalizedLandmarkListProto} from '../../../../framework/formats/landmark_pb';
import {Landmark, NormalizedLandmark} from '../../../../tasks/web/components/containers/landmark';

/** Converts raw data into a landmark. */
export function convertToLandmarks(proto: NormalizedLandmarkListProto):
    NormalizedLandmark[] {
  const landmarks: NormalizedLandmark[] = [];
  for (const landmark of proto.getLandmarkList()) {
    landmarks.push({
      x: landmark.getX() ?? 0,
      y: landmark.getY() ?? 0,
      z: landmark.getZ() ?? 0,
      visibility: landmark.getVisibility() ?? 0,
    });
  }
  return landmarks;
}

/** Converts raw data into a world landmark. */
export function convertToWorldLandmarks(proto: LandmarkListProto): Landmark[] {
  const worldLandmarks: Landmark[] = [];
  for (const worldLandmark of proto.getLandmarkList()) {
    worldLandmarks.push({
      x: worldLandmark.getX() ?? 0,
      y: worldLandmark.getY() ?? 0,
      z: worldLandmark.getZ() ?? 0,
      visibility: worldLandmark.getVisibility() ?? 0,
    });
  }
  return worldLandmarks;
}
