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

import {Detection as DetectionProto} from '../../../../framework/formats/detection_pb';
import {Detection} from '../../../../tasks/web/components/containers/detection_result';

const DEFAULT_CATEGORY_INDEX = -1;

/** Converts a Detection proto into a Detection object. */
export function convertFromDetectionProto(source: DetectionProto): Detection {
  const scores = source.getScoreList();
  const indexes = source.getLabelIdList();
  const labels = source.getLabelList();
  const displayNames = source.getDisplayNameList();

  const detection: Detection = {categories: [], keypoints: []};
  for (let i = 0; i < scores.length; i++) {
    detection.categories.push({
      score: scores[i],
      index: indexes[i] ?? DEFAULT_CATEGORY_INDEX,
      categoryName: labels[i] ?? '',
      displayName: displayNames[i] ?? '',
    });
  }

  const boundingBox = source.getLocationData()?.getBoundingBox();
  if (boundingBox) {
    detection.boundingBox = {
      originX: boundingBox.getXmin() ?? 0,
      originY: boundingBox.getYmin() ?? 0,
      width: boundingBox.getWidth() ?? 0,
      height: boundingBox.getHeight() ?? 0,
      angle: 0.0,
    };
  }

  if (source.getLocationData()?.getRelativeKeypointsList().length) {
    for (const keypoint of
             source.getLocationData()!.getRelativeKeypointsList()) {
      detection.keypoints.push({
        x: keypoint.getX() ?? 0.0,
        y: keypoint.getY() ?? 0.0,
        score: keypoint.getScore() ?? 0.0,
        label: keypoint.getKeypointLabel() ?? '',
      });
    }
  }

  return detection;
}
