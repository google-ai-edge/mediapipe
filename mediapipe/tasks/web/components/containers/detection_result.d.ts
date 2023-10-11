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

import {BoundingBox} from '../../../../tasks/web/components/containers/bounding_box';
import {Category} from '../../../../tasks/web/components/containers/category';
import {NormalizedKeypoint} from '../../../../tasks/web/components/containers/keypoint';

/** Represents one detection by a detection task. */
export declare interface Detection {
  /** A list of `Category` objects. */
  categories: Category[];

  /** The bounding box of the detected objects. */
  boundingBox?: BoundingBox;

  /**
   * List of keypoints associated with the detection. Keypoints represent
   * interesting points related to the detection. For example, the keypoints
   * represent the eye, ear and mouth from face detection model. Or in the
   * template matching detection, e.g. KNIFT, they can represent the feature
   * points for template matching. Contains an empty list if no keypoints are
   * detected.
   */
  keypoints: NormalizedKeypoint[];
}

/** Detection results of a model. */
export declare interface DetectionResult {
  /** A list of Detections. */
  detections: Detection[];
}
