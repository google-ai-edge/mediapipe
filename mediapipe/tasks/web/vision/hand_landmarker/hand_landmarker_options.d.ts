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

import {VisionTaskOptions} from '../../../../tasks/web/vision/core/vision_task_options';

/** Options to configure the MediaPipe HandLandmarker Task */
export declare interface HandLandmarkerOptions extends VisionTaskOptions {
  /**
   * The maximum number of hands can be detected by the HandLandmarker.
   * Defaults to 1.
   */
  numHands?: number|undefined;

  /**
   * The minimum confidence score for the hand detection to be considered
   * successful. Defaults to 0.5.
   */
  minHandDetectionConfidence?: number|undefined;

  /**
   * The minimum confidence score of hand presence score in the hand landmark
   * detection. Defaults to 0.5.
   */
  minHandPresenceConfidence?: number|undefined;

  /**
   * The minimum confidence score for the hand tracking to be considered
   * successful. Defaults to 0.5.
   */
  minTrackingConfidence?: number|undefined;
}
