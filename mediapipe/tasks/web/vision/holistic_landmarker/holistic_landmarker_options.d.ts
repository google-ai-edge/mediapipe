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

import {VisionTaskOptions} from '../../../../tasks/web/vision/core/vision_task_options';

/** Options to configure the MediaPipe HolisticLandmarker Task */
export declare interface HolisticLandmarkerOptions extends VisionTaskOptions {
  /**
   * The minimum confidence score for the face detection to be considered
   * successful. Defaults to 0.5.
   */
  minFaceDetectionConfidence?: number|undefined;

  /**
   * The minimum non-maximum-suppression threshold for face detection to be
   * considered overlapped. Defaults to 0.3.
   */
  minFaceSuppressionThreshold?: number|undefined;

  /**
   * The minimum confidence score of face presence score in the face landmarks
   * detection. Defaults to 0.5.
   */
  minFacePresenceConfidence?: number|undefined;

  /**
   * Whether FaceLandmarker outputs face blendshapes classification. Face
   * blendshapes are used for rendering the 3D face model.
   */
  outputFaceBlendshapes?: boolean|undefined;

  /**
   * The minimum confidence score for the pose detection to be considered
   * successful. Defaults to 0.5.
   */
  minPoseDetectionConfidence?: number|undefined;

  /**
   * The minimum non-maximum-suppression threshold for pose detection to be
   * considered overlapped. Defaults to 0.3.
   */
  minPoseSuppressionThreshold?: number|undefined;

  /**
   * The minimum confidence score of pose presence score in the pose landmarks
   * detection. Defaults to 0.5.
   */
  minPosePresenceConfidence?: number|undefined;

  /** Whether to output segmentation masks. Defaults to false. */
  outputPoseSegmentationMasks?: boolean|undefined;

  /**
   * The minimum confidence score of hand presence score in the hand landmarks
   * detection. Defaults to 0.5.
   */
  minHandLandmarksConfidence?: number|undefined;
}
