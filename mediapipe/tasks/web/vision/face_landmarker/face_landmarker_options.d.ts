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

/** Options to configure the MediaPipe FaceLandmarker Task */
export declare interface FaceLandmarkerOptions extends VisionTaskOptions {
  /**
   * The maximum number of faces can be detected by the FaceLandmarker.
   * Defaults to 1.
   */
  numFaces?: number|undefined;

  /**
   * The minimum confidence score for the face detection to be considered
   * successful. Defaults to 0.5.
   */
  minFaceDetectionConfidence?: number|undefined;

  /**
   * The minimum confidence score of face presence score in the face landmark
   * detection. Defaults to 0.5.
   */
  minFacePresenceConfidence?: number|undefined;

  /**
   * The minimum confidence score for the face tracking to be considered
   * successful. Defaults to 0.5.
   */
  minTrackingConfidence?: number|undefined;

  /**
   * Whether FaceLandmarker outputs face blendshapes classification. Face
   * blendshapes are used for rendering the 3D face model.
   */
  outputFaceBlendshapes?: boolean|undefined;

  /**
   * Whether FaceLandmarker outputs facial transformation_matrix. Facial
   * transformation matrix is used to transform the face landmarks in canonical
   * face to the detected face, so that users can apply face effects on the
   * detected landmarks.
   */
  outputFacialTransformationMatrixes?: boolean|undefined;
}
