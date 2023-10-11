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

/**
 * A keypoint, defined by the coordinates (x, y), normalized by the image
 * dimensions.
 */
export declare interface NormalizedKeypoint {
  /** X in normalized image coordinates. */
  x: number;

  /** Y in normalized image coordinates. */
  y: number;

  /** Optional label of the keypoint. */
  label?: string;

  /** Optional score of the keypoint. */
  score?: number;
}
