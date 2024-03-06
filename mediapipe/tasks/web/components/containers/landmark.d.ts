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

/**
 * Normalized Landmark represents a point in 3D space with x, y, z coordinates.
 * x and y are normalized to [0.0, 1.0] by the image width and height
 * respectively. z represents the landmark depth, and the smaller the value the
 * closer the landmark is to the camera. The magnitude of z uses roughly the
 * same scale as x.
 */
export declare interface NormalizedLandmark {
  /** The x coordinates of the normalized landmark. */
  x: number;

  /** The y coordinates of the normalized landmark. */
  y: number;

  /** The z coordinates of the normalized landmark. */
  z: number;

  /** The likelihood of the landmark being visible within the image. */
  visibility: number;
}

/**
 * Landmark represents a point in 3D space with x, y, z coordinates. The
 * landmark coordinates are in meters. z represents the landmark depth,
 * and the smaller the value the closer the world landmark is to the camera.
 */
export declare interface Landmark {
  /** The x coordinates of the landmark. */
  x: number;

  /** The y coordinates of the landmark. */
  y: number;

  /** The z coordinates of the landmark. */
  z: number;

  /** The likelihood of the landmark being visible within the image. */
  visibility: number;
}
