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

import {NormalizedKeypoint} from '../../../../tasks/web/components/containers/keypoint';

/** A Region-Of-Interest (ROI) to represent a region within an image. */
export declare interface RegionOfInterest {
  /** The ROI in keypoint format. */
  keypoint?: NormalizedKeypoint;

  /** The ROI as scribbles over the object that the user wants to segment. */
  scribble?: NormalizedKeypoint[];
}

/** A connection between two landmarks. */
export declare interface Connection {
  start: number;
  end: number;
}

/** Converts a list of connection in array notation to a list of Connections. */
export function convertToConnections(...connections: Array<[number, number]>):
    Connection[] {
  return connections.map(([start, end]) => ({start, end}));
}
