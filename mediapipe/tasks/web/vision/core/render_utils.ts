/** @fileoverview Utility functions used in the vision demos. */

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

import {MPImageChannelConverter} from '../../../../tasks/web/vision/core/image';

// Pre-baked color table for a maximum of 12 classes.
const CM_ALPHA = 128;
const COLOR_MAP: Array<[number, number, number, number]> = [
  [0, 0, 0, CM_ALPHA],        // class 0 is BG = transparent
  [255, 0, 0, CM_ALPHA],      // class 1 is red
  [0, 255, 0, CM_ALPHA],      // class 2 is light green
  [0, 0, 255, CM_ALPHA],      // class 3 is blue
  [255, 255, 0, CM_ALPHA],    // class 4 is yellow
  [255, 0, 255, CM_ALPHA],    // class 5 is light purple / magenta
  [0, 255, 255, CM_ALPHA],    // class 6 is light blue / aqua
  [128, 128, 128, CM_ALPHA],  // class 7 is gray
  [255, 128, 0, CM_ALPHA],    // class 8 is orange
  [128, 0, 255, CM_ALPHA],    // class 9 is dark purple
  [0, 128, 0, CM_ALPHA],      // class 10 is dark green
  [255, 255, 255, CM_ALPHA]   // class 11 is white; could do black instead?
];

/** The color converter we use in our demos. */
export const RENDER_UTIL_CONVERTER: MPImageChannelConverter = {
  floatToRGBAConverter: v => [128, 0, 0, v * 255],
  uint8ToRGBAConverter: v => COLOR_MAP[v % COLOR_MAP.length],
};
