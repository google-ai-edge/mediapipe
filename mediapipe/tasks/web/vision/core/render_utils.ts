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


/** Helper function to draw a confidence mask */
export function drawConfidenceMask(
    ctx: CanvasRenderingContext2D, image: Float32Array, width: number,
    height: number): void {
  const uint8Array = new Uint8ClampedArray(width * height * 4);
  for (let i = 0; i < image.length; i++) {
    uint8Array[4 * i] = 128;
    uint8Array[4 * i + 1] = 0;
    uint8Array[4 * i + 2] = 0;
    uint8Array[4 * i + 3] = image[i] * 255;
  }
  ctx.putImageData(new ImageData(uint8Array, width, height), 0, 0);
}

/**
 * Helper function to draw a category mask. For GPU, we only have F32Arrays
 * for now.
 */
export function drawCategoryMask(
    ctx: CanvasRenderingContext2D, image: Uint8Array|Float32Array,
    width: number, height: number): void {
  const rgbaArray = new Uint8ClampedArray(width * height * 4);
  const isFloatArray = image instanceof Float32Array;
  for (let i = 0; i < image.length; i++) {
    const colorIndex = isFloatArray ? Math.round(image[i] * 255) : image[i];
    const color = COLOR_MAP[colorIndex % COLOR_MAP.length];
    rgbaArray[4 * i] = color[0];
    rgbaArray[4 * i + 1] = color[1];
    rgbaArray[4 * i + 2] = color[2];
    rgbaArray[4 * i + 3] = color[3];
  }
  ctx.putImageData(new ImageData(rgbaArray, width, height), 0, 0);
}
