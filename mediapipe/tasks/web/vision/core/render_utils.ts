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
