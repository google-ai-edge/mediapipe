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

import {MPMask} from '../../../../tasks/web/vision/core/mask';

/** The output result of InteractiveSegmenter. */
export class InteractiveSegmenterResult {
  constructor(
      /**
       * Multiple masks represented as `Float32Array` or `WebGLTexture`-backed
       * `MPImage`s where, for each mask, each pixel represents the prediction
       * confidence, usually in the [0, 1] range.
       */
      readonly confidenceMasks?: MPMask[],
      /**
       * A category mask represented as a `Uint8ClampedArray` or
       * `WebGLTexture`-backed `MPImage` where each pixel represents the class
       * which the pixel in the original image was predicted to belong to.
       */
      readonly categoryMask?: MPMask) {}

  /** Frees the resources held by the category and confidence masks. */
  close(): void {
    this.confidenceMasks?.forEach(m => {
      m.close();
    });
    this.categoryMask?.close();
  }
}
