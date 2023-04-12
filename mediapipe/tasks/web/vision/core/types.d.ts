/**
 * Copyright 2023 The MediaPipe Authors. All Rights Reserved.
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

/**
 * The segmentation tasks return the segmentation either as a WebGLTexture (when
 * the output is on GPU) or as a typed JavaScript arrays for CPU-based
 * category or confidence masks. `Uint8ClampedArray`s are used to represent
 * CPU-based category masks and `Float32Array`s are used for CPU-based
 * confidence masks.
 */
export type SegmentationMask = Uint8ClampedArray|Float32Array|WebGLTexture;

/**
 * A callback that receives the computed masks from the segmentation tasks. The
 * callback either receives a single element array with a category mask (as a
 * `[Uint8ClampedArray]`) or multiple confidence masks (as a `Float32Array[]`).
 * The returned data is only valid for the duration of the callback. If
 * asynchronous processing is needed, all data needs to be copied before the
 * callback returns.
 */
export type SegmentationMaskCallback =
    (masks: SegmentationMask[], width: number, height: number) => void;

/**
 * A callback that receives an `ImageData` object from a Vision task. The
 * lifetime of the underlying data is limited to the duration of the callback.
 * If asynchronous processing is needed, all data needs to be copied before the
 * callback returns.
 *
 * The `WebGLTexture` output type is reserved for future usage.
 */
export type ImageCallback =
    (image: ImageData|WebGLTexture, width: number, height: number) => void;

/** A Region-Of-Interest (ROI) to represent a region within an image. */
export declare interface RegionOfInterest {
  /** The ROI in keypoint format. */
  keypoint: NormalizedKeypoint;
}

/** A connection between two landmarks. */
export declare interface Connection {
  start: number;
  end: number;
}
