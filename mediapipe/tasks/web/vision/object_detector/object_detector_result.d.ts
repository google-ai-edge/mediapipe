/**
 * Copyright 2022 The MediaPipe Authors. All Rights Reserved.
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

import {Category} from '../../../../tasks/web/components/containers/category';

export {Category};

/** An integer bounding box, axis aligned. */
export declare interface BoundingBox {
  /** The X coordinate of the top-left corner, in pixels. */
  originX: number;
  /** The Y coordinate of the top-left corner, in pixels. */
  originY: number;
  /** The width of the bounding box, in pixels. */
  width: number;
  /** The height of the bounding box, in pixels. */
  height: number;
}

/** Represents one object detected by the `ObjectDetector`. */
export declare interface Detection {
  /** A list of `Category` objects. */
  categories: Category[];

  /** The bounding box of the detected objects. */
  boundingBox?: BoundingBox;
}
