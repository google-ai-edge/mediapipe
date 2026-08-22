/**
 * Copyright 2026 The MediaPipe Authors.
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

import {TaskRunnerOptions} from '../../../../tasks/web/core/task_runner_options';

/**
 * Specifies the polarity or mode of a brush stroke.
 * Represents whether the brush stroke is meant to add, subtract, or lasso
 * from the selection.
 */
export enum BrushMode {
  UNSPECIFIED = 0,
  POSITIVE = 1,
  NEGATIVE = 2,
  LASSO = 3,
}

/** Represents a single 2D coordinate point in normalized [0, 1] space. */
export declare interface Point {
  readonly x: number;
  readonly y: number;
}

/** Represents a single user-drawn stroke for interactive segmentation. */
export declare interface Stroke {
  /** The brush mode used for this stroke. */
  readonly brushMode: BrushMode;
  /** The sequence of normalized points forming the stroke. */
  readonly point: readonly Point[];
  /** Whether the stroke is complete. */
  readonly isCompleted: boolean;
}

/** Options to configure the MediaPipe Interactive Segmenter Task. */
export declare interface InteractiveSegmenterOptions extends TaskRunnerOptions {
  /**
   * The canvas element or offscreen canvas to use for WebGL context creation and
   * mask rendering/transfers. If not provided, a canvas is created internally.
   */
  canvas?: HTMLCanvasElement | OffscreenCanvas;
}
