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

import {BoundingBox} from '../../../../tasks/web/components/containers/bounding_box';
import {NormalizedLandmark} from '../../../../tasks/web/components/containers/landmark';
import {Connection} from '../../../../tasks/web/vision/core/types';

/**
 * A user-defined callback to take input data and map it to a custom output
 * value.
 */
export type Callback<I, O> = (input: I) => O;

/** Data that a user can use to specialize drawing options. */
export declare interface LandmarkData {
  index?: number;
  from?: NormalizedLandmark;
  to?: NormalizedLandmark;
}

/**
 * Options for customizing the drawing routines
 */
export declare interface DrawingOptions {
  /** The color that is used to draw the shape. Defaults to white. */
  color?: string|CanvasGradient|CanvasPattern|
      Callback<LandmarkData, string|CanvasGradient|CanvasPattern>;
  /**
   * The color that is used to fill the shape. Defaults to `.color` (or black
   * if color is not set).
   */
  fillColor?: string|CanvasGradient|CanvasPattern|
      Callback<LandmarkData, string|CanvasGradient|CanvasPattern>;
  /** The width of the line boundary of the shape. Defaults to 4. */
  lineWidth?: number|Callback<LandmarkData, number>;
  /** The radius of location marker. Defaults to 6. */
  radius?: number|Callback<LandmarkData, number>;
}

/**
 * This will be merged with user supplied options.
 */
const DEFAULT_OPTIONS: DrawingOptions = {
  color: 'white',
  lineWidth: 4,
  radius: 6
};

/** Merges the user's options with the default options. */
function addDefaultOptions(style?: DrawingOptions): DrawingOptions {
  style = style || {};
  return {
    ...DEFAULT_OPTIONS,
    ...{fillColor: style.color},
    ...style,
  };
}

/**
 * Resolve the value from `value`. Invokes `value` with `data` if it is a
 * function.
 */
function resolve<O, I>(value: O|Callback<I, O>, data: I): O {
  return value instanceof Function ? value(data) : value;
}

/** Helper class to visualize the result of a MediaPipe Vision task. */
export class DrawingUtils {
  /**
   * Creates a new DrawingUtils class.
   *
   * @param ctx The canvas to render onto.
   */
  constructor(private readonly ctx: CanvasRenderingContext2D) {}

  /**
   * Restricts a number between two endpoints (order doesn't matter).
   *
   * @param x The number to clamp.
   * @param x0 The first boundary.
   * @param x1 The second boundary.
   * @return The clamped value.
   */
  static clamp(x: number, x0: number, x1: number): number {
    const lo = Math.min(x0, x1);
    const hi = Math.max(x0, x1);
    return Math.max(lo, Math.min(hi, x));
  }

  /**
   * Linearly interpolates a value between two points, clamping that value to
   * the endpoints.
   *
   * @param x The number to interpolate.
   * @param x0 The x coordinate of the start value.
   * @param x1 The x coordinate of the end value.
   * @param y0 The y coordinate of the start value.
   * @param y1 The y coordinate of the end value.
   * @return The interpolated value.
   */
  static lerp(x: number, x0: number, x1: number, y0: number, y1: number):
      number {
    const out =
        y0 * (1 - (x - x0) / (x1 - x0)) + y1 * (1 - (x1 - x) / (x1 - x0));
    return DrawingUtils.clamp(out, y0, y1);
  }

  /**
   * Draws circles onto the provided landmarks.
   *
   * @param landmarks The landmarks to draw.
   * @param style The style to visualize the landmarks.
   */
  drawLandmarks(landmarks?: NormalizedLandmark[], style?: DrawingOptions):
      void {
    if (!landmarks) {
      return;
    }
    const ctx = this.ctx;
    const options = addDefaultOptions(style);
    ctx.save();
    const canvas = ctx.canvas;
    let index = 0;
    for (const landmark of landmarks) {
      // All of our points are normalized, so we need to scale the unit canvas
      // to match our actual canvas size.
      ctx.fillStyle = resolve(options.fillColor!, {index, from: landmark});
      ctx.strokeStyle = resolve(options.color!, {index, from: landmark});
      ctx.lineWidth = resolve(options.lineWidth!, {index, from: landmark});

      const circle = new Path2D();
      // Decrease the size of the arc to compensate for the scale()
      circle.arc(
          landmark.x * canvas.width, landmark.y * canvas.height,
          resolve(options.radius!, {index, from: landmark}), 0, 2 * Math.PI);
      ctx.fill(circle);
      ctx.stroke(circle);
      ++index;
    }
    ctx.restore();
  }

  /**
   * Draws lines between landmarks (given a connection graph).
   *
   * @param landmarks The landmarks to draw.
   * @param connections The connections array that contains the start and the
   *     end indices for the connections to draw.
   * @param style The style to visualize the landmarks.
   */
  drawConnectors(
      landmarks?: NormalizedLandmark[], connections?: Connection[],
      style?: DrawingOptions): void {
    if (!landmarks || !connections) {
      return;
    }
    const ctx = this.ctx;
    const options = addDefaultOptions(style);
    ctx.save();
    const canvas = ctx.canvas;
    let index = 0;
    for (const connection of connections) {
      ctx.beginPath();
      const from = landmarks[connection.start];
      const to = landmarks[connection.end];
      if (from && to) {
        ctx.strokeStyle = resolve(options.color!, {index, from, to});
        ctx.lineWidth = resolve(options.lineWidth!, {index, from, to});
        ctx.moveTo(from.x * canvas.width, from.y * canvas.height);
        ctx.lineTo(to.x * canvas.width, to.y * canvas.height);
      }
      ++index;
      ctx.stroke();
    }
    ctx.restore();
  }

  /**
   * Draws a bounding box.
   *
   * @param boundingBox The bounding box to draw.
   * @param style The style to visualize the boundin box.
   */
  drawBoundingBox(boundingBox: BoundingBox, style?: DrawingOptions): void {
    const ctx = this.ctx;
    const options = addDefaultOptions(style);
    ctx.save();
    ctx.beginPath();
    ctx.lineWidth = resolve(options.lineWidth!, {});
    ctx.strokeStyle = resolve(options.color!, {});
    ctx.fillStyle = resolve(options.fillColor!, {});
    ctx.moveTo(boundingBox.originX, boundingBox.originY);
    ctx.lineTo(boundingBox.originX + boundingBox.width, boundingBox.originY);
    ctx.lineTo(
        boundingBox.originX + boundingBox.width,
        boundingBox.originY + boundingBox.height);
    ctx.lineTo(boundingBox.originX, boundingBox.originY + boundingBox.height);
    ctx.lineTo(boundingBox.originX, boundingBox.originY);
    ctx.stroke();
    ctx.fill();
    ctx.restore();
  }
}


