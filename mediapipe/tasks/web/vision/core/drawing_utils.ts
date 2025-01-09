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
import {CategoryMaskShaderContext, CategoryToColorMap, RGBAColor} from '../../../../tasks/web/vision/core/drawing_utils_category_mask';
import {ConfidenceMaskShaderContext} from '../../../../tasks/web/vision/core/drawing_utils_confidence_mask';
import {MPImageShaderContext} from '../../../../tasks/web/vision/core/image_shader_context';
import {MPMask} from '../../../../tasks/web/vision/core/mask';
import {Connection} from '../../../../tasks/web/vision/core/types';
import {ImageSource} from '../../../../web/graph_runner/graph_runner';

/**
 * A user-defined callback to take input data and map it to a custom output
 * value.
 */
export type Callback<I, O> = (input: I) => O;

// Used in public API
export {type ImageSource};

/** Data that a user can use to specialize drawing options. */
export declare interface LandmarkData {
  index?: number;
  from?: NormalizedLandmark;
  to?: NormalizedLandmark;
}

/** A color map with 22 classes. Used in our demos. */
export const DEFAULT_CATEGORY_TO_COLOR_MAP = [
  [0, 0, 0, 0],          // class 0 is BG = transparent
  [255, 0, 0, 255],      // class 1 is red
  [0, 255, 0, 255],      // class 2 is light green
  [0, 0, 255, 255],      // class 3 is blue
  [255, 255, 0, 255],    // class 4 is yellow
  [255, 0, 255, 255],    // class 5 is light purple / magenta
  [0, 255, 255, 255],    // class 6 is light blue / aqua
  [128, 128, 128, 255],  // class 7 is gray
  [255, 100, 0, 255],    // class 8 is dark orange
  [128, 0, 255, 255],    // class 9 is dark purple
  [0, 150, 0, 255],      // class 10 is green
  [255, 255, 255, 255],  // class 11 is white
  [255, 105, 180, 255],  // class 12 is pink
  [255, 150, 0, 255],    // class 13 is orange
  [255, 250, 224, 255],  // class 14 is light yellow
  [148, 0, 211, 255],    // class 15 is dark violet
  [0, 100, 0, 255],      // class 16 is dark green
  [0, 0, 128, 255],      // class 17 is navy blue
  [165, 42, 42, 255],    // class 18 is brown
  [64, 224, 208, 255],   // class 19 is turquoise
  [255, 218, 185, 255],  // class 20 is peach
  [192, 192, 192, 255],  // class 21 is silver
];

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

export {type RGBAColor, type CategoryToColorMap};

/** Helper class to visualize the result of a MediaPipe Vision task. */
export class DrawingUtils {
  private categoryMaskShaderContext?: CategoryMaskShaderContext;
  private confidenceMaskShaderContext?: ConfidenceMaskShaderContext;
  private convertToWebGLTextureShaderContext?: MPImageShaderContext;
  private readonly context2d?: CanvasRenderingContext2D|
      OffscreenCanvasRenderingContext2D;
  private readonly contextWebGL?: WebGL2RenderingContext;

  /**
   * Creates a new DrawingUtils class.
   *
   * @param gpuContext The WebGL canvas rendering context to render into. If
   *     your Task is using a GPU delegate, the context must be obtained from
   * its canvas (provided via `setOptions({ canvas: .. })`).
   */
  constructor(gpuContext: WebGL2RenderingContext);
  /**
   * Creates a new DrawingUtils class.
   *
   * @param cpuContext The 2D canvas rendering context to render into. If
   *     you are rendering GPU data you must also provide `gpuContext` to allow
   *     for data conversion.
   * @param gpuContext A WebGL canvas that is used for GPU rendering and for
   *     converting GPU to CPU data. If your Task is using a GPU delegate, the
   *     context must be obtained from  its canvas (provided via
   *     `setOptions({ canvas: .. })`).
   */
  constructor(
      cpuContext: CanvasRenderingContext2D|OffscreenCanvasRenderingContext2D,
      gpuContext?: WebGL2RenderingContext);
  constructor(
      cpuOrGpuGontext: CanvasRenderingContext2D|
      OffscreenCanvasRenderingContext2D|WebGL2RenderingContext,
      gpuContext?: WebGL2RenderingContext) {
    if (cpuOrGpuGontext instanceof CanvasRenderingContext2D ||
        cpuOrGpuGontext instanceof OffscreenCanvasRenderingContext2D) {
      this.context2d = cpuOrGpuGontext;
      this.contextWebGL = gpuContext;
    } else {
      this.contextWebGL = cpuOrGpuGontext;
    }
  }

  /**
   * Restricts a number between two endpoints (order doesn't matter).
   *
   * @export
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
   * @export
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

  private getCanvasRenderingContext(): CanvasRenderingContext2D
      |OffscreenCanvasRenderingContext2D {
    if (!this.context2d) {
      throw new Error(
          'CPU rendering requested but CanvasRenderingContext2D not provided.');
    }
    return this.context2d;
  }

  private getWebGLRenderingContext(): WebGL2RenderingContext {
    if (!this.contextWebGL) {
      throw new Error(
          'GPU rendering requested but WebGL2RenderingContext not provided.');
    }
    return this.contextWebGL;
  }

  private getCategoryMaskShaderContext(): CategoryMaskShaderContext {
    if (!this.categoryMaskShaderContext) {
      this.categoryMaskShaderContext = new CategoryMaskShaderContext();
    }
    return this.categoryMaskShaderContext;
  }

  private getConfidenceMaskShaderContext(): ConfidenceMaskShaderContext {
    if (!this.confidenceMaskShaderContext) {
      this.confidenceMaskShaderContext = new ConfidenceMaskShaderContext();
    }
    return this.confidenceMaskShaderContext;
  }

  /**
   * Draws circles onto the provided landmarks.
   *
   * This method can only be used when `DrawingUtils` is initialized with a
   * `CanvasRenderingContext2D`.
   *
   * @export
   * @param landmarks The landmarks to draw.
   * @param style The style to visualize the landmarks.
   */
  drawLandmarks(landmarks?: NormalizedLandmark[], style?: DrawingOptions):
      void {
    if (!landmarks) {
      return;
    }
    const ctx = this.getCanvasRenderingContext();
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
   * This method can only be used when `DrawingUtils` is initialized with a
   * `CanvasRenderingContext2D`.
   *
   * @export
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
    const ctx = this.getCanvasRenderingContext();
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
   * This method can only be used when `DrawingUtils` is initialized with a
   * `CanvasRenderingContext2D`.
   *
   * @export
   * @param boundingBox The bounding box to draw.
   * @param style The style to visualize the boundin box.
   */
  drawBoundingBox(boundingBox: BoundingBox, style?: DrawingOptions): void {
    const ctx = this.getCanvasRenderingContext();
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

  /** Draws a category mask on a CanvasRenderingContext2D. */
  private drawCategoryMask2D(
      mask: MPMask, background: RGBAColor|ImageSource,
      categoryToColorMap: Map<number, RGBAColor>|RGBAColor[]): void {
    // Use the WebGL renderer to draw result on our internal canvas.
    const gl = this.getWebGLRenderingContext();
    this.runWithWebGLTexture(mask, texture => {
      this.drawCategoryMaskWebGL(texture, background, categoryToColorMap);
      // Draw the result on the user canvas.
      const ctx = this.getCanvasRenderingContext();
      ctx.drawImage(gl.canvas, 0, 0, ctx.canvas.width, ctx.canvas.height);
    });
  }

  /** Draws a category mask on a WebGL2RenderingContext2D. */
  private drawCategoryMaskWebGL(
      categoryTexture: WebGLTexture, background: RGBAColor|ImageSource,
      categoryToColorMap: Map<number, RGBAColor>|RGBAColor[]): void {
    const shaderContext = this.getCategoryMaskShaderContext();
    const gl = this.getWebGLRenderingContext();
    const backgroundImage = Array.isArray(background) ?
        new ImageData(new Uint8ClampedArray(background), 1, 1) :
        background;

    shaderContext.run(gl, /* flipTexturesVertically= */ true, () => {
      shaderContext.bindAndUploadTextures(
          categoryTexture, backgroundImage, categoryToColorMap);
      gl.clearColor(0, 0, 0, 0);
      gl.clear(gl.COLOR_BUFFER_BIT);
      gl.drawArrays(gl.TRIANGLE_FAN, 0, 4);
      shaderContext.unbindTextures();
    });
  }

  /**
   * Draws a category mask using the provided category-to-color mapping.
   *
   * @export
   * @param mask A category mask that was returned from a segmentation task.
   * @param categoryToColorMap A map that maps category indices to RGBA
   *     values. You must specify a map entry for each category.
   * @param background A color or image to use as the background. Defaults to
   *     black.
   */
  drawCategoryMask(
      mask: MPMask, categoryToColorMap: Map<number, RGBAColor>,
      background?: RGBAColor|ImageSource): void;
  /**
   * Draws a category mask using the provided color array.
   *
   * @export
   * @param mask A category mask that was returned from a segmentation task.
   * @param categoryToColorMap An array that maps indices to RGBA values. The
   *     array's indices must correspond to the category indices of the model
   *     and an entry must be provided for each category.
   * @param background A color or image to use as the background. Defaults to
   *     black.
   */
  drawCategoryMask(
      mask: MPMask, categoryToColorMap: RGBAColor[],
      background?: RGBAColor|ImageSource): void;
  /** @export */
  drawCategoryMask(
      mask: MPMask, categoryToColorMap: CategoryToColorMap,
      background: RGBAColor|ImageSource = [0, 0, 0, 255]): void {
    if (this.context2d) {
      this.drawCategoryMask2D(mask, background, categoryToColorMap);
    } else {
      this.drawCategoryMaskWebGL(
          mask.getAsWebGLTexture(), background, categoryToColorMap);
    }
  }

  /**
   * Converts the given mask to a WebGLTexture and runs the callback. Cleans
   * up any new resources after the callback finished executing.
   */
  private runWithWebGLTexture(
      mask: MPMask, callback: (texture: WebGLTexture) => void): void {
    if (!mask.hasWebGLTexture()) {
      // Re-create the MPMask but use our the WebGL canvas so we can draw the
      // texture directly.
      const data = mask.hasFloat32Array() ? mask.getAsFloat32Array() :
                                            mask.getAsUint8Array();
      this.convertToWebGLTextureShaderContext =
          this.convertToWebGLTextureShaderContext ?? new MPImageShaderContext();
      const gl = this.getWebGLRenderingContext();

      const convertedMask = new MPMask(
          [data],
          mask.interpolateValues,
          /* ownsWebGlTexture= */ false,
          gl.canvas,
          this.convertToWebGLTextureShaderContext,
          mask.width,
          mask.height,
      );
      callback(convertedMask.getAsWebGLTexture());
      convertedMask.close();
    } else {
      callback(mask.getAsWebGLTexture());
    }
  }

  /** Draws a confidence mask on a WebGL2RenderingContext2D. */
  private drawConfidenceMaskWebGL(
      maskTexture: WebGLTexture, defaultTexture: RGBAColor|ImageSource,
      overlayTexture: RGBAColor|ImageSource): void {
    const gl = this.getWebGLRenderingContext();
    const shaderContext = this.getConfidenceMaskShaderContext();
    const defaultImage = Array.isArray(defaultTexture) ?
        new ImageData(new Uint8ClampedArray(defaultTexture), 1, 1) :
        defaultTexture;
    const overlayImage = Array.isArray(overlayTexture) ?
        new ImageData(new Uint8ClampedArray(overlayTexture), 1, 1) :
        overlayTexture;

    shaderContext.run(gl, /* flipTexturesVertically= */ true, () => {
      shaderContext.bindAndUploadTextures(
          defaultImage, overlayImage, maskTexture);
      gl.clearColor(0, 0, 0, 0);
      gl.clear(gl.COLOR_BUFFER_BIT);
      gl.drawArrays(gl.TRIANGLE_FAN, 0, 4);
      gl.bindTexture(gl.TEXTURE_2D, null);
      shaderContext.unbindTextures();
    });
  }

  /** Draws a confidence mask on a CanvasRenderingContext2D. */
  private drawConfidenceMask2D(
      mask: MPMask, defaultTexture: RGBAColor|ImageSource,
      overlayTexture: RGBAColor|ImageSource): void {
    // Use the WebGL renderer to draw result on our internal canvas.
    const gl = this.getWebGLRenderingContext();
    this.runWithWebGLTexture(mask, texture => {
      this.drawConfidenceMaskWebGL(texture, defaultTexture, overlayTexture);
      // Draw the result on the user canvas.
      const ctx = this.getCanvasRenderingContext();
      ctx.drawImage(gl.canvas, 0, 0, ctx.canvas.width, ctx.canvas.height);
    });
  }

  /**
   * Blends two images using the provided confidence mask.
   *
   * If you are using an `ImageData` or `HTMLImageElement` as your data source
   * and drawing the result onto a `WebGL2RenderingContext`, this method uploads
   * the image data to the GPU. For still image input that gets re-used every
   * frame, you can reduce the cost of re-uploading these images by passing a
   * `HTMLCanvasElement` instead.
   *
   * @export
   * @param mask A confidence mask that was returned from a segmentation task.
   * @param defaultTexture An image or a four-channel color that will be used
   *     when confidence values are low.
   * @param overlayTexture An image or four-channel color that will be used when
   *     confidence values are high.
   */
  drawConfidenceMask(
      mask: MPMask, defaultTexture: RGBAColor|ImageSource,
      overlayTexture: RGBAColor|ImageSource): void {
    if (this.context2d) {
      this.drawConfidenceMask2D(mask, defaultTexture, overlayTexture);
    } else {
      this.drawConfidenceMaskWebGL(
          mask.getAsWebGLTexture(), defaultTexture, overlayTexture);
    }
  }
  /**
   * Frees all WebGL resources held by this class.
   * @export
   */
  close(): void {
    this.categoryMaskShaderContext?.close();
    this.categoryMaskShaderContext = undefined;
    this.confidenceMaskShaderContext?.close();
    this.confidenceMaskShaderContext = undefined;
    this.convertToWebGLTextureShaderContext?.close();
    this.convertToWebGLTextureShaderContext = undefined;
  }
}


