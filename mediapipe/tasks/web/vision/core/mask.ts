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

import {assertExists, MPImageShaderContext} from '../../../../tasks/web/vision/core/image_shader_context';
import {isIOS} from '../../../../web/graph_runner/platform_utils';

/** Number of instances a user can keep alive before we raise a warning. */
const INSTANCE_COUNT_WARNING_THRESHOLD = 250;

/**
 * Samples the single-channel mask and writes grayscale RGBA (`vec4(m)`) so
 * the ImageBitmap can be used as a VideoFrame / compositing mask without a
 * GPU→CPU readback.
 */
const MASK_TO_BITMAP_FRAGMENT_SHADER = `
  precision mediump float;
  varying vec2 vTex;
  uniform sampler2D inputTexture;
  void main() {
    float m = texture2D(inputTexture, vTex).r;
    gl_FragColor = vec4(m, m, m, m);
  }
 `;

/** Shader used only for `MPMask.getAsImageBitmap()`. */
class MPMaskBitmapShaderContext extends MPImageShaderContext {
  protected override getFragmentShader(): string {
    return MASK_TO_BITMAP_FRAGMENT_SHADER;
  }
}

/** Bitmap shaders keyed by GL context so video pipelines compile once. */
const bitmapShaderContexts =
    new WeakMap<WebGL2RenderingContext, MPMaskBitmapShaderContext>();

/** The underlying type of the image. */
enum MPMaskType {
  /** Represents the native `UInt8Array` type. */
  UINT8_ARRAY,
  /** Represents the native `Float32Array` type.  */
  FLOAT32_ARRAY,
  /** Represents the native `WebGLTexture` type. */
  WEBGL_TEXTURE,
  /** Represents the native `ImageBitmap` type. */
  IMAGE_BITMAP,
}

/** The supported mask formats. For internal usage. */
export type MPMaskContainer =
    Uint8Array|Float32Array|WebGLTexture|ImageBitmap;



/**
 * The wrapper class for MediaPipe segmentation masks.
 *
 * Masks are stored as `Uint8Array`, `Float32Array`, `WebGLTexture` or
 * `ImageBitmap` objects. You can convert the underlying type to any other type
 * by passing the desired type to `getAs...()`. As type conversions can be
 * expensive, it is recommended to limit these conversions. You can verify what
 * underlying types are already available by invoking `has...()`.
 *
 * Masks that are returned from a MediaPipe Tasks are owned by by the
 * underlying C++ Task. If you need to extend the lifetime of these objects,
 * you can invoke the `clone()` method. To free up the resources obtained
 * during any clone or type conversion operation, it is important to invoke
 * `close()` on the `MPMask` instance.
 *
 * Converting to `ImageBitmap` requires that the MediaPipe task is initialized
 * with an `OffscreenCanvas`. As we require WebGL2 support, this places some
 * limitations on Browser support as outlined here:
 * https://developer.mozilla.org/en-US/docs/Web/API/OffscreenCanvas/getContext
 */
export class MPMask {
  private gl?: WebGL2RenderingContext;

  /**
   * A counter to track the number of instances of MPMask that own resources.
   * This is used to raise a warning if the user does not close the instances.
   */
  private static instancesBeforeWarning = INSTANCE_COUNT_WARNING_THRESHOLD;

  /** The format used to write pixel values from textures. */
  private static texImage2DFormat?: GLenum;

  /**
   * @param containers The data source for this mask as a `WebGLTexture`,
   *     `Unit8Array` or `Float32Array`. Multiple sources of the same data can
   *     be provided to reduce conversions.
   * @param interpolateValues If enabled, uses `gl.LINEAR` instead of
   *     `gl.NEAREST` to interpolate between mask values.
   * @param ownsWebGLTexture Whether the MPMask should take ownership of the
   *     `WebGLTexture` and free it when closed.
   * @param canvas The canvas to use for rendering and conversion. Must be the
   *     same canvas for any WebGL resources.
   * @param shaderContext A shader context that is shared between all masks from
   *     a single task.
   * @param width The width of the mask.
   * @param height The height of the mask.
   * @param ownsImageBitmap Whether the MPMask should take ownership of an
   *     `ImageBitmap` and free it when closed.
   * @hideconstructor
   */
  constructor(
      private readonly containers: MPMaskContainer[],
      readonly interpolateValues: boolean,
      private ownsWebGLTexture: boolean,
      /** Returns the canvas element that the mask is bound to. */
      readonly canvas: HTMLCanvasElement|OffscreenCanvas|undefined,
      private shaderContext: MPImageShaderContext|undefined,
      /** Returns the width of the mask. */
      readonly width: number,
      /** Returns the height of the mask. */
      readonly height: number,
      private ownsImageBitmap = false,
  ) {
    if (this.ownsWebGLTexture || this.ownsImageBitmap) {
      --MPMask.instancesBeforeWarning;
      if (MPMask.instancesBeforeWarning === 0) {
        console.error(
            'You seem to be creating MPMask instances without invoking ' +
            '.close(). This leaks resources.');
      }
    }
  }

  /**
   * Returns whether this `MPMask` contains a mask of type `Uint8Array`.
   * @export
   */
  hasUint8Array(): boolean {
    return !!this.getContainer(MPMaskType.UINT8_ARRAY);
  }

  /**
   * Returns whether this `MPMask` contains a mask of type `Float32Array`.
   * @export
   */
  hasFloat32Array(): boolean {
    return !!this.getContainer(MPMaskType.FLOAT32_ARRAY);
  }

  /**
   * Returns whether this `MPMask` contains a mask of type `WebGLTexture`.
   * @export
   */
  hasWebGLTexture(): boolean {
    return !!this.getContainer(MPMaskType.WEBGL_TEXTURE);
  }

  /**
   * Returns whether this `MPMask` contains a mask of type `ImageBitmap`.
   * @export
   */
  hasImageBitmap(): boolean {
    return !!this.getContainer(MPMaskType.IMAGE_BITMAP);
  }

  /**
   * Returns the underlying mask as a Uint8Array`. Note that this involves an
   * expensive GPU to CPU transfer if the current mask is only available as a
   * `WebGLTexture`.
   *
   * @export
   * @return The current data as a Uint8Array.
   */
  getAsUint8Array(): Uint8Array {
    return this.convertToUint8Array();
  }

  /**
   * Returns the underlying mask as a single channel `Float32Array`. Note that
   * this involves an expensive GPU to CPU transfer if the current mask is
   * only available as a `WebGLTexture`.
   *
   * @export
   * @return The current mask as a Float32Array.
   */
  getAsFloat32Array(): Float32Array {
    return this.convertToFloat32Array();
  }

  /**
   * Returns the underlying mask as a `WebGLTexture` object. Note that this
   * involves a CPU to GPU transfer if the current mask is only available as
   * a CPU array. The returned texture is bound to the current canvas (see
   * `.canvas`).
   *
   * @export
   * @return The current mask as a WebGLTexture.
   */
  getAsWebGLTexture(): WebGLTexture {
    return this.convertToWebGLTexture();
  }

  /**
   * Returns the underlying mask as an `ImageBitmap`. The bitmap is grayscale
   * RGBA (`vec4(m)`) so it can be passed to `VideoFrame`, Insertable Streams,
   * or canvas compositing without a GPU→CPU `readPixels` of the float mask.
   *
   * Conversion requires that the MediaPipe Task was initialized with an
   * `OffscreenCanvas` with WebGL2 support. See
   * https://developer.mozilla.org/en-US/docs/Web/API/OffscreenCanvas/getContext
   * for a list of supported platforms.
   *
   * @export
   * @return The current mask as an ImageBitmap object.
   */
  getAsImageBitmap(): ImageBitmap {
    return this.convertToImageBitmap();
  }

  /**
   * Returns the texture format used for writing float textures on this
   * platform.
   */
  private getTexImage2DFormat(): GLenum {
    const gl = this.getGL();
    if (!MPMask.texImage2DFormat) {
      // Note: This is the same check we use in
      // `SegmentationPostprocessorGl::GetSegmentationResultGpu()`.
      if (gl.getExtension('EXT_color_buffer_float') &&
          gl.getExtension('OES_texture_float_linear') &&
          gl.getExtension('EXT_float_blend')) {
        MPMask.texImage2DFormat = gl.R32F;
      } else if (gl.getExtension('EXT_color_buffer_half_float')) {
        MPMask.texImage2DFormat = gl.R16F;
      } else {
        throw new Error(
            'GPU does not fully support 4-channel float32 or float16 formats');
      }
    }
    return MPMask.texImage2DFormat;
  }

  private getContainer(type: MPMaskType.UINT8_ARRAY): Uint8Array|undefined;
  private getContainer(type: MPMaskType.FLOAT32_ARRAY): Float32Array|undefined;
  private getContainer(type: MPMaskType.WEBGL_TEXTURE): WebGLTexture|undefined;
  private getContainer(type: MPMaskType.IMAGE_BITMAP): ImageBitmap|undefined;
  private getContainer(type: MPMaskType): MPMaskContainer|undefined;
  /** Returns the container for the requested storage type iff it exists. */
  private getContainer(type: MPMaskType): MPMaskContainer|undefined {
    switch (type) {
      case MPMaskType.UINT8_ARRAY:
        return this.containers.find(img => img instanceof Uint8Array);
      case MPMaskType.FLOAT32_ARRAY:
        return this.containers.find(img => img instanceof Float32Array);
      case MPMaskType.WEBGL_TEXTURE:
        return this.containers.find(
            img => typeof WebGLTexture !== 'undefined' &&
                img instanceof WebGLTexture);
      case MPMaskType.IMAGE_BITMAP:
        return this.containers.find(
            img => typeof ImageBitmap !== 'undefined' &&
                img instanceof ImageBitmap);
      default:
        throw new Error(`Type is not supported: ${type}`);
    }
  }

  /**
   * Creates a copy of the resources stored in this `MPMask`. You can
   * invoke this method to extend the lifetime of a mask returned by a
   * MediaPipe Task. Note that performance critical applications should aim to
   * only use the `MPMask` within the MediaPipe Task callback so that
   * copies can be avoided.
   *
   * @export
   */
  clone(): MPMask {
    const destinationContainers: MPMaskContainer[] = [];

    // TODO: We might only want to clone one backing datastructure
    // even if multiple are defined;
    for (const container of this.containers) {
      let destinationContainer: MPMaskContainer;

      if (container instanceof Uint8Array) {
        destinationContainer = new Uint8Array(container);
      } else if (container instanceof Float32Array) {
        destinationContainer = new Float32Array(container);
      } else if (container instanceof WebGLTexture) {
        const gl = this.getGL();
        const shaderContext = this.getShaderContext();

        // Create a new texture and use it to back a framebuffer
        gl.activeTexture(gl.TEXTURE1);
        destinationContainer = shaderContext.createTexture(
            gl, this.interpolateValues ? gl.LINEAR : gl.NEAREST);
        gl.bindTexture(gl.TEXTURE_2D, destinationContainer);
        const format = this.getTexImage2DFormat();
        gl.texImage2D(
            gl.TEXTURE_2D, 0, format, this.width, this.height, 0, gl.RED,
            gl.FLOAT, null);
        gl.bindTexture(gl.TEXTURE_2D, null);

        shaderContext.bindFramebuffer(gl, destinationContainer);
        shaderContext.run(gl, /* flipVertically= */ false, () => {
          this.bindTexture();  // This activates gl.TEXTURE0
          gl.clearColor(0, 0, 0, 0);
          gl.clear(gl.COLOR_BUFFER_BIT);
          gl.drawArrays(gl.TRIANGLE_FAN, 0, 4);
          this.unbindTexture();
        });
        shaderContext.unbindFramebuffer();

        this.unbindTexture();
      } else if (
          typeof ImageBitmap !== 'undefined' &&
          container instanceof ImageBitmap) {
        this.convertToWebGLTexture();
        this.bindTexture();
        destinationContainer = this.copyTextureToBitmap();
        this.unbindTexture();
      } else {
        throw new Error(`Type is not supported: ${container}`);
      }

      destinationContainers.push(destinationContainer);
    }

    return new MPMask(
        destinationContainers, this.interpolateValues, this.hasWebGLTexture(),
        this.canvas, this.shaderContext, this.width, this.height,
        this.hasImageBitmap());
  }

  private getOffscreenCanvas(): OffscreenCanvas {
    if (!(this.canvas instanceof OffscreenCanvas)) {
      throw new Error(
          'Conversion to ImageBitmap requires that the MediaPipe Tasks is ' +
          'initialized with an OffscreenCanvas');
    }
    return this.canvas;
  }

  private getGL(): WebGL2RenderingContext {
    if (!this.canvas) {
      throw new Error(
          'Conversion to different image formats require that a canvas ' +
          'is passed when initializing the image.');
    }
    if (!this.gl) {
      this.gl = assertExists(
          this.canvas.getContext('webgl2') as WebGL2RenderingContext,
          'You cannot use a canvas that is already bound to a different ' +
              'type of rendering context.');
    }
    return this.gl;
  }

  private getShaderContext(): MPImageShaderContext {
    if (!this.shaderContext) {
      this.shaderContext = new MPImageShaderContext();
    }
    return this.shaderContext;
  }

  private getBitmapShaderContext(): MPMaskBitmapShaderContext {
    const gl = this.getGL();
    let context = bitmapShaderContexts.get(gl);
    if (!context) {
      context = new MPMaskBitmapShaderContext();
      bitmapShaderContexts.set(gl, context);
    }
    return context;
  }

  private convertToFloat32Array(): Float32Array {
    let float32Array = this.getContainer(MPMaskType.FLOAT32_ARRAY);
    if (!float32Array) {
      const uint8Array = this.getContainer(MPMaskType.UINT8_ARRAY);
      if (uint8Array) {
        float32Array = new Float32Array(uint8Array).map(v => v / 255);
      } else {
        float32Array = new Float32Array(this.width * this.height);

        const gl = this.getGL();
        const shaderContext = this.getShaderContext();

        // Create texture if needed
        const webGlTexture = this.convertToWebGLTexture();

        // Create a framebuffer from the texture and read back pixels
        shaderContext.bindFramebuffer(gl, webGlTexture);

        if (isIOS()) {
          // WebKit on iOS only supports gl.HALF_FLOAT for single channel reads
          // (as tested on iOS 16.4). HALF_FLOAT requires reading data into a
          // Uint16Array, however, and requires a manual bitwise conversion from
          // Uint16 to floating point numbers. This conversion is more expensive
          // that reading back a Float32Array from the RGBA image and dropping
          // the superfluous data, so we do this instead.
          const outputArray = new Float32Array(this.width * this.height * 4);
          gl.readPixels(
              0, 0, this.width, this.height, gl.RGBA, gl.FLOAT, outputArray);
          for (let i = 0, j = 0; i < float32Array.length; ++i, j += 4) {
            float32Array[i] = outputArray[j];
          }
        } else {
          gl.readPixels(
              0, 0, this.width, this.height, gl.RED, gl.FLOAT, float32Array);
        }
      }
      this.containers.push(float32Array);
    }

    return float32Array;
  }

  private convertToUint8Array(): Uint8Array {
    let uint8Array = this.getContainer(MPMaskType.UINT8_ARRAY);
    if (!uint8Array) {
      const floatArray = this.convertToFloat32Array();
      uint8Array = new Uint8Array(floatArray.map(v => Math.round(255 * v)));
      this.containers.push(uint8Array);
    }
    return uint8Array;
  }

  private convertToWebGLTexture(): WebGLTexture {
    let webGLTexture = this.getContainer(MPMaskType.WEBGL_TEXTURE);
    if (!webGLTexture) {
      const gl = this.getGL();
      webGLTexture = this.bindTexture();

      const data = this.convertToFloat32Array();
      const format = this.getTexImage2DFormat();
      gl.texImage2D(
          gl.TEXTURE_2D, 0, format, this.width, this.height, 0, gl.RED,
          gl.FLOAT, data);
      this.unbindTexture();
    }

    return webGLTexture;
  }

  private convertToImageBitmap(): ImageBitmap {
    let imageBitmap = this.getContainer(MPMaskType.IMAGE_BITMAP);
    if (!imageBitmap) {
      this.convertToWebGLTexture();
      imageBitmap = this.convertWebGLTextureToImageBitmap();
      this.containers.push(imageBitmap);
      this.ownsImageBitmap = true;
    }
    return imageBitmap;
  }

  /**
   * Binds the backing texture to the canvas. If the texture does not yet
   * exist, creates it first.
   */
  private bindTexture(): WebGLTexture {
    const gl = this.getGL();

    gl.viewport(0, 0, this.width, this.height);
    gl.activeTexture(gl.TEXTURE0);

    let webGLTexture = this.getContainer(MPMaskType.WEBGL_TEXTURE);
    if (!webGLTexture) {
      const shaderContext = this.getShaderContext();
      webGLTexture = shaderContext.createTexture(
          gl, this.interpolateValues ? gl.LINEAR : gl.NEAREST);
      this.containers.push(webGLTexture);
      this.ownsWebGLTexture = true;
    }

    gl.bindTexture(gl.TEXTURE_2D, webGLTexture);
    return webGLTexture;
  }

  private unbindTexture(): void {
    this.gl!.bindTexture(this.gl!.TEXTURE_2D, null);
  }

  /**
   * Invokes a shader to render the current mask texture as grayscale RGBA and
   * return it as an ImageBitmap.
   */
  private copyTextureToBitmap(): ImageBitmap {
    const gl = this.getGL();
    const shaderContext = this.getBitmapShaderContext();

    return shaderContext.run(gl, /* flipVertically= */ true, () => {
      return this.runWithResizedCanvas(() => {
        // Unbind any framebuffer that may be bound since
        // `transferToImageBitmap()` requires rendering into the display (null)
        // framebuffer.
        gl.bindFramebuffer(gl.FRAMEBUFFER, null);

        gl.clearColor(0, 0, 0, 0);
        gl.clear(gl.COLOR_BUFFER_BIT);
        gl.drawArrays(gl.TRIANGLE_FAN, 0, 4);
        return this.getOffscreenCanvas().transferToImageBitmap();
      });
    });
  }

  private convertWebGLTextureToImageBitmap(): ImageBitmap {
    this.bindTexture();
    const result = this.copyTextureToBitmap();
    this.unbindTexture();
    return result;
  }

  /**
   * Temporarily resizes the underlying canvas to match the dimensions of the
   * mask. Runs the provided callback on the resized canvas.
   *
   * Note that while resizing is an expensive operation, it allows us to use
   * the synchronous `transferToImageBitmap()` API.
   */
  private runWithResizedCanvas<T>(callback: () => T): T {
    const canvas = this.canvas!;

    if (canvas.width === this.width && canvas.height === this.height) {
      return callback();
    }

    const originalWidth = canvas.width;
    const originalHeight = canvas.height;
    canvas.width = this.width;
    canvas.height = this.height;

    const result = callback();

    canvas.width = originalWidth;
    canvas.height = originalHeight;

    return result;
  }

  /**
   * Frees up any resources owned by this `MPMask` instance.
   *
   * Note that this method does not free masks that are owned by the C++
   * Task, as these are freed automatically once you leave the MediaPipe
   * callback. Additionally, some shared state is freed only once you invoke
   * the Task's `close()` method.
   *
   * @export
   */
  close(): void {
    if (this.ownsImageBitmap) {
      this.getContainer(MPMaskType.IMAGE_BITMAP)!.close();
    }

    if (this.ownsWebGLTexture) {
      const gl = this.getGL();
      gl.deleteTexture(this.getContainer(MPMaskType.WEBGL_TEXTURE)!);
    }

    // User called close(). We no longer issue warning.
    MPMask.instancesBeforeWarning = -1;
  }
}
