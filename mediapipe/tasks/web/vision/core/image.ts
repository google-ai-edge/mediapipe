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

/** Number of instances a user can keep alive before we raise a warning. */
const INSTANCE_COUNT_WARNING_THRESHOLD = 250;

/** The underlying type of the image. */
enum MPImageType {
  /** Represents the native `ImageData` type. */
  IMAGE_DATA,
  /** Represents the native `ImageBitmap` type. */
  IMAGE_BITMAP,
  /** Represents the native `WebGLTexture` type. */
  WEBGL_TEXTURE
}

/** The supported image formats. For internal usage. */
export type MPImageContainer = ImageData|ImageBitmap|WebGLTexture;

/**
 * The wrapper class for MediaPipe Image objects.
 *
 * Images are stored as `ImageData`, `ImageBitmap` or `WebGLTexture` objects.
 * You can convert the underlying type to any other type by passing the
 * desired type to `getAs...()`. As type conversions can be expensive, it is
 * recommended to limit these conversions. You can verify what underlying
 * types are already available by invoking `has...()`.
 *
 * Images that are returned from a MediaPipe Tasks are owned by by the
 * underlying C++ Task. If you need to extend the lifetime of these objects,
 * you can invoke the `clone()` method. To free up the resources obtained
 * during any clone or type conversion operation, it is important to invoke
 * `close()` on the `MPImage` instance.
 *
 * Converting to and from ImageBitmap requires that the MediaPipe task is
 * initialized with an `OffscreenCanvas`. As we require WebGL2 support, this
 * places some limitations on Browser support as outlined here:
 * https://developer.mozilla.org/en-US/docs/Web/API/OffscreenCanvas/getContext
 */
export class MPImage {
  private gl?: WebGL2RenderingContext;

  /**
   * A counter to track the number of instances of MPImage that own resources..
   * This is used to raise a warning if the user does not close the instances.
   */
  private static instancesBeforeWarning = INSTANCE_COUNT_WARNING_THRESHOLD;

  /** @hideconstructor */
  constructor(
      private readonly containers: MPImageContainer[],
      private ownsImageBitmap: boolean,
      private ownsWebGLTexture: boolean,
      /** Returns the canvas element that the image is bound to. */
      readonly canvas: HTMLCanvasElement|OffscreenCanvas|undefined,
      private shaderContext: MPImageShaderContext|undefined,
      /** Returns the width of the image. */
      readonly width: number,
      /** Returns the height of the image. */
      readonly height: number,
  ) {
    if (this.ownsImageBitmap || this.ownsWebGLTexture) {
      --MPImage.instancesBeforeWarning;
      if (MPImage.instancesBeforeWarning === 0) {
        console.error(
            'You seem to be creating MPImage instances without invoking ' +
            '.close(). This leaks resources.');
      }
    }
  }

  /**
   * Returns whether this `MPImage` contains a mask of type `ImageData`.
   * @export
   */
  hasImageData(): boolean {
    return !!this.getContainer(MPImageType.IMAGE_DATA);
  }

  /**
   * Returns whether this `MPImage` contains a mask of type `ImageBitmap`.
   * @export
   */
  hasImageBitmap(): boolean {
    return !!this.getContainer(MPImageType.IMAGE_BITMAP);
  }

  /**
   * Returns whether this `MPImage` contains a mask of type `WebGLTexture`.
   * @export
   */
  hasWebGLTexture(): boolean {
    return !!this.getContainer(MPImageType.WEBGL_TEXTURE);
  }

  /**
   * Returns the underlying image as an `ImageData` object. Note that this
   * involves an expensive GPU to CPU transfer if the current image is only
   * available as an `ImageBitmap` or `WebGLTexture`.
   *
   * @export
   * @return The current image as an ImageData object.
   */
  getAsImageData(): ImageData {
    return this.convertToImageData();
  }

  /**
   * Returns the underlying image as an `ImageBitmap`. Note that
   * conversions to `ImageBitmap` are expensive, especially if the data
   * currently resides on CPU.
   *
   * Processing with `ImageBitmap`s requires that the MediaPipe Task was
   * initialized with an `OffscreenCanvas` with WebGL2 support. See
   * https://developer.mozilla.org/en-US/docs/Web/API/OffscreenCanvas/getContext
   * for a list of supported platforms.
   *
   * @export
   * @return The current image as an ImageBitmap object.
   */
  getAsImageBitmap(): ImageBitmap {
    return this.convertToImageBitmap();
  }

  /**
   * Returns the underlying image as a `WebGLTexture` object. Note that this
   * involves a CPU to GPU transfer if the current image is only available as
   * an `ImageData` object. The returned texture is bound to the current
   * canvas (see `.canvas`).
   *
   * @export
   * @return The current image as a WebGLTexture.
   */
  getAsWebGLTexture(): WebGLTexture {
    return this.convertToWebGLTexture();
  }

  private getContainer(type: MPImageType.IMAGE_DATA): ImageData|undefined;
  private getContainer(type: MPImageType.IMAGE_BITMAP): ImageBitmap|undefined;
  private getContainer(type: MPImageType.WEBGL_TEXTURE): WebGLTexture|undefined;
  private getContainer(type: MPImageType): MPImageContainer|undefined;
  /** Returns the container for the requested storage type iff it exists. */
  private getContainer(type: MPImageType): MPImageContainer|undefined {
    switch (type) {
      case MPImageType.IMAGE_DATA:
        return this.containers.find(img => img instanceof ImageData);
      case MPImageType.IMAGE_BITMAP:
        return this.containers.find(
            img => typeof ImageBitmap !== 'undefined' &&
                img instanceof ImageBitmap);
      case MPImageType.WEBGL_TEXTURE:
        return this.containers.find(
            img => typeof WebGLTexture !== 'undefined' &&
                img instanceof WebGLTexture);
      default:
        throw new Error(`Type is not supported: ${type}`);
    }
  }

  /**
   * Creates a copy of the resources stored in this `MPImage`. You can invoke
   * this method to extend the lifetime of an image returned by a MediaPipe
   * Task. Note that performance critical applications should aim to only use
   * the `MPImage` within the MediaPipe Task callback so that copies can be
   * avoided.
   *
   * @export
   */
  clone(): MPImage {
    const destinationContainers: MPImageContainer[] = [];

    // TODO: We might only want to clone one backing datastructure
    // even if multiple are defined;
    for (const container of this.containers) {
      let destinationContainer: MPImageContainer;

      if (container instanceof ImageData) {
        destinationContainer =
            new ImageData(container.data, this.width, this.height);
      } else if (container instanceof WebGLTexture) {
        const gl = this.getGL();
        const shaderContext = this.getShaderContext();

        // Create a new texture and use it to back a framebuffer
        gl.activeTexture(gl.TEXTURE1);
        destinationContainer = shaderContext.createTexture(gl);
        gl.bindTexture(gl.TEXTURE_2D, destinationContainer);
        gl.texImage2D(
            gl.TEXTURE_2D, 0, gl.RGBA, this.width, this.height, 0, gl.RGBA,
            gl.UNSIGNED_BYTE, null);
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
      } else if (container instanceof ImageBitmap) {
        this.convertToWebGLTexture();
        this.bindTexture();
        destinationContainer = this.copyTextureToBitmap();
        this.unbindTexture();
      } else {
        throw new Error(`Type is not supported: ${container}`);
      }

      destinationContainers.push(destinationContainer);
    }

    return new MPImage(
        destinationContainers, this.hasImageBitmap(), this.hasWebGLTexture(),
        this.canvas, this.shaderContext, this.width, this.height);
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

  private convertToImageBitmap(): ImageBitmap {
    let imageBitmap = this.getContainer(MPImageType.IMAGE_BITMAP);
    if (!imageBitmap) {
      this.convertToWebGLTexture();
      imageBitmap = this.convertWebGLTextureToImageBitmap();
      this.containers.push(imageBitmap);
      this.ownsImageBitmap = true;
    }

    return imageBitmap;
  }

  private convertToImageData(): ImageData {
    let imageData = this.getContainer(MPImageType.IMAGE_DATA);
    if (!imageData) {
      const gl = this.getGL();
      const shaderContext = this.getShaderContext();
      const pixels = new Uint8Array(this.width * this.height * 4);

      // Create texture if needed
      const webGlTexture = this.convertToWebGLTexture();

      // Create a framebuffer from the texture and read back pixels
      shaderContext.bindFramebuffer(gl, webGlTexture);
      gl.readPixels(
          0, 0, this.width, this.height, gl.RGBA, gl.UNSIGNED_BYTE, pixels);
      shaderContext.unbindFramebuffer();

      imageData = new ImageData(
          new Uint8ClampedArray(pixels.buffer), this.width, this.height);
      this.containers.push(imageData);
    }

    return imageData;
  }

  private convertToWebGLTexture(): WebGLTexture {
    let webGLTexture = this.getContainer(MPImageType.WEBGL_TEXTURE);
    if (!webGLTexture) {
      const gl = this.getGL();
      webGLTexture = this.bindTexture();
      const source = this.getContainer(MPImageType.IMAGE_BITMAP) ||
          this.convertToImageData();
      gl.texImage2D(
          gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, source);
      this.unbindTexture();
    }

    return webGLTexture;
  }

  /**
   * Binds the backing texture to the canvas. If the texture does not yet
   * exist, creates it first.
   */
  private bindTexture(): WebGLTexture {
    const gl = this.getGL();

    gl.viewport(0, 0, this.width, this.height);
    gl.activeTexture(gl.TEXTURE0);

    let webGLTexture = this.getContainer(MPImageType.WEBGL_TEXTURE);
    if (!webGLTexture) {
      const shaderContext = this.getShaderContext();
      webGLTexture = shaderContext.createTexture(gl);
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
   * Invokes a shader to render the current texture and return it as an
   * ImageBitmap
   */
  private copyTextureToBitmap(): ImageBitmap {
    const gl = this.getGL();
    const shaderContext = this.getShaderContext();

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
   * image. Runs the provided callback on the resized canvas.
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
   * Frees up any resources owned by this `MPImage` instance.
   *
   * Note that this method does not free images that are owned by the C++
   * Task, as these are freed automatically once you leave the MediaPipe
   * callback. Additionally, some shared state is freed only once you invoke the
   * Task's `close()` method.
   *
   * @export
   */
  close(): void {
    if (this.ownsImageBitmap) {
      this.getContainer(MPImageType.IMAGE_BITMAP)!.close();
    }

    if (this.ownsWebGLTexture) {
      const gl = this.getGL();
      gl.deleteTexture(this.getContainer(MPImageType.WEBGL_TEXTURE)!);
    }

    // User called close(). We no longer issue warning.
    MPImage.instancesBeforeWarning = -1;
  }
}


