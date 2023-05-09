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

import {assertNotNull, MPImageShaderContext} from '../../../../tasks/web/vision/core/image_shader_context';

/** The underlying type of the image. */
enum MPMaskType {
  /** Represents the native `UInt8Array` type. */
  UINT8_ARRAY,
  /** Represents the native `Float32Array` type.  */
  FLOAT32_ARRAY,
  /** Represents the native `WebGLTexture` type. */
  WEBGL_TEXTURE
}

/** The supported mask formats. For internal usage. */
export type MPMaskContainer = Uint8Array|Float32Array|WebGLTexture;

/**
 * The wrapper class for MediaPipe segmentation masks.
 *
 * Masks are stored as `Uint8Array`, `Float32Array` or `WebGLTexture` objects.
 * You can convert the underlying type to any other type by passing the desired
 * type to `getAs...()`. As type conversions can be expensive, it is recommended
 * to limit these conversions. You can verify what underlying types are already
 * available by invoking `has...()`.
 *
 * Masks that are returned from a MediaPipe Tasks are owned by by the
 * underlying C++ Task. If you need to extend the lifetime of these objects,
 * you can invoke the `clone()` method. To free up the resources obtained
 * during any clone or type conversion operation, it is important to invoke
 * `close()` on the `MPMask` instance.
 */
export class MPMask {
  private gl?: WebGL2RenderingContext;

  /** @hideconstructor */
  constructor(
      private readonly containers: MPMaskContainer[],
      private ownsWebGLTexture: boolean,
      /** Returns the canvas element that the mask is bound to. */
      readonly canvas: HTMLCanvasElement|OffscreenCanvas|undefined,
      private shaderContext: MPImageShaderContext|undefined,
      /** Returns the width of the mask. */
      readonly width: number,
      /** Returns the height of the mask. */
      readonly height: number,
  ) {}

  /** Returns whether this `MPMask` contains a mask of type `Uint8Array`. */
  hasUint8Array(): boolean {
    return !!this.getContainer(MPMaskType.UINT8_ARRAY);
  }

  /** Returns whether this `MPMask` contains a mask of type `Float32Array`. */
  hasFloat32Array(): boolean {
    return !!this.getContainer(MPMaskType.FLOAT32_ARRAY);
  }

  /** Returns whether this `MPMask` contains a mask of type `WebGLTexture`. */
  hasWebGLTexture(): boolean {
    return !!this.getContainer(MPMaskType.WEBGL_TEXTURE);
  }

  /**
   * Returns the underlying mask as a Uint8Array`. Note that this involves an
   * expensive GPU to CPU transfer if the current mask is only available as a
   * `WebGLTexture`.
   *
   * @return The current data as a Uint8Array.
   */
  getAsUint8Array(): Uint8Array {
    return this.convertToUint8Array();
  }

  /**
   * Returns the underlying mask as a single channel `Float32Array`. Note that
   * this involves an expensive GPU to CPU transfer if the current mask is only
   * available as a `WebGLTexture`.
   *
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
   * @return The current mask as a WebGLTexture.
   */
  getAsWebGLTexture(): WebGLTexture {
    return this.convertToWebGLTexture();
  }

  private getContainer(type: MPMaskType.UINT8_ARRAY): Uint8Array|undefined;
  private getContainer(type: MPMaskType.FLOAT32_ARRAY): Float32Array|undefined;
  private getContainer(type: MPMaskType.WEBGL_TEXTURE): WebGLTexture|undefined;
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
        destinationContainer =
            assertNotNull(gl.createTexture(), 'Failed to create texture');
        gl.bindTexture(gl.TEXTURE_2D, destinationContainer);
        gl.texImage2D(
            gl.TEXTURE_2D, 0, gl.R32F, this.width, this.height, 0, gl.RED,
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
      } else {
        throw new Error(`Type is not supported: ${container}`);
      }

      destinationContainers.push(destinationContainer);
    }

    return new MPMask(
        destinationContainers, this.hasWebGLTexture(), this.canvas,
        this.shaderContext, this.width, this.height);
  }

  private getGL(): WebGL2RenderingContext {
    if (!this.canvas) {
      throw new Error(
          'Conversion to different image formats require that a canvas ' +
          'is passed when iniitializing the image.');
    }
    if (!this.gl) {
      this.gl = assertNotNull(
          this.canvas.getContext('webgl2') as WebGL2RenderingContext | null,
          'You cannot use a canvas that is already bound to a different ' +
              'type of rendering context.');
    }
    const ext = this.gl.getExtension('EXT_color_buffer_float');
    if (!ext) {
      // TODO: Ensure this works on iOS
      throw new Error('Missing required EXT_color_buffer_float extension');
    }
    return this.gl;
  }

  private getShaderContext(): MPImageShaderContext {
    if (!this.shaderContext) {
      this.shaderContext = new MPImageShaderContext();
    }
    return this.shaderContext;
  }

  private convertToFloat32Array(): Float32Array {
    let float32Array = this.getContainer(MPMaskType.FLOAT32_ARRAY);
    if (!float32Array) {
      const uint8Array = this.getContainer(MPMaskType.UINT8_ARRAY);
      if (uint8Array) {
        float32Array = new Float32Array(uint8Array).map(v => v / 255);
      } else {
        const gl = this.getGL();
        const shaderContext = this.getShaderContext();
        float32Array = new Float32Array(this.width * this.height);

        // Create texture if needed
        const webGlTexture = this.convertToWebGLTexture();

        // Create a framebuffer from the texture and read back pixels
        shaderContext.bindFramebuffer(gl, webGlTexture);
        gl.readPixels(
            0, 0, this.width, this.height, gl.RED, gl.FLOAT, float32Array);
        shaderContext.unbindFramebuffer();
      }
      this.containers.push(float32Array);
    }

    return float32Array;
  }

  private convertToUint8Array(): Uint8Array {
    let uint8Array = this.getContainer(MPMaskType.UINT8_ARRAY);
    if (!uint8Array) {
      const floatArray = this.convertToFloat32Array();
      uint8Array = new Uint8Array(floatArray.map(v => 255 * v));
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
      // TODO: Add support for R16F to support iOS
      gl.texImage2D(
          gl.TEXTURE_2D, 0, gl.R32F, this.width, this.height, 0, gl.RED,
          gl.FLOAT, data);
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

    let webGLTexture = this.getContainer(MPMaskType.WEBGL_TEXTURE);
    if (!webGLTexture) {
      webGLTexture =
          assertNotNull(gl.createTexture(), 'Failed to create texture');
      this.containers.push(webGLTexture);
      this.ownsWebGLTexture = true;
    }

    gl.bindTexture(gl.TEXTURE_2D, webGLTexture);
    // TODO: Ideally, we would only set these once per texture and
    // not once every frame.
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);

    return webGLTexture;
  }

  private unbindTexture(): void {
    this.gl!.bindTexture(this.gl!.TEXTURE_2D, null);
  }

  /**
   * Frees up any resources owned by this `MPMask` instance.
   *
   * Note that this method does not free masks that are owned by the C++
   * Task, as these are freed automatically once you leave the MediaPipe
   * callback. Additionally, some shared state is freed only once you invoke
   * the Task's `close()` method.
   */
  close(): void {
    if (this.ownsWebGLTexture) {
      const gl = this.getGL();
      gl.deleteTexture(this.getContainer(MPMaskType.WEBGL_TEXTURE)!);
    }
  }
}


