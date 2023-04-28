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

/** The underlying type of the image. */
export enum MPImageStorageType {
  /** Represents the native `UInt8ClampedArray` type. */
  UINT8_CLAMPED_ARRAY,
  /**
   * Represents the native `Float32Array` type. Values range from [0.0, 1.0].
   */
  FLOAT32_ARRAY,
  /** Represents the native `ImageData` type. */
  IMAGE_DATA,
  /** Represents the native `ImageBitmap` type. */
  IMAGE_BITMAP,
  /** Represents the native `WebGLTexture` type. */
  WEBGL_TEXTURE
}

/** The supported image formats. For internal usage. */
export type MPImageNativeContainer =
    Uint8ClampedArray|Float32Array|ImageData|ImageBitmap|WebGLTexture;

const VERTEX_SHADER = `
  attribute vec2 aVertex;
  attribute vec2 aTex;
  varying vec2 vTex;
  void main(void) {
    gl_Position = vec4(aVertex, 0.0, 1.0);
    vTex = aTex;
  }`;

const FRAGMENT_SHADER = `
  precision mediump float;
  varying vec2 vTex;
  uniform sampler2D inputTexture;
   void main() {
     gl_FragColor = texture2D(inputTexture, vTex);
   }
 `;

function assertNotNull<T>(value: T|null, msg: string): T {
  if (value === null) {
    throw new Error(`Unable to obtain required WebGL resource: ${msg}`);
  }
  return value;
}

/**
 * Utility class that encapsulates the buffers used by `MPImageShaderContext`.
 */
class MPImageShaderBuffers {
  constructor(
      private readonly gl: WebGL2RenderingContext,
      private readonly vertexArrayObject: WebGLVertexArrayObject,
      private readonly vertexBuffer: WebGLBuffer,
      private readonly textureBuffer: WebGLBuffer) {}

  bind() {
    this.gl.bindVertexArray(this.vertexArrayObject);
  }

  unbind() {
    this.gl.bindVertexArray(null);
  }

  close() {
    this.gl.deleteVertexArray(this.vertexArrayObject);
    this.gl.deleteBuffer(this.vertexBuffer);
    this.gl.deleteBuffer(this.textureBuffer);
  }
}

/**
 * A class that encapsulates the shaders used by an MPImage. Can be re-used
 * across MPImages that use the same WebGL2Rendering context.
 */
export class MPImageShaderContext {
  private gl?: WebGL2RenderingContext;
  private framebuffer?: WebGLFramebuffer;
  private program?: WebGLProgram;
  private vertexShader?: WebGLShader;
  private fragmentShader?: WebGLShader;
  private aVertex?: GLint;
  private aTex?: GLint;

  /**
   * The shader buffers used for passthrough renders that don't modify the
   * input texture.
   */
  private shaderBuffersPassthrough?: MPImageShaderBuffers;

  /**
   * The shader buffers used for passthrough renders that flip the input texture
   * vertically before conversion to a different type. This is used to flip the
   * texture to the expected orientation for drawing in the browser.
   */
  private shaderBuffersFlipVertically?: MPImageShaderBuffers;

  private compileShader(source: string, type: number): WebGLShader {
    const gl = this.gl!;
    const shader =
        assertNotNull(gl.createShader(type), 'Failed to create WebGL shader');
    gl.shaderSource(shader, source);
    gl.compileShader(shader);
    if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
      const info = gl.getShaderInfoLog(shader);
      throw new Error(`Could not compile WebGL shader: ${info}`);
    }
    gl.attachShader(this.program!, shader);
    return shader;
  }

  private setupShaders(): void {
    const gl = this.gl!;
    this.program =
        assertNotNull(gl.createProgram()!, 'Failed to create WebGL program');

    this.vertexShader = this.compileShader(VERTEX_SHADER, gl.VERTEX_SHADER);
    this.fragmentShader =
        this.compileShader(FRAGMENT_SHADER, gl.FRAGMENT_SHADER);

    gl.linkProgram(this.program);
    const linked = gl.getProgramParameter(this.program, gl.LINK_STATUS);
    if (!linked) {
      const info = gl.getProgramInfoLog(this.program);
      throw new Error(`Error during program linking: ${info}`);
    }

    this.aVertex = gl.getAttribLocation(this.program, 'aVertex');
    this.aTex = gl.getAttribLocation(this.program, 'aTex');
  }

  private createBuffers(flipVertically: boolean): MPImageShaderBuffers {
    const gl = this.gl!;
    const vertexArrayObject =
        assertNotNull(gl.createVertexArray(), 'Failed to create vertex array');
    gl.bindVertexArray(vertexArrayObject);

    const vertexBuffer =
        assertNotNull(gl.createBuffer(), 'Failed to create buffer');
    gl.bindBuffer(gl.ARRAY_BUFFER, vertexBuffer);
    gl.enableVertexAttribArray(this.aVertex!);
    gl.vertexAttribPointer(this.aVertex!, 2, gl.FLOAT, false, 0, 0);
    gl.bufferData(
        gl.ARRAY_BUFFER, new Float32Array([-1, -1, -1, 1, 1, 1, 1, -1]),
        gl.STATIC_DRAW);

    const textureBuffer =
        assertNotNull(gl.createBuffer(), 'Failed to create buffer');
    gl.bindBuffer(gl.ARRAY_BUFFER, textureBuffer);
    gl.enableVertexAttribArray(this.aTex!);
    gl.vertexAttribPointer(this.aTex!, 2, gl.FLOAT, false, 0, 0);

    const bufferData =
        flipVertically ? [0, 1, 0, 0, 1, 0, 1, 1] : [0, 0, 0, 1, 1, 1, 1, 0];
    gl.bufferData(
        gl.ARRAY_BUFFER, new Float32Array(bufferData), gl.STATIC_DRAW);

    gl.bindBuffer(gl.ARRAY_BUFFER, null);
    gl.bindVertexArray(null);

    return new MPImageShaderBuffers(
        gl, vertexArrayObject, vertexBuffer, textureBuffer);
  }

  private getShaderBuffers(flipVertically: boolean): MPImageShaderBuffers {
    if (flipVertically) {
      if (!this.shaderBuffersFlipVertically) {
        this.shaderBuffersFlipVertically =
            this.createBuffers(/* flipVertically= */ true);
      }
      return this.shaderBuffersFlipVertically;
    } else {
      if (!this.shaderBuffersPassthrough) {
        this.shaderBuffersPassthrough =
            this.createBuffers(/* flipVertically= */ false);
      }
      return this.shaderBuffersPassthrough;
    }
  }

  private maybeInitGL(gl: WebGL2RenderingContext): void {
    if (!this.gl) {
      this.gl = gl;
    } else if (gl !== this.gl) {
      throw new Error('Cannot change GL context once initialized');
    }
  }

  /** Runs the callback using the shader. */
  run<T>(
      gl: WebGL2RenderingContext, flipVertically: boolean,
      callback: () => T): T {
    this.maybeInitGL(gl);

    if (!this.program) {
      this.setupShaders();
    }

    const shaderBuffers = this.getShaderBuffers(flipVertically);
    gl.useProgram(this.program!);
    shaderBuffers.bind();
    const result = callback();
    shaderBuffers.unbind();

    return result;
  }

  /**
   * Binds a framebuffer to the canvas. If the framebuffer does not yet exist,
   * creates it first. Binds the provided texture to the framebuffer.
   */
  bindFramebuffer(gl: WebGL2RenderingContext, texture: WebGLTexture): void {
    this.maybeInitGL(gl);
    if (!this.framebuffer) {
      this.framebuffer =
          assertNotNull(gl.createFramebuffer(), 'Failed to create framebuffe.');
    }
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.framebuffer);
    gl.framebufferTexture2D(
        gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, texture, 0);
  }

  unbindFramebuffer(): void {
    this.gl?.bindFramebuffer(this.gl.FRAMEBUFFER, null);
  }

  close() {
    if (this.program) {
      const gl = this.gl!;
      gl.deleteProgram(this.program);
      gl.deleteShader(this.vertexShader!);
      gl.deleteShader(this.fragmentShader!);
    }
    if (this.framebuffer) {
      this.gl!.deleteFramebuffer(this.framebuffer);
    }
    if (this.shaderBuffersPassthrough) {
      this.shaderBuffersPassthrough.close();
    }
    if (this.shaderBuffersFlipVertically) {
      this.shaderBuffersFlipVertically.close();
    }
  }
}

/**
 * An interface that can be used to provide custom conversion functions. These
 * functions are invoked to convert pixel values between different channel
 * counts and value ranges. Any conversion function that is not specified will
 * result in a default conversion.
 */
export interface MPImageChannelConverter {
  /**
   * A conversion function to convert a number in the [0.0, 1.0] range to RGBA.
   * The output is an array with four elemeents whose values range from 0 to 255
   * inclusive.
   *
   * The default conversion function is `[v * 255, v * 255, v * 255, 255]`
   * and will log a warning if invoked.
   */
  floatToRGBAConverter?: (value: number) => [number, number, number, number];

  /*
   * A conversion function to convert a number in the [0, 255] range to RGBA.
   * The output is an array with four elemeents whose values range from 0 to 255
   * inclusive.
   *
   * The default conversion function is `[v, v , v , 255]` and will log a
   * warning if invoked.
   */
  uint8ToRGBAConverter?: (value: number) => [number, number, number, number];

  /**
   * A conversion function to convert an RGBA value in the range of 0 to 255 to
   * a single value in the [0.0, 1.0] range.
   *
   * The default conversion function is `(r / 3 + g / 3 + b / 3) / 255` and will
   * log a warning if invoked.
   */
  rgbaToFloatConverter?: (r: number, g: number, b: number, a: number) => number;

  /**
   * A conversion function to convert an RGBA value in the range of 0 to 255 to
   * a single value in the [0, 255] range.
   *
   * The default conversion function is `r / 3 + g / 3 + b / 3` and will log a
   * warning if invoked.
   */
  rgbaToUint8Converter?: (r: number, g: number, b: number, a: number) => number;

  /**
   * A conversion function to convert a single value in the 0.0 to 1.0 range to
   * [0, 255].
   *
   * The default conversion function is `r * 255` and will log a warning if
   * invoked.
   */
  floatToUint8Converter?: (value: number) => number;

  /**
   * A conversion function to convert a single value in the 0 to 255 range to
   * [0.0, 1.0] .
   *
   * The default conversion function is `r / 255` and will log a warning if
   * invoked.
   */
  uint8ToFloatConverter?: (value: number) => number;
}

const DEFAULT_CONVERTER: Required<MPImageChannelConverter> = {
  floatToRGBAConverter: v => {
    console.log('Using default floatToRGBAConverter');
    return [v * 255, v * 255, v * 255, 255];
  },
  uint8ToRGBAConverter: v => {
    console.log('Using default uint8ToRGBAConverter');
    return [v, v, v, 255];
  },
  rgbaToFloatConverter: (r, g, b) => {
    console.log('Using default floatToRGBAConverter');
    return (r / 3 + g / 3 + b / 3) / 255;
  },
  rgbaToUint8Converter: (r, g, b) => {
    console.log('Using default rgbaToUint8Converter');
    return r / 3 + g / 3 + b / 3;
  },
  floatToUint8Converter: v => {
    console.log('Using default floatToUint8Converter');
    return v * 255;
  },
  uint8ToFloatConverter: v => {
    console.log('Using default uint8ToFloatConverter');
    return v / 255;
  },
};

/**
 * The wrapper class for MediaPipe Image objects.
 *
 * Images are stored as `ImageData`, `ImageBitmap` or `WebGLTexture` objects.
 * You can convert the underlying type to any other type by passing the
 * desired type to `getImage()`. As type conversions can be expensive, it is
 * recommended to limit these conversions. You can verify what underlying
 * types are already available by invoking `hasType()`.
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
 *
 * Some MediaPipe tasks return single channel masks. These masks are stored
 * using an underlying `Uint8ClampedArray` an `Float32Array` (represented as
 * single-channel arrays). To convert these type to other formats a conversion
 * function is invoked to convert pixel values between single channel and four
 * channel RGBA values. To customize this conversion, you can specify these
 * conversion functions when you invoke `getImage()`. If you use the default
 * conversion function a warning will be logged to the console.
 */
export class MPImage {
  private gl?: WebGL2RenderingContext;

  /** @hideconstructor */
  constructor(
      private readonly containers: MPImageNativeContainer[],
      private ownsImageBitmap: boolean,
      private ownsWebGLTexture: boolean,
      /** Returns the canvas element that the image is bound to. */
      readonly canvas: HTMLCanvasElement|OffscreenCanvas|undefined,
      private shaderContext: MPImageShaderContext|undefined,
      /** Returns the width of the image. */
      readonly width: number,
      /** Returns the height of the image. */
      readonly height: number,
  ) {}

  /**
   * Returns whether this `MPImage` stores the image in the desired format.
   * This method can be called to reduce expensive conversion before invoking
   * `getType()`.
   */
  hasType(type: MPImageStorageType): boolean {
    return !!this.getContainer(type);
  }

  /**
   * Returns the underlying image as a single channel `Uint8ClampedArray`. Note
   * that this involves an expensive GPU to CPU transfer if the current image is
   * only available as an `ImageBitmap` or `WebGLTexture`. If necessary, this
   * function converts RGBA data pixel-by-pixel to a single channel value by
   * invoking a conversion function (see class comment for detail).
   *
   * @param type The type of image to return.
   * @param converter A set of conversion functions that will be invoked to
   *     convert the underlying pixel data if necessary. You may omit this
   *     function if the requested conversion does not change the pixel format.
   * @return The current data as a Uint8ClampedArray.
   */
  getImage(
      type: MPImageStorageType.UINT8_CLAMPED_ARRAY,
      converter?: MPImageChannelConverter): Uint8ClampedArray;
  /**
   * Returns the underlying image as a single channel `Float32Array`. Note
   * that this involves an expensive GPU to CPU transfer if the current image is
   * only available as an `ImageBitmap` or `WebGLTexture`. If necessary, this
   * function converts RGBA data pixel-by-pixel to a single channel value by
   * invoking a conversion function (see class comment for detail).
   *
   * @param type The type of image to return.
   * @param converter A set of conversion functions that will be invoked to
   *     convert the underlying pixel data if necessary. You may omit this
   *     function if the requested conversion does not change the pixel format.
   * @return The current image as a Float32Array.
   */
  getImage(
      type: MPImageStorageType.FLOAT32_ARRAY,
      converter?: MPImageChannelConverter): Float32Array;
  /**
   * Returns the underlying image as an `ImageData` object. Note that this
   * involves an expensive GPU to CPU transfer if the current image is only
   * available as an `ImageBitmap` or `WebGLTexture`. If necessary, this
   * function converts single channel pixel values to RGBA by invoking a
   * conversion function (see class comment for detail).
   *
   * @return The current image as an ImageData object.
   */
  getImage(
      type: MPImageStorageType.IMAGE_DATA,
      converter?: MPImageChannelConverter): ImageData;
  /**
   * Returns the underlying image as an `ImageBitmap`. Note that
   * conversions to `ImageBitmap` are expensive, especially if the data
   * currently resides on CPU. If necessary, this function first converts single
   * channel pixel values to RGBA by invoking a conversion function (see class
   * comment for detail).
   *
   * Processing with `ImageBitmap`s requires that the MediaPipe Task was
   * initialized with an `OffscreenCanvas` with WebGL2 support. See
   * https://developer.mozilla.org/en-US/docs/Web/API/OffscreenCanvas/getContext
   * for a list of supported platforms.
   *
   * @param type The type of image to return.
   * @param converter A set of conversion functions that will be invoked to
   *     convert the underlying pixel data if necessary. You may omit this
   *     function if the requested conversion does not change the pixel format.
   * @return The current image as an ImageBitmap object.
   */
  getImage(
      type: MPImageStorageType.IMAGE_BITMAP,
      converter?: MPImageChannelConverter): ImageBitmap;
  /**
   * Returns the underlying image as a `WebGLTexture` object. Note that this
   * involves a CPU to GPU transfer if the current image is only available as
   * an `ImageData` object. The returned texture is bound to the current
   * canvas (see `.canvas`).
   *
   * @param type The type of image to return.
   * @param converter A set of conversion functions that will be invoked to
   *     convert the underlying pixel data if necessary. You may omit this
   *     function if the requested conversion does not change the pixel format.
   * @return The current image as a WebGLTexture.
   */
  getImage(
      type: MPImageStorageType.WEBGL_TEXTURE,
      converter?: MPImageChannelConverter): WebGLTexture;
  getImage(type?: MPImageStorageType, converter?: MPImageChannelConverter):
      MPImageNativeContainer {
    const internalConverter = {...DEFAULT_CONVERTER, ...converter};
    switch (type) {
      case MPImageStorageType.UINT8_CLAMPED_ARRAY:
        return this.convertToUint8ClampedArray(internalConverter);
      case MPImageStorageType.FLOAT32_ARRAY:
        return this.convertToFloat32Array(internalConverter);
      case MPImageStorageType.IMAGE_DATA:
        return this.convertToImageData(internalConverter);
      case MPImageStorageType.IMAGE_BITMAP:
        return this.convertToImageBitmap(internalConverter);
      case MPImageStorageType.WEBGL_TEXTURE:
        return this.convertToWebGLTexture(internalConverter);
      default:
        throw new Error(`Type is not supported: ${type}`);
    }
  }


  private getContainer(type: MPImageStorageType.UINT8_CLAMPED_ARRAY):
      Uint8ClampedArray|undefined;
  private getContainer(type: MPImageStorageType.FLOAT32_ARRAY): Float32Array
      |undefined;
  private getContainer(type: MPImageStorageType.IMAGE_DATA): ImageData
      |undefined;
  private getContainer(type: MPImageStorageType.IMAGE_BITMAP): ImageBitmap
      |undefined;
  private getContainer(type: MPImageStorageType.WEBGL_TEXTURE): WebGLTexture
      |undefined;
  private getContainer(type: MPImageStorageType): MPImageNativeContainer
      |undefined;
  /** Returns the container for the requested storage type iff it exists. */
  private getContainer(type: MPImageStorageType): MPImageNativeContainer
      |undefined {
    switch (type) {
      case MPImageStorageType.UINT8_CLAMPED_ARRAY:
        return this.containers.find(img => img instanceof Uint8ClampedArray);
      case MPImageStorageType.FLOAT32_ARRAY:
        return this.containers.find(img => img instanceof Float32Array);
      case MPImageStorageType.IMAGE_DATA:
        return this.containers.find(img => img instanceof ImageData);
      case MPImageStorageType.IMAGE_BITMAP:
        return this.containers.find(img => img instanceof ImageBitmap);
      case MPImageStorageType.WEBGL_TEXTURE:
        return this.containers.find(img => img instanceof WebGLTexture);
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
   */
  clone(): MPImage {
    const destinationContainers: MPImageNativeContainer[] = [];

    // TODO: We might only want to clone one backing datastructure
    // even if multiple are defined;
    for (const container of this.containers) {
      let destinationContainer: MPImageNativeContainer;

      if (container instanceof Uint8ClampedArray) {
        destinationContainer = new Uint8ClampedArray(container);
      } else if (container instanceof Float32Array) {
        destinationContainer = new Float32Array(container);
      } else if (container instanceof ImageData) {
        destinationContainer =
            new ImageData(container.data, this.width, this.height);
      } else if (container instanceof WebGLTexture) {
        const gl = this.getGL();
        const shaderContext = this.getShaderContext();

        // Create a new texture and use it to back a framebuffer
        gl.activeTexture(gl.TEXTURE1);
        destinationContainer =
            assertNotNull(gl.createTexture(), 'Failed to create texture');
        gl.bindTexture(gl.TEXTURE_2D, destinationContainer);

        gl.texImage2D(
            gl.TEXTURE_2D, 0, gl.RGBA, this.width, this.height, 0, gl.RGBA,
            gl.UNSIGNED_BYTE, null);

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
        this.convertToWebGLTexture(DEFAULT_CONVERTER);
        this.bindTexture();
        destinationContainer = this.copyTextureToBitmap();
        this.unbindTexture();
      } else {
        throw new Error(`Type is not supported: ${container}`);
      }

      destinationContainers.push(destinationContainer);
    }

    return new MPImage(
        destinationContainers, this.hasType(MPImageStorageType.IMAGE_BITMAP),
        this.hasType(MPImageStorageType.WEBGL_TEXTURE), this.canvas,
        this.shaderContext, this.width, this.height);
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
          'is passed when iniitializing the image.');
    }
    if (!this.gl) {
      this.gl = assertNotNull(
          this.canvas.getContext('webgl2') as WebGL2RenderingContext | null,
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

  private convertToImageBitmap(converter: Required<MPImageChannelConverter>):
      ImageBitmap {
    let imageBitmap = this.getContainer(MPImageStorageType.IMAGE_BITMAP);
    if (!imageBitmap) {
      this.convertToWebGLTexture(converter);
      imageBitmap = this.convertWebGLTextureToImageBitmap();
      this.containers.push(imageBitmap);
      this.ownsImageBitmap = true;
    }

    return imageBitmap;
  }

  private convertToImageData(converter: Required<MPImageChannelConverter>):
      ImageData {
    let imageData = this.getContainer(MPImageStorageType.IMAGE_DATA);
    if (!imageData) {
      if (this.hasType(MPImageStorageType.UINT8_CLAMPED_ARRAY)) {
        const source =
            this.getContainer(MPImageStorageType.UINT8_CLAMPED_ARRAY)!;
        const destination = new Uint8ClampedArray(this.width * this.height * 4);
        for (let i = 0; i < this.width * this.height; i++) {
          const rgba = converter.uint8ToRGBAConverter(source[i]);
          destination[i * 4] = rgba[0];
          destination[i * 4 + 1] = rgba[1];
          destination[i * 4 + 2] = rgba[2];
          destination[i * 4 + 3] = rgba[3];
        }
        imageData = new ImageData(destination, this.width, this.height);
        this.containers.push(imageData);
      } else if (this.hasType(MPImageStorageType.FLOAT32_ARRAY)) {
        const source = this.getContainer(MPImageStorageType.FLOAT32_ARRAY)!;
        const destination = new Uint8ClampedArray(this.width * this.height * 4);
        for (let i = 0; i < this.width * this.height; i++) {
          const rgba = converter.floatToRGBAConverter(source[i]);
          destination[i * 4] = rgba[0];
          destination[i * 4 + 1] = rgba[1];
          destination[i * 4 + 2] = rgba[2];
          destination[i * 4 + 3] = rgba[3];
        }
        imageData = new ImageData(destination, this.width, this.height);
        this.containers.push(imageData);
      } else if (
          this.hasType(MPImageStorageType.IMAGE_BITMAP) ||
          this.hasType(MPImageStorageType.WEBGL_TEXTURE)) {
        const gl = this.getGL();
        const shaderContext = this.getShaderContext();
        const pixels = new Uint8Array(this.width * this.height * 4);

        // Create texture if needed
        const webGlTexture = this.convertToWebGLTexture(converter);

        // Create a framebuffer from the texture and read back pixels
        shaderContext.bindFramebuffer(gl, webGlTexture);
        gl.readPixels(
            0, 0, this.width, this.height, gl.RGBA, gl.UNSIGNED_BYTE, pixels);
        shaderContext.unbindFramebuffer();

        imageData = new ImageData(
            new Uint8ClampedArray(pixels.buffer), this.width, this.height);
        this.containers.push(imageData);
      } else {
        throw new Error('Couldn\t find backing image for ImageData conversion');
      }
    }

    return imageData;
  }

  private convertToUint8ClampedArray(
      converter: Required<MPImageChannelConverter>): Uint8ClampedArray {
    let uint8ClampedArray =
        this.getContainer(MPImageStorageType.UINT8_CLAMPED_ARRAY);
    if (!uint8ClampedArray) {
      if (this.hasType(MPImageStorageType.FLOAT32_ARRAY)) {
        const source = this.getContainer(MPImageStorageType.FLOAT32_ARRAY)!;
        uint8ClampedArray = new Uint8ClampedArray(
            source.map(v => converter.floatToUint8Converter(v)));
      } else {
        const source = this.convertToImageData(converter).data;
        uint8ClampedArray = new Uint8ClampedArray(this.width * this.height);
        for (let i = 0; i < this.width * this.height; i++) {
          uint8ClampedArray[i] = converter.rgbaToUint8Converter(
              source[i * 4], source[i * 4 + 1], source[i * 4 + 2],
              source[i * 4 + 3]);
        }
      }
      this.containers.push(uint8ClampedArray);
    }

    return uint8ClampedArray;
  }

  private convertToFloat32Array(converter: Required<MPImageChannelConverter>):
      Float32Array {
    let float32Array = this.getContainer(MPImageStorageType.FLOAT32_ARRAY);
    if (!float32Array) {
      if (this.hasType(MPImageStorageType.UINT8_CLAMPED_ARRAY)) {
        const source =
            this.getContainer(MPImageStorageType.UINT8_CLAMPED_ARRAY)!;
        float32Array = new Float32Array(source).map(
            v => converter.uint8ToFloatConverter(v));
      } else {
        const source = this.convertToImageData(converter).data;
        float32Array = new Float32Array(this.width * this.height);
        for (let i = 0; i < this.width * this.height; i++) {
          float32Array[i] = converter.rgbaToFloatConverter(
              source[i * 4], source[i * 4 + 1], source[i * 4 + 2],
              source[i * 4 + 3]);
        }
      }
      this.containers.push(float32Array);
    }

    return float32Array;
  }

  private convertToWebGLTexture(converter: Required<MPImageChannelConverter>):
      WebGLTexture {
    let webGLTexture = this.getContainer(MPImageStorageType.WEBGL_TEXTURE);
    if (!webGLTexture) {
      const gl = this.getGL();
      webGLTexture = this.bindTexture();
      const source = this.getContainer(MPImageStorageType.IMAGE_BITMAP) ||
          this.convertToImageData(converter);
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

    let webGLTexture = this.getContainer(MPImageStorageType.WEBGL_TEXTURE);
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
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);

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
   */
  close(): void {
    if (this.ownsImageBitmap) {
      this.getContainer(MPImageStorageType.IMAGE_BITMAP)!.close();
    }

    if (this.ownsWebGLTexture) {
      const gl = this.getGL();
      gl.deleteTexture(this.getContainer(MPImageStorageType.WEBGL_TEXTURE)!);
    }
  }
}
