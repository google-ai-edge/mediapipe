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

/** Helper to assert that `value` is not null or undefined.  */
export function assertExists<T>(value: T, msg: string): NonNullable<T> {
  if (!value) {
    throw new Error(`Unable to obtain required WebGL resource: ${msg}`);
  }
  return value;
}

/**
 * Utility class that encapsulates the buffers used by `MPImageShaderContext`.
 * For internal use only.
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
 *
 * For internal use only.
 */
export class MPImageShaderContext {
  protected gl?: WebGL2RenderingContext;
  private framebuffer?: WebGLFramebuffer;
  protected program?: WebGLProgram;
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

  protected getFragmentShader(): string {
    return FRAGMENT_SHADER;
  }

  protected getVertexShader(): string {
    return VERTEX_SHADER;
  }

  private compileShader(source: string, type: number): WebGLShader {
    const gl = this.gl!;
    const shader =
        assertExists(gl.createShader(type), 'Failed to create WebGL shader');
    gl.shaderSource(shader, source);
    gl.compileShader(shader);
    if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
      const info = gl.getShaderInfoLog(shader);
      throw new Error(`Could not compile WebGL shader: ${info}`);
    }
    gl.attachShader(this.program!, shader);
    return shader;
  }

  protected setupShaders(): void {
    const gl = this.gl!;
    this.program =
        assertExists(gl.createProgram()!, 'Failed to create WebGL program');

    this.vertexShader =
        this.compileShader(this.getVertexShader(), gl.VERTEX_SHADER);
    this.fragmentShader =
        this.compileShader(this.getFragmentShader(), gl.FRAGMENT_SHADER);

    gl.linkProgram(this.program);
    const linked = gl.getProgramParameter(this.program, gl.LINK_STATUS);
    if (!linked) {
      const info = gl.getProgramInfoLog(this.program);
      throw new Error(`Error during program linking: ${info}`);
    }

    this.aVertex = gl.getAttribLocation(this.program, 'aVertex');
    this.aTex = gl.getAttribLocation(this.program, 'aTex');
  }

  protected setupTextures(): void {}

  protected configureUniforms(): void {}

  private createBuffers(flipVertically: boolean): MPImageShaderBuffers {
    const gl = this.gl!;
    const vertexArrayObject =
        assertExists(gl.createVertexArray(), 'Failed to create vertex array');
    gl.bindVertexArray(vertexArrayObject);

    const vertexBuffer =
        assertExists(gl.createBuffer(), 'Failed to create buffer');
    gl.bindBuffer(gl.ARRAY_BUFFER, vertexBuffer);
    gl.enableVertexAttribArray(this.aVertex!);
    gl.vertexAttribPointer(this.aVertex!, 2, gl.FLOAT, false, 0, 0);
    gl.bufferData(
        gl.ARRAY_BUFFER, new Float32Array([-1, -1, -1, 1, 1, 1, 1, -1]),
        gl.STATIC_DRAW);

    const textureBuffer =
        assertExists(gl.createBuffer(), 'Failed to create buffer');
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
      this.setupTextures();
    }

    const shaderBuffers = this.getShaderBuffers(flipVertically);
    gl.useProgram(this.program!);
    shaderBuffers.bind();
    this.configureUniforms();
    const result = callback();
    shaderBuffers.unbind();

    return result;
  }

  /**
   * Creates and configures a texture.
   *
   * @param gl The rendering context.
   * @param filter The setting to use for `gl.TEXTURE_MIN_FILTER` and
   *     `gl.TEXTURE_MAG_FILTER`. Defaults to `gl.LINEAR`.
   * @param wrapping The setting to use for `gl.TEXTURE_WRAP_S` and
   *     `gl.TEXTURE_WRAP_T`. Defaults to `gl.CLAMP_TO_EDGE`.
   */
  createTexture(gl: WebGL2RenderingContext, filter?: GLenum, wrapping?: GLenum):
      WebGLTexture {
    this.maybeInitGL(gl);
    const texture =
        assertExists(gl.createTexture(), 'Failed to create texture');
    gl.bindTexture(gl.TEXTURE_2D, texture);
    gl.texParameteri(
        gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, wrapping ?? gl.CLAMP_TO_EDGE);
    gl.texParameteri(
        gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, wrapping ?? gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, filter ?? gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, filter ?? gl.LINEAR);
    gl.bindTexture(gl.TEXTURE_2D, null);
    return texture;
  }

  /**
   * Binds a framebuffer to the canvas. If the framebuffer does not yet exist,
   * creates it first. Binds the provided texture to the framebuffer.
   */
  bindFramebuffer(gl: WebGL2RenderingContext, texture: WebGLTexture): void {
    this.maybeInitGL(gl);
    if (!this.framebuffer) {
      this.framebuffer =
          assertExists(gl.createFramebuffer(), 'Failed to create framebuffe.');
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
