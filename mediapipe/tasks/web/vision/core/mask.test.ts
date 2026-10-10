/**
 * Copyright 2022 The MediaPipe Authors.
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

import 'jasmine';

import {MPImageShaderContext} from './image_shader_context';
import {MPMask} from './mask';

const WIDTH = 2;
const HEIGHT = 2;

const skip = typeof document === 'undefined';
if (skip) {
  console.log('These tests must be run in a browser.');
}

/** The mask types supported by MPMask. */
type MaskType = Uint8Array|Float32Array|WebGLTexture;

const MASK_2_1 = [1, 2];
const MASK_2_2 = [1, 2, 3, 4];
const MASK_2_3 = [1, 2, 3, 4, 5, 6];

/** The test images and data to use for the unit tests below. */
class MPMaskTestContext {
  canvas!: OffscreenCanvas;
  gl!: WebGL2RenderingContext;
  uint8Array!: Uint8Array;
  float32Array!: Float32Array;
  webGLTexture!: WebGLTexture;

  async init(pixels = MASK_2_2, width = WIDTH, height = HEIGHT): Promise<void> {
    // Initialize a canvas with default dimensions. Note that the canvas size
    // can be different from the mask size.
    this.canvas = new OffscreenCanvas(WIDTH, HEIGHT);
    this.gl = this.canvas.getContext('webgl2') as WebGL2RenderingContext;

    const gl = this.gl;
    if (!gl.getExtension('EXT_color_buffer_float')) {
      throw new Error('Missing required EXT_color_buffer_float extension');
    }

    this.uint8Array = new Uint8Array(pixels);
    this.float32Array = new Float32Array(pixels.length);
    for (let i = 0; i < this.uint8Array.length; ++i) {
      this.float32Array[i] = pixels[i] / 255;
    }

    this.webGLTexture = gl.createTexture()!;
    gl.bindTexture(gl.TEXTURE_2D, this.webGLTexture);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
    gl.texImage2D(
        gl.TEXTURE_2D, 0, gl.R32F, width, height, 0, gl.RED, gl.FLOAT,
        new Float32Array(pixels).map(v => v / 255));
    gl.bindTexture(gl.TEXTURE_2D, null);
  }

  get(type: unknown) {
    switch (type) {
      case Uint8Array:
        return this.uint8Array;
      case Float32Array:
        return this.float32Array;
      case WebGLTexture:
        return this.webGLTexture;
      default:
        throw new Error(`Unsupported  type: ${type}`);
    }
  }

  close(): void {
    this.gl.deleteTexture(this.webGLTexture);
  }
}

(skip ? xdescribe : describe)('MPMask', () => {
  const context = new MPMaskTestContext();

  afterEach(() => {
    context.close();
  });

  function readPixelsFromWebGLTexture(texture: WebGLTexture): Float32Array {
    const pixels = new Float32Array(WIDTH * HEIGHT);

    const gl = context.gl;
    gl.bindTexture(gl.TEXTURE_2D, texture);

    const framebuffer = gl.createFramebuffer()!;
    gl.bindFramebuffer(gl.FRAMEBUFFER, framebuffer);
    gl.framebufferTexture2D(
        gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, texture, 0);
    gl.readPixels(0, 0, WIDTH, HEIGHT, gl.RED, gl.FLOAT, pixels);
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    gl.deleteFramebuffer(framebuffer);

    gl.bindTexture(gl.TEXTURE_2D, null);

    // Sanity check values
    expect(pixels[0]).not.toBe(0);

    return pixels;
  }

  function assertEquality(mask: MPMask, expected: MaskType): void {
    if (expected instanceof Uint8Array) {
      const result = mask.getAsUint8Array();
      expect(result).toEqual(expected);
    } else if (expected instanceof Float32Array) {
      const result = mask.getAsFloat32Array();
      expect(result).toEqual(expected);
    } else {  // WebGLTexture
      const result = mask.getAsWebGLTexture();
      expect(readPixelsFromWebGLTexture(result))
          .toEqual(readPixelsFromWebGLTexture(expected));
    }
  }

  function createImage(
      shaderContext: MPImageShaderContext, input: MaskType, width: number,
      height: number): MPMask {
    return new MPMask(
        [input], /* interpolateValues= */ false,
        /* ownsWebGLTexture= */ false, context.canvas, shaderContext, width,
        height);
  }

  function runConversionTest(
      input: MaskType, output: MaskType, width = WIDTH, height = HEIGHT): void {
    const shaderContext = new MPImageShaderContext();
    const mask = createImage(shaderContext, input, width, height);
    assertEquality(mask, output);
    mask.close();
    shaderContext.close();
  }

  function runCloneTest(input: MaskType): void {
    const shaderContext = new MPImageShaderContext();
    const mask = createImage(shaderContext, input, WIDTH, HEIGHT);
    const clone = mask.clone();
    assertEquality(clone, input);
    clone.close();
    shaderContext.close();
  }

  const sources = skip ? [] : [Uint8Array, Float32Array, WebGLTexture];

  for (let i = 0; i < sources.length; i++) {
    for (let j = 0; j < sources.length; j++) {
      it(`converts from ${sources[i].name} to ${sources[j].name}`, async () => {
        await context.init();
        runConversionTest(context.get(sources[i]), context.get(sources[j]));
      });
    }
  }

  for (let i = 0; i < sources.length; i++) {
    it(`clones ${sources[i].name}`, async () => {
      await context.init();
      runCloneTest(context.get(sources[i]));
    });
  }

  it(`does not flip textures twice`, async () => {
    await context.init();

    const shaderContext = new MPImageShaderContext();
    const mask = new MPMask(
        [context.webGLTexture], /* interpolateValues= */ false,
        /* ownsWebGLTexture= */ false, context.canvas, shaderContext, WIDTH,
        HEIGHT);

    const result = mask.clone().getAsUint8Array();
    expect(result).toEqual(context.uint8Array);
    shaderContext.close();
  });

  it(`can clone and get mask`, async () => {
    await context.init();

    const shaderContext = new MPImageShaderContext();
    const mask = new MPMask(
        [context.webGLTexture], /* interpolateValues= */ false,
        /* ownsWebGLTexture= */ false, context.canvas, shaderContext, WIDTH,
        HEIGHT);

    // Verify that we can mix the different shader modes by running them out of
    // order.
    let result = mask.getAsUint8Array();
    expect(result).toEqual(context.uint8Array);

    result = mask.clone().getAsUint8Array();
    expect(result).toEqual(context.uint8Array);

    result = mask.getAsUint8Array();
    expect(result).toEqual(context.uint8Array);

    shaderContext.close();
  });

  it('supports has()', async () => {
    await context.init();

    const shaderContext = new MPImageShaderContext();
    const mask = createImage(shaderContext, context.uint8Array, WIDTH, HEIGHT);

    expect(mask.hasUint8Array()).toBe(true);
    expect(mask.hasFloat32Array()).toBe(false);
    expect(mask.hasWebGLTexture()).toBe(false);
    expect(mask.hasImageBitmap()).toBe(false);

    mask.getAsFloat32Array();

    expect(mask.hasUint8Array()).toBe(true);
    expect(mask.hasFloat32Array()).toBe(true);
    expect(mask.hasWebGLTexture()).toBe(false);
    expect(mask.hasImageBitmap()).toBe(false);

    mask.getAsWebGLTexture();

    expect(mask.hasUint8Array()).toBe(true);
    expect(mask.hasFloat32Array()).toBe(true);
    expect(mask.hasWebGLTexture()).toBe(true);
    expect(mask.hasImageBitmap()).toBe(false);

    mask.getAsImageBitmap();

    expect(mask.hasUint8Array()).toBe(true);
    expect(mask.hasFloat32Array()).toBe(true);
    expect(mask.hasWebGLTexture()).toBe(true);
    expect(mask.hasImageBitmap()).toBe(true);

    mask.close();
    shaderContext.close();
  });

  it('supports mask that is smaller than the canvas', async () => {
    await context.init(MASK_2_1, /* width= */ 2, /* height= */ 1);

    runConversionTest(
        context.uint8Array, context.webGLTexture, /* width= */ 2,
        /* height= */ 1);
    runConversionTest(
        context.webGLTexture, context.float32Array, /* width= */ 2,
        /* height= */ 1);
    runConversionTest(
        context.float32Array, context.uint8Array, /* width= */ 2,
        /* height= */ 1);

    context.close();
  });

  it('supports mask that is larger than the canvas', async () => {
    await context.init(MASK_2_3, /* width= */ 2, /* height= */ 3);

    runConversionTest(
        context.uint8Array, context.webGLTexture, /* width= */ 2,
        /* height= */ 3);
    runConversionTest(
        context.webGLTexture, context.float32Array, /* width= */ 2,
        /* height= */ 3);
    runConversionTest(
        context.float32Array, context.uint8Array, /* width= */ 2,
        /* height= */ 3);
  });

  function readPixelsFromImageBitmap(imageBitmap: ImageBitmap): ImageData {
    const canvas = new OffscreenCanvas(imageBitmap.width, imageBitmap.height);
    const ctx = canvas.getContext('2d') as OffscreenCanvasRenderingContext2D;
    ctx.drawImage(imageBitmap, 0, 0);
    return ctx.getImageData(0, 0, imageBitmap.width, imageBitmap.height);
  }

  function expectRedsCloseTo(reds: number[], expected: number[]): void {
    const sorted = reds.slice().sort((a, b) => a - b);
    const exp = expected.slice().sort((a, b) => a - b);
    expect(sorted.length).toEqual(exp.length);
    for (let i = 0; i < exp.length; i++) {
      expect(Math.abs(sorted[i] - exp[i])).toBeLessThanOrEqual(1);
    }
  }

  it('getAsImageBitmap returns grayscale RGBA from Uint8Array', async () => {
    await context.init([255, 0, 128, 64]);

    const shaderContext = new MPImageShaderContext();
    const mask = createImage(shaderContext, context.uint8Array, WIDTH, HEIGHT);
    const bitmap = mask.getAsImageBitmap();

    expect(bitmap.width).toEqual(WIDTH);
    expect(bitmap.height).toEqual(HEIGHT);
    expect(mask.hasImageBitmap()).toBe(true);

    const pixels = readPixelsFromImageBitmap(bitmap).data;
    const reds: number[] = [];
    for (let i = 0; i < pixels.length; i += 4) {
      expect(pixels[i]).toEqual(pixels[i + 1]);
      expect(pixels[i]).toEqual(pixels[i + 2]);
      expect(pixels[i]).toEqual(pixels[i + 3]);
      reds.push(pixels[i]);
    }
    expectRedsCloseTo(reds, [0, 64, 128, 255]);

    mask.close();
    shaderContext.close();
  });

  it('getAsImageBitmap converts a WebGLTexture without throwing', async () => {
    await context.init([255, 0, 128, 64]);

    const shaderContext = new MPImageShaderContext();
    const mask =
        createImage(shaderContext, context.webGLTexture, WIDTH, HEIGHT);
    const bitmap = mask.getAsImageBitmap();

    expect(bitmap.width).toEqual(WIDTH);
    expect(bitmap.height).toEqual(HEIGHT);
    const pixels = readPixelsFromImageBitmap(bitmap).data;
    for (let i = 0; i < pixels.length; i += 4) {
      expect(pixels[i]).toEqual(pixels[i + 1]);
      expect(pixels[i]).toEqual(pixels[i + 2]);
      expect(pixels[i]).toEqual(pixels[i + 3]);
    }
    const reds: number[] = [];
    for (let i = 0; i < pixels.length; i += 4) {
      reds.push(pixels[i]);
    }
    expectRedsCloseTo(reds, [0, 64, 128, 255]);

    mask.close();
    shaderContext.close();
  });

  it('getAsImageBitmap is cached until close', async () => {
    await context.init();

    const shaderContext = new MPImageShaderContext();
    const mask = createImage(shaderContext, context.uint8Array, WIDTH, HEIGHT);
    const first = mask.getAsImageBitmap();
    const second = mask.getAsImageBitmap();
    expect(second).toBe(first);

    mask.close();
    shaderContext.close();
  });

  it('clone owns a copy of the ImageBitmap', async () => {
    await context.init([255, 0, 0, 0]);

    const shaderContext = new MPImageShaderContext();
    const mask = createImage(shaderContext, context.uint8Array, WIDTH, HEIGHT);
    mask.getAsImageBitmap();
    const clone = mask.clone();
    expect(clone.hasImageBitmap()).toBe(true);
    expect(clone.getAsImageBitmap().width).toEqual(WIDTH);
    expect(clone.getAsImageBitmap()).not.toBe(mask.getAsImageBitmap());

    clone.close();
    mask.close();
    shaderContext.close();
  });

  it('getAsImageBitmap throws without a canvas', () => {
    const mask = new MPMask(
        [new Uint8Array([1, 2, 3, 4])], /* interpolateValues= */ false,
        /* ownsWebGLTexture= */ false, /* canvas= */ undefined,
        /* shaderContext= */ undefined, WIDTH, HEIGHT);
    expect(() => {
      mask.getAsImageBitmap();
    }).toThrowError(/canvas/);
  });
});
