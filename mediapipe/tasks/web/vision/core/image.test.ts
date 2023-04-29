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

import {MPImage, MPImageShaderContext, MPImageType} from './image';

const WIDTH = 2;
const HEIGHT = 2;

const skip = typeof document === 'undefined';
if (skip) {
  console.log('These tests must be run in a browser.');
}

/** The image types supported by MPImage. */
type ImageType = ImageData|ImageBitmap|WebGLTexture;

const IMAGE_2_2 = [1, 1, 1, 255, 2, 2, 2, 255, 3, 3, 3, 255, 4, 4, 4, 255];
const IMAGE_2_1 = [1, 1, 1, 255, 2, 2, 2, 255];
const IMAGE_2_3 = [
  1, 1, 1, 255, 2, 2, 2, 255, 3, 3, 3, 255,
  4, 4, 4, 255, 5, 5, 5, 255, 6, 6, 6, 255
];

/** The test images and data to use for the unit tests below. */
class MPImageTestContext {
  canvas!: OffscreenCanvas;
  gl!: WebGL2RenderingContext;
  uint8ClampedArray!: Uint8ClampedArray;
  float32Array!: Float32Array;
  imageData!: ImageData;
  imageBitmap!: ImageBitmap;
  webGLTexture!: WebGLTexture;

  async init(pixels = IMAGE_2_2, width = WIDTH, height = HEIGHT):
      Promise<void> {
    // Initialize a canvas with default dimensions. Note that the canvas size
    // can be different from the image size.
    this.canvas = new OffscreenCanvas(WIDTH, HEIGHT);
    this.gl = this.canvas.getContext('webgl2') as WebGL2RenderingContext;

    const gl = this.gl;

    this.uint8ClampedArray = new Uint8ClampedArray(pixels.length / 4);
    this.float32Array = new Float32Array(pixels.length / 4);
    for (let i = 0; i < this.uint8ClampedArray.length; ++i) {
      this.uint8ClampedArray[i] = pixels[i * 4];
      this.float32Array[i] = pixels[i * 4] / 255;
    }
    this.imageData =
        new ImageData(new Uint8ClampedArray(pixels), width, height);
    this.imageBitmap = await createImageBitmap(this.imageData);
    this.webGLTexture = gl.createTexture()!;

    gl.bindTexture(gl.TEXTURE_2D, this.webGLTexture);
    gl.texImage2D(
        gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, this.imageBitmap);
    gl.bindTexture(gl.TEXTURE_2D, null);
  }

  get(type: unknown) {
    switch (type) {
      case Uint8ClampedArray:
        return this.uint8ClampedArray;
      case Float32Array:
        return this.float32Array;
      case ImageData:
        return this.imageData;
      case ImageBitmap:
        return this.imageBitmap;
      case WebGLTexture:
        return this.webGLTexture;
      default:
        throw new Error(`Unsupported  type: ${type}`);
    }
  }

  close(): void {
    this.gl.deleteTexture(this.webGLTexture);
    this.imageBitmap.close();
  }
}

(skip ? xdescribe : describe)('MPImage', () => {
  const context = new MPImageTestContext();

  afterEach(() => {
    context.close();
  });

  function readPixelsFromImageBitmap(imageBitmap: ImageBitmap): ImageData {
    const canvas = new OffscreenCanvas(imageBitmap.width, imageBitmap.height);
    const ctx = canvas.getContext('2d') as OffscreenCanvasRenderingContext2D;
    ctx.drawImage(imageBitmap, 0, 0);
    return ctx.getImageData(0, 0, imageBitmap.width, imageBitmap.height);
  }

  function readPixelsFromWebGLTexture(texture: WebGLTexture): Uint8Array {
    const pixels = new Uint8Array(WIDTH * WIDTH * 4);

    const gl = context.gl;
    gl.bindTexture(gl.TEXTURE_2D, texture);

    const framebuffer = gl.createFramebuffer()!;
    gl.bindFramebuffer(gl.FRAMEBUFFER, framebuffer);
    gl.framebufferTexture2D(
        gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, texture, 0);
    gl.readPixels(0, 0, WIDTH, HEIGHT, gl.RGBA, gl.UNSIGNED_BYTE, pixels);
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    gl.deleteFramebuffer(framebuffer);

    gl.bindTexture(gl.TEXTURE_2D, null);

    return pixels;
  }

  function assertEquality(image: MPImage, expected: ImageType): void {
    if (expected instanceof Uint8ClampedArray) {
      const result = image.get(MPImageType.UINT8_CLAMPED_ARRAY);
      expect(result).toEqual(expected);
    } else if (expected instanceof Float32Array) {
      const result = image.get(MPImageType.FLOAT32_ARRAY);
      expect(result).toEqual(expected);
    } else if (expected instanceof ImageData) {
      const result = image.get(MPImageType.IMAGE_DATA);
      expect(result).toEqual(expected);
    } else if (expected instanceof ImageBitmap) {
      const result = image.get(MPImageType.IMAGE_BITMAP);
      expect(readPixelsFromImageBitmap(result))
          .toEqual(readPixelsFromImageBitmap(expected));
    } else {  // WebGLTexture
      const result = image.get(MPImageType.WEBGL_TEXTURE);
      expect(readPixelsFromWebGLTexture(result))
          .toEqual(readPixelsFromWebGLTexture(expected));
    }
  }

  function createImage(
      shaderContext: MPImageShaderContext, input: ImageType, width: number,
      height: number): MPImage {
    return new MPImage(
        [input],
        /* ownsImageBitmap= */ false, /* ownsWebGLTexture= */ false,
        context.canvas, shaderContext, width, height);
  }

  function runConversionTest(
      input: ImageType, output: ImageType, width = WIDTH,
      height = HEIGHT): void {
    const shaderContext = new MPImageShaderContext();
    const image = createImage(shaderContext, input, width, height);
    assertEquality(image, output);
    image.close();
    shaderContext.close();
  }

  function runCloneTest(input: ImageType): void {
    const shaderContext = new MPImageShaderContext();
    const image = createImage(shaderContext, input, WIDTH, HEIGHT);
    const clone = image.clone();
    assertEquality(clone, input);
    clone.close();
    shaderContext.close();
  }

  const sources = skip ?
      [] :
      [Uint8ClampedArray, Float32Array, ImageData, ImageBitmap, WebGLTexture];

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
    const image = new MPImage(
        [context.webGLTexture],
        /* ownsImageBitmap= */ false, /* ownsWebGLTexture= */ false,
        context.canvas, shaderContext, WIDTH, HEIGHT);

    const result = image.clone().get(MPImageType.IMAGE_DATA);
    expect(result).toEqual(context.imageData);

    shaderContext.close();
  });

  it(`can clone and get image`, async () => {
    await context.init();

    const shaderContext = new MPImageShaderContext();
    const image = new MPImage(
        [context.webGLTexture],
        /* ownsImageBitmap= */ false, /* ownsWebGLTexture= */ false,
        context.canvas, shaderContext, WIDTH, HEIGHT);

    // Verify that we can mix the different shader modes by running them out of
    // order.
    let result = image.get(MPImageType.IMAGE_DATA);
    expect(result).toEqual(context.imageData);

    result = image.clone().get(MPImageType.IMAGE_DATA);
    expect(result).toEqual(context.imageData);

    result = image.get(MPImageType.IMAGE_DATA);
    expect(result).toEqual(context.imageData);

    shaderContext.close();
  });

  it('supports has()', async () => {
    await context.init();

    const shaderContext = new MPImageShaderContext();
    const image = createImage(shaderContext, context.imageData, WIDTH, HEIGHT);

    expect(image.has(MPImageType.IMAGE_DATA)).toBe(true);
    expect(image.has(MPImageType.UINT8_CLAMPED_ARRAY)).toBe(false);
    expect(image.has(MPImageType.FLOAT32_ARRAY)).toBe(false);
    expect(image.has(MPImageType.WEBGL_TEXTURE)).toBe(false);
    expect(image.has(MPImageType.IMAGE_BITMAP)).toBe(false);

    image.get(MPImageType.UINT8_CLAMPED_ARRAY);

    expect(image.has(MPImageType.IMAGE_DATA)).toBe(true);
    expect(image.has(MPImageType.UINT8_CLAMPED_ARRAY)).toBe(true);
    expect(image.has(MPImageType.FLOAT32_ARRAY)).toBe(false);
    expect(image.has(MPImageType.WEBGL_TEXTURE)).toBe(false);
    expect(image.has(MPImageType.IMAGE_BITMAP)).toBe(false);

    image.get(MPImageType.FLOAT32_ARRAY);

    expect(image.has(MPImageType.IMAGE_DATA)).toBe(true);
    expect(image.has(MPImageType.UINT8_CLAMPED_ARRAY)).toBe(true);
    expect(image.has(MPImageType.FLOAT32_ARRAY)).toBe(true);
    expect(image.has(MPImageType.WEBGL_TEXTURE)).toBe(false);
    expect(image.has(MPImageType.IMAGE_BITMAP)).toBe(false);

    image.get(MPImageType.WEBGL_TEXTURE);

    expect(image.has(MPImageType.IMAGE_DATA)).toBe(true);
    expect(image.has(MPImageType.UINT8_CLAMPED_ARRAY)).toBe(true);
    expect(image.has(MPImageType.FLOAT32_ARRAY)).toBe(true);
    expect(image.has(MPImageType.WEBGL_TEXTURE)).toBe(true);
    expect(image.has(MPImageType.IMAGE_BITMAP)).toBe(false);

    image.get(MPImageType.IMAGE_BITMAP);

    expect(image.has(MPImageType.IMAGE_DATA)).toBe(true);
    expect(image.has(MPImageType.UINT8_CLAMPED_ARRAY)).toBe(true);
    expect(image.has(MPImageType.FLOAT32_ARRAY)).toBe(true);
    expect(image.has(MPImageType.WEBGL_TEXTURE)).toBe(true);
    expect(image.has(MPImageType.IMAGE_BITMAP)).toBe(true);

    image.close();
    shaderContext.close();
  });

  it('supports image that is smaller than the canvas', async () => {
    await context.init(IMAGE_2_1, /* width= */ 2, /* height= */ 1);

    runConversionTest(
        context.imageData, context.webGLTexture, /* width= */ 2,
        /* height= */ 1);
    runConversionTest(
        context.webGLTexture, context.imageBitmap, /* width= */ 2,
        /* height= */ 1);
    runConversionTest(
        context.imageBitmap, context.imageData, /* width= */ 2,
        /* height= */ 1);

    context.close();
  });

  it('supports image that is larger than the canvas', async () => {
    await context.init(IMAGE_2_3, /* width= */ 2, /* height= */ 3);

    runConversionTest(
        context.imageData, context.webGLTexture, /* width= */ 2,
        /* height= */ 3);
    runConversionTest(
        context.webGLTexture, context.imageBitmap, /* width= */ 2,
        /* height= */ 3);
    runConversionTest(
        context.imageBitmap, context.imageData, /* width= */ 2,
        /* height= */ 3);
  });
});
