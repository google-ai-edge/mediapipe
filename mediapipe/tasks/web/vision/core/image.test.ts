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

import {MPImage, MPImageShaderContext, MPImageStorageType} from './image';

const WIDTH = 2;
const HEIGHT = 2;

const skip = typeof document === 'undefined';
if (skip) {
  console.log('These tests must be run in a browser.');
}

/** The image types supported by MPImage. */
type ImageType = ImageData|ImageBitmap|WebGLTexture;

async function createTestData(
    gl: WebGL2RenderingContext, data: number[], width: number,
    height: number): Promise<[ImageData, ImageBitmap, WebGLTexture]> {
  const imageData = new ImageData(new Uint8ClampedArray(data), width, height);
  const imageBitmap = await createImageBitmap(imageData);
  const webGlTexture = gl.createTexture()!;

  gl.bindTexture(gl.TEXTURE_2D, webGlTexture);
  gl.texImage2D(
      gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, imageBitmap);
  gl.bindTexture(gl.TEXTURE_2D, null);

  return [imageData, imageBitmap, webGlTexture];
}

(skip ? xdescribe : describe)('MPImage', () => {
  let canvas: OffscreenCanvas;
  let gl: WebGL2RenderingContext;
  let imageData: ImageData;
  let imageBitmap: ImageBitmap;
  let webGlTexture: WebGLTexture;

  beforeEach(async () => {
    canvas = new OffscreenCanvas(WIDTH, HEIGHT);
    gl = canvas.getContext('webgl2') as WebGL2RenderingContext;

    const images = await createTestData(
        gl, [1, 0, 0, 255, 2, 0, 0, 255, 3, 0, 0, 255, 4, 0, 0, 255], WIDTH,
        HEIGHT);
    imageData = images[0];
    imageBitmap = images[1];
    webGlTexture = images[2];
  });

  afterEach(() => {
    gl.deleteTexture(webGlTexture);
    imageBitmap.close();
  });

  function readPixelsFromImageBitmap(imageBitmap: ImageBitmap): ImageData {
    const canvas = new OffscreenCanvas(imageBitmap.width, imageBitmap.height);
    const ctx = canvas.getContext('2d') as OffscreenCanvasRenderingContext2D;
    ctx.drawImage(imageBitmap, 0, 0);
    return ctx.getImageData(0, 0, imageBitmap.width, imageBitmap.height);
  }

  function readPixelsFromWebGLTexture(texture: WebGLTexture): Uint8Array {
    const pixels = new Uint8Array(WIDTH * WIDTH * 4);

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
    if (expected instanceof ImageData) {
      const result = image.getImage(MPImageStorageType.IMAGE_DATA);
      expect(result).toEqual(expected);
    } else if (expected instanceof ImageBitmap) {
      const result = image.getImage(MPImageStorageType.IMAGE_BITMAP);
      expect(readPixelsFromImageBitmap(result))
          .toEqual(readPixelsFromImageBitmap(expected));
    } else {  // WebGLTexture
      const result = image.getImage(MPImageStorageType.WEBGL_TEXTURE);
      expect(readPixelsFromWebGLTexture(result))
          .toEqual(readPixelsFromWebGLTexture(expected));
    }
  }

  function createImage(
      shaderContext: MPImageShaderContext, input: ImageType, width: number,
      height: number): MPImage {
    return new MPImage(
        input instanceof ImageData ? input : null,
        input instanceof ImageBitmap ? input : null,
        input instanceof WebGLTexture ? input : null,
        /* ownsImageBitmap= */ false, /* ownsWebGLTexture= */ false, canvas,
        shaderContext, width, height);
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

  it(`converts from ImageData to ImageData`, () => {
    runConversionTest(imageData, imageData);
  });

  it(`converts from ImageData to ImageBitmap`, () => {
    runConversionTest(imageData, imageBitmap);
  });

  it(`converts from ImageData to WebGLTexture`, () => {
    runConversionTest(imageData, webGlTexture);
  });

  it(`converts from ImageBitmap to ImageData`, () => {
    runConversionTest(imageBitmap, imageData);
  });

  it(`converts from ImageBitmap to ImageBitmap`, () => {
    runConversionTest(imageBitmap, imageBitmap);
  });

  it(`converts from ImageBitmap to WebGLTexture`, () => {
    runConversionTest(imageBitmap, webGlTexture);
  });

  it(`converts from WebGLTexture to ImageData`, () => {
    runConversionTest(webGlTexture, imageData);
  });

  it(`converts from WebGLTexture to ImageBitmap`, () => {
    runConversionTest(webGlTexture, imageBitmap);
  });

  it(`converts from WebGLTexture to WebGLTexture`, () => {
    runConversionTest(webGlTexture, webGlTexture);
  });

  it(`clones ImageData`, () => {
    runCloneTest(imageData);
  });

  it(`clones ImageBitmap`, () => {
    runCloneTest(imageBitmap);
  });

  it(`clones WebGLTextures`, () => {
    runCloneTest(webGlTexture);
  });

  it(`does not flip textures twice`, async () => {
    const [imageData, , webGlTexture] = await createTestData(
        gl, [1, 0, 0, 255, 2, 0, 0, 255, 3, 0, 0, 255, 4, 0, 0, 255], WIDTH,
        HEIGHT);

    const shaderContext = new MPImageShaderContext();
    const image = new MPImage(
        /* imageData= */ null, /* imageBitmap= */ null, webGlTexture,
        /* ownsImageBitmap= */ false, /* ownsWebGLTexture= */ false, canvas,
        shaderContext, WIDTH, HEIGHT);

    const result = image.clone().getImage(MPImageStorageType.IMAGE_DATA);
    expect(result).toEqual(imageData);

    gl.deleteTexture(webGlTexture);
    shaderContext.close();
  });

  it(`can clone and get image`, async () => {
    const [imageData, , webGlTexture] = await createTestData(
        gl, [1, 0, 0, 255, 2, 0, 0, 255, 3, 0, 0, 255, 4, 0, 0, 255], WIDTH,
        HEIGHT);

    const shaderContext = new MPImageShaderContext();
    const image = new MPImage(
        /* imageData= */ null, /* imageBitmap= */ null, webGlTexture,
        /* ownsImageBitmap= */ false, /* ownsWebGLTexture= */ false, canvas,
        shaderContext, WIDTH, HEIGHT);

    // Verify that we can mix the different shader modes by running them out of
    // order.
    let result = image.getImage(MPImageStorageType.IMAGE_DATA);
    expect(result).toEqual(imageData);

    result = image.clone().getImage(MPImageStorageType.IMAGE_DATA);
    expect(result).toEqual(imageData);

    result = image.getImage(MPImageStorageType.IMAGE_DATA);
    expect(result).toEqual(imageData);

    gl.deleteTexture(webGlTexture);
    shaderContext.close();
  });

  it('supports hasType()', async () => {
    const shaderContext = new MPImageShaderContext();
    const image = createImage(shaderContext, imageData, WIDTH, HEIGHT);

    expect(image.hasType(MPImageStorageType.IMAGE_DATA)).toBe(true);
    expect(image.hasType(MPImageStorageType.WEBGL_TEXTURE)).toBe(false);
    expect(image.hasType(MPImageStorageType.IMAGE_BITMAP)).toBe(false);

    image.getImage(MPImageStorageType.WEBGL_TEXTURE);

    expect(image.hasType(MPImageStorageType.IMAGE_DATA)).toBe(true);
    expect(image.hasType(MPImageStorageType.WEBGL_TEXTURE)).toBe(true);
    expect(image.hasType(MPImageStorageType.IMAGE_BITMAP)).toBe(false);

    await image.getImage(MPImageStorageType.IMAGE_BITMAP);

    expect(image.hasType(MPImageStorageType.IMAGE_DATA)).toBe(true);
    expect(image.hasType(MPImageStorageType.WEBGL_TEXTURE)).toBe(true);
    expect(image.hasType(MPImageStorageType.IMAGE_BITMAP)).toBe(true);

    image.close();
    shaderContext.close();
  });

  it('supports image that is smaller than the canvas', async () => {
    const [imageData, imageBitmap, webGlTexture] = await createTestData(
        gl, [1, 0, 0, 255, 2, 0, 0, 255], /* width= */ 2, /* height= */ 1);

    runConversionTest(imageData, webGlTexture, /* width= */ 2, /* height= */ 1);
    runConversionTest(
        webGlTexture, imageBitmap, /* width= */ 2, /* height= */ 1);
    runConversionTest(imageBitmap, imageData, /* width= */ 2, /* height= */ 1);

    gl.deleteTexture(webGlTexture);
    imageBitmap.close();
  });

  it('supports image that is larger than the canvas', async () => {
    const [imageData, imageBitmap, webGlTexture] = await createTestData(
        gl,
        [
          1, 0, 0, 255, 2, 0, 0, 255, 3, 0, 0, 255,
          4, 0, 0, 255, 5, 0, 0, 255, 6, 0, 0, 255
        ],
        /* width= */ 2, /* height= */ 3);

    runConversionTest(imageData, webGlTexture, /* width= */ 2, /* height= */ 3);
    runConversionTest(
        webGlTexture, imageBitmap, /* width= */ 2, /* height= */ 3);
    runConversionTest(imageBitmap, imageData, /* width= */ 2, /* height= */ 3);

    gl.deleteTexture(webGlTexture);
    imageBitmap.close();
  });
});
