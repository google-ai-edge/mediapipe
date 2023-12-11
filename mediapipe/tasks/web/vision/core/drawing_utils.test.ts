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

import 'jasmine';

import {DrawingUtils} from './drawing_utils';
import {MPImageShaderContext} from './image_shader_context';
import {MPMask} from './mask';

const WIDTH = 2;
const HEIGHT = 2;

const skip = typeof document === 'undefined';
if (skip) {
  console.log('These tests must be run in a browser.');
}

(skip ? xdescribe : describe)('DrawingUtils', () => {
  let shaderContext = new MPImageShaderContext();
  let canvas2D: OffscreenCanvas;
  let context2D: OffscreenCanvasRenderingContext2D;
  let drawingUtils2D: DrawingUtils;
  let canvasWebGL: OffscreenCanvas;
  let contextWebGL: WebGL2RenderingContext;
  let drawingUtilsWebGL: DrawingUtils;

  beforeEach(() => {
    canvas2D = canvas2D ?? new OffscreenCanvas(WIDTH, HEIGHT);
    canvasWebGL = canvasWebGL ?? new OffscreenCanvas(WIDTH, HEIGHT);

    shaderContext = new MPImageShaderContext();
    contextWebGL = canvasWebGL.getContext('webgl2')!;
    drawingUtilsWebGL = new DrawingUtils(contextWebGL);
    context2D = canvas2D.getContext('2d')!;
    drawingUtils2D = new DrawingUtils(context2D, contextWebGL);
  });

  afterEach(() => {
    shaderContext.close();
    drawingUtils2D.close();
    drawingUtilsWebGL.close();
  });

  describe(
      'drawConfidenceMask() blends background with foreground color', () => {
        const defaultColor = [255, 255, 255, 255];
        const overlayImage = new ImageData(
            new Uint8ClampedArray(
                [0, 0, 0, 255, 0, 0, 0, 255, 0, 0, 0, 255, 0, 0, 0, 255]),
            WIDTH, HEIGHT);
        const expectedResult = new Uint8Array([
          255, 255, 255, 255, 178, 178, 178, 255, 102, 102, 102, 255, 0, 0, 0,
          255
        ]);

        it('on 2D canvas', () => {
          const confidenceMask = new MPMask(
              [new Float32Array([0.0, 0.3, 0.6, 1.0])],
              /* interpolateValues= */ true,
              /* ownsWebGLTexture= */ false, canvas2D, shaderContext, WIDTH,
              HEIGHT);

          drawingUtils2D.drawConfidenceMask(
              confidenceMask, defaultColor, overlayImage);

          const actualResult = context2D.getImageData(0, 0, WIDTH, HEIGHT).data;
          expect(actualResult)
              .toEqual(new Uint8ClampedArray(expectedResult.buffer));
          confidenceMask.close();
        });

        it('on WebGL canvas', () => {
          const confidenceMask = new MPMask(
              [new Float32Array(
                  [0.6, 1.0, 0.0, 0.3])],  // Note: Vertically flipped
              /* interpolateValues= */ true,
              /* ownsWebGLTexture= */ false, canvasWebGL, shaderContext, WIDTH,
              HEIGHT);

          drawingUtilsWebGL.drawConfidenceMask(
              confidenceMask, defaultColor, overlayImage);

          const actualResult = new Uint8Array(WIDTH * HEIGHT * 4);
          contextWebGL.readPixels(
              0, 0, WIDTH, HEIGHT, contextWebGL.RGBA,
              contextWebGL.UNSIGNED_BYTE, actualResult);
          expect(actualResult).toEqual(expectedResult);
          confidenceMask.close();
        });
      });


  describe(
      'drawConfidenceMask() blends background with foreground image', () => {
        const defaultImage = new ImageData(
            new Uint8ClampedArray([
              255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
              255, 255, 255
            ]),
            WIDTH, HEIGHT);
        const overlayImage = new ImageData(
            new Uint8ClampedArray(
                [0, 0, 0, 255, 0, 0, 0, 255, 0, 0, 0, 255, 0, 0, 0, 255]),
            WIDTH, HEIGHT);
        const expectedResult = new Uint8Array([
          255, 255, 255, 255, 178, 178, 178, 255, 102, 102, 102, 255, 0, 0, 0,
          255
        ]);

        it('on 2D canvas', () => {
          const confidenceMask = new MPMask(
              [new Float32Array([0.0, 0.3, 0.6, 1.0])],
              /* interpolateValues= */ true,
              /* ownsWebGLTexture= */ false, canvas2D, shaderContext, WIDTH,
              HEIGHT);

          drawingUtils2D.drawConfidenceMask(
              confidenceMask, defaultImage, overlayImage);

          const actualResult = context2D.getImageData(0, 0, WIDTH, HEIGHT).data;
          expect(actualResult)
              .toEqual(new Uint8ClampedArray(expectedResult.buffer));
          confidenceMask.close();
        });

        it('on WebGL canvas', () => {
          const confidenceMask = new MPMask(
              [new Float32Array(
                  [0.6, 1.0, 0.0, 0.3])],  // Note: Vertically flipped
              /* interpolateValues= */ true,
              /* ownsWebGLTexture= */ false, canvasWebGL, shaderContext, WIDTH,
              HEIGHT);

          drawingUtilsWebGL.drawConfidenceMask(
              confidenceMask, defaultImage, overlayImage);

          const actualResult = new Uint8Array(WIDTH * HEIGHT * 4);
          contextWebGL.readPixels(
              0, 0, WIDTH, HEIGHT, contextWebGL.RGBA,
              contextWebGL.UNSIGNED_BYTE, actualResult);
          expect(actualResult).toEqual(expectedResult);
          confidenceMask.close();
        });
      });

  describe('drawCategoryMask() ', () => {
    const colors = [
      [0, 0, 0, 255],
      [0, 255, 0, 255],
      [0, 0, 255, 255],
      [255, 255, 255, 255],
    ];
    const expectedResult = new Uint8Array(
        [0, 0, 0, 255, 0, 255, 0, 255, 0, 0, 255, 255, 255, 255, 255, 255],
    );

    it('on 2D canvas', () => {
      const categoryMask = new MPMask(
          [new Uint8Array([0, 1, 2, 3])],
          /* interpolateValues= */ false,
          /* ownsWebGLTexture= */ false, canvas2D, shaderContext, WIDTH,
          HEIGHT);

      drawingUtils2D.drawCategoryMask(categoryMask, colors);

      const actualResult = context2D.getImageData(0, 0, WIDTH, HEIGHT).data;
      expect(actualResult)
          .toEqual(new Uint8ClampedArray(expectedResult.buffer));
      categoryMask.close();
    });

    it('on WebGL canvas', () => {
      const categoryMask = new MPMask(
          [new Uint8Array([2, 3, 0, 1])],  // Note: Vertically flipped
          /* interpolateValues= */ false,
          /* ownsWebGLTexture= */ false, canvasWebGL, shaderContext, WIDTH,
          HEIGHT);

      drawingUtilsWebGL.drawCategoryMask(categoryMask, colors);

      const actualResult = new Uint8Array(WIDTH * WIDTH * 4);
      contextWebGL.readPixels(
          0, 0, WIDTH, HEIGHT, contextWebGL.RGBA, contextWebGL.UNSIGNED_BYTE,
          actualResult);
      expect(actualResult).toEqual(expectedResult);
      categoryMask.close();
    });
  });

  // TODO: Add tests for drawConnectors/drawLandmarks/drawBoundingBox
});
