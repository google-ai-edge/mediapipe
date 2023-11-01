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
  let canvas2D: HTMLCanvasElement;
  let context2D: CanvasRenderingContext2D;
  let drawingUtils2D: DrawingUtils;
  let canvasWebGL: HTMLCanvasElement;
  let contextWebGL: WebGL2RenderingContext;
  let drawingUtilsWebGL: DrawingUtils;

  beforeEach(() => {
    shaderContext = new MPImageShaderContext();

    canvasWebGL = document.createElement('canvas');
    canvasWebGL.width = WIDTH;
    canvasWebGL.height = HEIGHT;
    contextWebGL = canvasWebGL.getContext('webgl2')!;
    drawingUtilsWebGL = new DrawingUtils(contextWebGL);

    canvas2D = document.createElement('canvas');
    canvas2D.width = WIDTH;
    canvas2D.height = HEIGHT;
    context2D = canvas2D.getContext('2d')!;
    drawingUtils2D = new DrawingUtils(context2D, contextWebGL);
  });

  afterEach(() => {
    shaderContext.close();
    drawingUtils2D.close();
    drawingUtilsWebGL.close();
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
          /* ownsWebGLTexture= */ false, canvas2D, shaderContext, WIDTH,
          HEIGHT);

      drawingUtils2D.drawCategoryMask(categoryMask, colors);

      const actualResult = context2D.getImageData(0, 0, WIDTH, HEIGHT).data;
      expect(actualResult)
          .toEqual(new Uint8ClampedArray(expectedResult.buffer));
    });

    it('on WebGL canvas', () => {
      const categoryMask = new MPMask(
          [new Uint8Array([2, 3, 0, 1])],  // Note: Vertically flipped
          /* ownsWebGLTexture= */ false, canvasWebGL, shaderContext, WIDTH,
          HEIGHT);

      drawingUtilsWebGL.drawCategoryMask(categoryMask, colors);

      const actualResult = new Uint8Array(WIDTH * WIDTH * 4);
      contextWebGL.readPixels(
          0, 0, WIDTH, HEIGHT, contextWebGL.RGBA, contextWebGL.UNSIGNED_BYTE,
          actualResult);
      expect(actualResult).toEqual(expectedResult);
    });
  });

  // TODO: Add tests for drawConnectors/drawLandmarks/drawBoundingBox
});
