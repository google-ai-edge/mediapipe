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
import {ImageSource} from '../../../../web/graph_runner/graph_runner';

/**
 * A fragment shader that maps categories to colors based on a background
 * texture, a mask texture and a 256x1 "color mapping texture" that contains one
 * color for each pixel.
 */
const FRAGMENT_SHADER = `
  precision mediump float;
  uniform sampler2D backgroundTexture;
  uniform sampler2D maskTexture;
  uniform sampler2D colorMappingTexture;
  varying vec2 vTex;
  void main() {
    vec4 backgroundColor = texture2D(backgroundTexture, vTex);
    float category = texture2D(maskTexture, vTex).r;
    vec4 categoryColor = texture2D(colorMappingTexture, vec2(category, 0.0));
    gl_FragColor = mix(backgroundColor, categoryColor, categoryColor.a);
  }
 `;

/**
 * A four channel color with values for red, green, blue and alpha
 * respectively.
 */
export type RGBAColor = [number, number, number, number]|number[];

/**
 * A category to color mapping that uses either a map or an array to assign
 * category indexes to RGBA colors.
 */
export type CategoryToColorMap = Map<number, RGBAColor>|RGBAColor[];


/** Checks CategoryToColorMap maps for deep equality. */
function isEqualColorMap(
    a: CategoryToColorMap, b: CategoryToColorMap): boolean {
  if (a !== b) {
    return false;
  }

  const aEntries = a.entries();
  const bEntries = b.entries();
  for (const [aKey, aValue] of aEntries) {
    const bNext = bEntries.next();
    if (bNext.done) {
      return false;
    }

    const [bKey, bValue] = bNext.value;
    if (aKey !== bKey) {
      return false;
    }

    if (aValue[0] !== bValue[0] || aValue[1] !== bValue[1] ||
        aValue[2] !== bValue[2] || aValue[3] !== bValue[3]) {
      return false;
    }
  }
  return !!bEntries.next().done;
}


/** A drawing util class for category masks. */
export class CategoryMaskShaderContext extends MPImageShaderContext {
  backgroundTexture?: WebGLTexture;
  colorMappingTexture?: WebGLTexture;
  colorMappingTextureUniform?: WebGLUniformLocation;
  backgroundTextureUniform?: WebGLUniformLocation;
  maskTextureUniform?: WebGLUniformLocation;
  currentColorMap?: CategoryToColorMap;

  bindAndUploadTextures(
      categoryMask: WebGLTexture, background: ImageSource,
      colorMap: Map<number, number[]>|number[][]) {
    const gl = this.gl!;

    // Bind category mask
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, categoryMask);

    // TODO: We should avoid uploading textures from CPU to GPU
    // if the textures haven't changed. This can lead to drastic performance
    // slowdowns (~50ms per frame). Users can reduce the penalty by passing a
    // canvas object instead of ImageData/HTMLImageElement.
    gl.activeTexture(gl.TEXTURE1);
    gl.bindTexture(gl.TEXTURE_2D, this.backgroundTexture!);
    gl.texImage2D(
        gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, background);

    // Bind color mapping texture if changed.
    if (!this.currentColorMap ||
        !isEqualColorMap(this.currentColorMap, colorMap)) {
      this.currentColorMap = colorMap;

      const pixels = new Array(256 * 4).fill(0);
      colorMap.forEach((rgba, index) => {
        if (rgba.length !== 4) {
          throw new Error(
              `Color at index ${index} is not a four-channel value.`);
        }
        pixels[index * 4] = rgba[0];
        pixels[index * 4 + 1] = rgba[1];
        pixels[index * 4 + 2] = rgba[2];
        pixels[index * 4 + 3] = rgba[3];
      });
      gl.activeTexture(gl.TEXTURE2);
      gl.bindTexture(gl.TEXTURE_2D, this.colorMappingTexture!);
      gl.texImage2D(
          gl.TEXTURE_2D, 0, gl.RGBA, 256, 1, 0, gl.RGBA, gl.UNSIGNED_BYTE,
          new Uint8Array(pixels));
    } else {
      gl.activeTexture(gl.TEXTURE2);
      gl.bindTexture(gl.TEXTURE_2D, this.colorMappingTexture!);
    }
  }

  unbindTextures() {
    const gl = this.gl!;
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, null);
    gl.activeTexture(gl.TEXTURE1);
    gl.bindTexture(gl.TEXTURE_2D, null);
    gl.activeTexture(gl.TEXTURE2);
    gl.bindTexture(gl.TEXTURE_2D, null);
  }

  protected override getFragmentShader(): string {
    return FRAGMENT_SHADER;
  }

  protected override setupTextures(): void {
    const gl = this.gl!;
    gl.activeTexture(gl.TEXTURE1);
    this.backgroundTexture = this.createTexture(gl, gl.LINEAR);
    // Use `gl.NEAREST` to prevent interpolating values in our category to
    // color map.
    gl.activeTexture(gl.TEXTURE2);
    this.colorMappingTexture = this.createTexture(gl, gl.NEAREST);
  }

  protected override setupShaders(): void {
    super.setupShaders();
    const gl = this.gl!;
    this.backgroundTextureUniform = assertExists(
        gl.getUniformLocation(this.program!, 'backgroundTexture'),
        'Uniform location');
    this.colorMappingTextureUniform = assertExists(
        gl.getUniformLocation(this.program!, 'colorMappingTexture'),
        'Uniform location');
    this.maskTextureUniform = assertExists(
        gl.getUniformLocation(this.program!, 'maskTexture'),
        'Uniform location');
  }

  protected override configureUniforms(): void {
    super.configureUniforms();
    const gl = this.gl!;
    gl.uniform1i(this.maskTextureUniform!, 0);
    gl.uniform1i(this.backgroundTextureUniform!, 1);
    gl.uniform1i(this.colorMappingTextureUniform!, 2);
  }

  override close(): void {
    if (this.backgroundTexture) {
      this.gl!.deleteTexture(this.backgroundTexture);
    }
    if (this.colorMappingTexture) {
      this.gl!.deleteTexture(this.colorMappingTexture);
    }
    super.close();
  }
}
