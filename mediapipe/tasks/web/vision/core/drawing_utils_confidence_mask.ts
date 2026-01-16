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
 * A fragment shader that blends a default image and overlay texture based on an
 * input texture that contains confidence values.
 */
const FRAGMENT_SHADER = `
  precision mediump float;
  uniform sampler2D maskTexture;
  uniform sampler2D defaultTexture;
  uniform sampler2D overlayTexture;
  varying vec2 vTex;
  void main() {
    float confidence = texture2D(maskTexture, vTex).r;
    vec4 defaultColor = texture2D(defaultTexture, vTex);
    vec4 overlayColor = texture2D(overlayTexture, vTex);
    // Apply the alpha from the overlay and merge in the default color
    overlayColor = mix(defaultColor, overlayColor, overlayColor.a);
    gl_FragColor = mix(defaultColor, overlayColor, confidence);
  }
 `;

/** A drawing util class for confidence masks. */
export class ConfidenceMaskShaderContext extends MPImageShaderContext {
  defaultTexture?: WebGLTexture;
  overlayTexture?: WebGLTexture;
  defaultTextureUniform?: WebGLUniformLocation;
  overlayTextureUniform?: WebGLUniformLocation;
  maskTextureUniform?: WebGLUniformLocation;

  protected override getFragmentShader(): string {
    return FRAGMENT_SHADER;
  }

  protected override setupTextures(): void {
    const gl = this.gl!;
    gl.activeTexture(gl.TEXTURE1);
    this.defaultTexture = this.createTexture(gl);
    gl.activeTexture(gl.TEXTURE2);
    this.overlayTexture = this.createTexture(gl);
  }

  protected override setupShaders(): void {
    super.setupShaders();
    const gl = this.gl!;
    this.defaultTextureUniform = assertExists(
        gl.getUniformLocation(this.program!, 'defaultTexture'),
        'Uniform location');
    this.overlayTextureUniform = assertExists(
        gl.getUniformLocation(this.program!, 'overlayTexture'),
        'Uniform location');
    this.maskTextureUniform = assertExists(
        gl.getUniformLocation(this.program!, 'maskTexture'),
        'Uniform location');
  }

  protected override configureUniforms(): void {
    super.configureUniforms();
    const gl = this.gl!;
    gl.uniform1i(this.maskTextureUniform!, 0);
    gl.uniform1i(this.defaultTextureUniform!, 1);
    gl.uniform1i(this.overlayTextureUniform!, 2);
  }

  bindAndUploadTextures(
      defaultImage: ImageSource, overlayImage: ImageSource,
      confidenceMask: WebGLTexture) {
    // TODO: We should avoid uploading textures from CPU to GPU
    // if the textures haven't changed. This can lead to drastic performance
    // slowdowns (~50ms per frame). Users can reduce the penalty by passing a
    // canvas object instead of ImageData/HTMLImageElement.
    const gl = this.gl!;
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, confidenceMask);

    gl.activeTexture(gl.TEXTURE1);
    gl.bindTexture(gl.TEXTURE_2D, this.defaultTexture!);
    gl.texImage2D(
        gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, defaultImage);

    gl.activeTexture(gl.TEXTURE2);
    gl.bindTexture(gl.TEXTURE_2D, this.overlayTexture!);
    gl.texImage2D(
        gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, overlayImage);
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

  override close(): void {
    if (this.defaultTexture) {
      this.gl!.deleteTexture(this.defaultTexture);
    }
    if (this.overlayTexture) {
      this.gl!.deleteTexture(this.overlayTexture);
    }
    super.close();
  }
}
