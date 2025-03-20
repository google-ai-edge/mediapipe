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

import {NormalizedRect} from '../../../../framework/formats/rect_pb';
import {TaskRunner} from '../../../../tasks/web/core/task_runner';
import {WasmFileset} from '../../../../tasks/web/core/wasm_fileset';
import {MPImage} from '../../../../tasks/web/vision/core/image';
import {ImageProcessingOptions} from '../../../../tasks/web/vision/core/image_processing_options';
import {MPImageShaderContext} from '../../../../tasks/web/vision/core/image_shader_context';
import {MPMask} from '../../../../tasks/web/vision/core/mask';
import {getImageSourceSize, GraphRunner, ImageSource, WasmMediaPipeConstructor} from '../../../../web/graph_runner/graph_runner';
import {SupportImage, WasmImage} from '../../../../web/graph_runner/graph_runner_image_lib';
import {supportsOffscreenCanvas} from '../../../../web/graph_runner/platform_utils';
import {SupportModelResourcesGraphService} from '../../../../web/graph_runner/register_model_resources_graph_service';

import {VisionTaskOptions} from './vision_task_options';

// tslint:disable-next-line:enforce-name-casing
const GraphRunnerVisionType =
    SupportModelResourcesGraphService(SupportImage(GraphRunner));
/** An implementation of the GraphRunner that supports image operations */
export class VisionGraphRunner extends GraphRunnerVisionType {}

// The OSS JS API does not support the builder pattern.
// tslint:disable:jspb-use-builder-pattern


/**
 * Creates a canvas for a MediaPipe vision task. Returns `undefined` if the
 * GraphRunner should create its own canvas.
 */
function createCanvas(): HTMLCanvasElement|OffscreenCanvas|undefined {
  // Returns an HTML canvas or `undefined` if OffscreenCanvas is fully supported
  // (since the graph runner can initialize its own OffscreenCanvas).
  return supportsOffscreenCanvas() ? undefined :
                                     document.createElement('canvas');
}

/** Base class for all MediaPipe Vision Tasks. */
export abstract class VisionTaskRunner extends TaskRunner {
  private readonly shaderContext = new MPImageShaderContext();

  protected static async createVisionInstance<T extends VisionTaskRunner>(
      type: WasmMediaPipeConstructor<T>, fileset: WasmFileset,
      options: VisionTaskOptions): Promise<T> {
    const canvas = options.canvas ?? createCanvas();
    return TaskRunner.createInstance(type, canvas, fileset, options);
  }

  /**
   * Constructor to initialize a `VisionTaskRunner`.
   *
   * @param graphRunner the graph runner for this task.
   * @param imageStreamName the name of the input image stream.
   * @param normRectStreamName the name of the input normalized rect image
   *     stream used to provide (mandatory) rotation and (optional)
   *     region-of-interest. `null` if the graph does not support normalized
   *     rects.
   * @param roiAllowed Whether this task supports Region-Of-Interest
   *     pre-processing
   *
   * @hideconstructor protected
   */
  constructor(
      protected override readonly graphRunner: VisionGraphRunner,
      private readonly imageStreamName: string,
      private readonly normRectStreamName: string|null,
      private readonly roiAllowed: boolean) {
    super(graphRunner);
  }

  /**
   * Configures the shared options of a vision task.
   *
   * @param options The options for the task.
   * @param loadTfliteModel Whether to load the model specified in
   *     `options.baseOptions`.
   */
  protected override applyOptions(
      options: VisionTaskOptions, loadTfliteModel = true): Promise<void> {
    if ('runningMode' in options) {
      const useStreamMode =
          !!options.runningMode && options.runningMode !== 'IMAGE';
      this.baseOptions.setUseStreamMode(useStreamMode);
    }

    if (options.canvas !== undefined) {
      if (this.graphRunner.wasmModule.canvas !== options.canvas) {
        throw new Error('You must create a new task to reset the canvas.');
      }
    }

    return super.applyOptions(options, loadTfliteModel);
  }

  /** Sends a single image to the graph and awaits results. */
  protected processImageData(
      image: ImageSource,
      imageProcessingOptions: ImageProcessingOptions|undefined): void {
    if (!!this.baseOptions?.getUseStreamMode()) {
      throw new Error(
          'Task is not initialized with image mode. ' +
          '\'runningMode\' must be set to \'IMAGE\'.');
    }
    this.process(image, imageProcessingOptions, this.getSynctheticTimestamp());
  }

  /** Sends a single video frame to the graph and awaits results. */
  protected processVideoData(
      imageFrame: ImageSource,
      imageProcessingOptions: ImageProcessingOptions|undefined,
      timestamp: number): void {
    if (!this.baseOptions?.getUseStreamMode()) {
      throw new Error(
          'Task is not initialized with video mode. ' +
          '\'runningMode\' must be set to \'VIDEO\'.');
    }
    this.process(imageFrame, imageProcessingOptions, timestamp);
  }

  private convertToNormalizedRect(
      imageSource: ImageSource,
      imageProcessingOptions?: ImageProcessingOptions): NormalizedRect {
    const normalizedRect = new NormalizedRect();

    if (imageProcessingOptions?.regionOfInterest) {
      if (!this.roiAllowed) {
        throw new Error('This task doesn\'t support region-of-interest.');
      }

      const roi = imageProcessingOptions.regionOfInterest;

      if (roi.left >= roi.right || roi.top >= roi.bottom) {
        throw new Error('Expected RectF with left < right and top < bottom.');
      }
      if (roi.left < 0 || roi.top < 0 || roi.right > 1 || roi.bottom > 1) {
        throw new Error('Expected RectF values to be in [0,1].');
      }

      normalizedRect.setXCenter((roi.left + roi.right) / 2.0);
      normalizedRect.setYCenter((roi.top + roi.bottom) / 2.0);
      normalizedRect.setWidth(roi.right - roi.left);
      normalizedRect.setHeight(roi.bottom - roi.top);
    } else {
      normalizedRect.setXCenter(0.5);
      normalizedRect.setYCenter(0.5);
      normalizedRect.setWidth(1);
      normalizedRect.setHeight(1);
    }

    if (imageProcessingOptions?.rotationDegrees) {
      if (imageProcessingOptions?.rotationDegrees % 90 !== 0) {
        throw new Error(
            'Expected rotation to be a multiple of 90°.',
        );
      }

      // Convert to radians anti-clockwise.
      normalizedRect.setRotation(
          -Math.PI * imageProcessingOptions.rotationDegrees / 180.0);

      // For 90° and 270° rotations, we need to swap width and height.
      // This is due to the internal behavior of ImageToTensorCalculator, which:
      // - first denormalizes the provided rect by multiplying the rect width or
      //   height by the image width or height, respectively.
      // - then rotates this by denormalized rect by the provided rotation, and
      //   uses this for cropping,
      // - then finally rotates this back.
      if (imageProcessingOptions?.rotationDegrees % 180 !== 0) {
        const [imageWidth, imageHeight] = getImageSourceSize(imageSource);
        // tslint:disable:no-unnecessary-type-assertion
        const width = normalizedRect.getHeight()! * imageHeight / imageWidth;
        const height = normalizedRect.getWidth()! * imageWidth / imageHeight;
        // tslint:enable:no-unnecessary-type-assertion
        normalizedRect.setWidth(width);
        normalizedRect.setHeight(height);
      }
    }

    return normalizedRect;
  }

  /** Runs the graph and blocks on the response. */
  private process(
      imageSource: ImageSource,
      imageProcessingOptions: ImageProcessingOptions|undefined,
      timestamp: number): void {
    if (this.normRectStreamName) {
      const normalizedRect =
          this.convertToNormalizedRect(imageSource, imageProcessingOptions);
      this.graphRunner.addProtoToStream(
          normalizedRect.serializeBinary(), 'mediapipe.NormalizedRect',
          this.normRectStreamName, timestamp);
    }
    this.graphRunner.addGpuBufferAsImageToStream(
        imageSource, this.imageStreamName, timestamp ?? performance.now());
    this.finishProcessing();
  }

  /**
   * Converts a WasmImage to an MPImage.
   *
   * Converts the underlying Uint8Array-backed images to ImageData
   * (adding an alpha channel if necessary), passes through WebGLTextures and
   * throws for Float32Array-backed images.
   */
  protected convertToMPImage(wasmImage: WasmImage, shouldCopyData: boolean):
      MPImage {
    const {data, width, height} = wasmImage;
    const pixels = width * height;

    let container: ImageData|WebGLTexture;
    if (data instanceof Uint8Array) {
      if (data.length === pixels * 3) {
        // TODO: Convert in C++
        const rgba = new Uint8ClampedArray(pixels * 4);
        for (let i = 0; i < pixels; ++i) {
          rgba[4 * i] = data[3 * i];
          rgba[4 * i + 1] = data[3 * i + 1];
          rgba[4 * i + 2] = data[3 * i + 2];
          rgba[4 * i + 3] = 255;
        }
        container = new ImageData(rgba, width, height);
      } else if (data.length === pixels * 4) {
        container = new ImageData(
            new Uint8ClampedArray(data.buffer, data.byteOffset, data.length),
            width, height);
      } else {
        throw new Error(`Unsupported channel count: ${data.length/pixels}`);
      }
    } else if (data instanceof WebGLTexture) {
      container = data;
    } else {
      throw new Error(`Unsupported format: ${data.constructor.name}`);
    }

    const image = new MPImage(
        [container], /* ownsImageBitmap= */ false,
        /* ownsWebGLTexture= */ false, this.graphRunner.wasmModule.canvas!,
        this.shaderContext, width, height);
    return shouldCopyData ? image.clone() : image;
  }

  /** Converts a WasmImage to an MPMask.  */
  protected convertToMPMask(
      wasmImage: WasmImage, interpolateValues: boolean,
      shouldCopyData: boolean): MPMask {
    const {data, width, height} = wasmImage;
    const pixels = width * height;

    let container: WebGLTexture|Uint8Array|Float32Array;
    if (data instanceof Uint8Array || data instanceof Float32Array) {
      if (data.length === pixels) {
        container = data;
      } else {
        throw new Error(`Unsupported channel count: ${data.length / pixels}`);
      }
    } else {
      container = data;
    }

    const mask = new MPMask(
        [container], interpolateValues,
        /* ownsWebGLTexture= */ false, this.graphRunner.wasmModule.canvas!,
        this.shaderContext, width, height);
    return shouldCopyData ? mask.clone() : mask;
  }

  /**
   * Closes and cleans up the resources held by this task.
   * @export
   */
  override close(): void {
    this.shaderContext.close();
    super.close();
  }
}


