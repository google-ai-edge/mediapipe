/**
 * Copyright 2026 The MediaPipe Authors.
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

import {InferenceCalculatorOptions} from '../../../../calculators/tensor/inference_calculator_pb';
import {Acceleration as AccelerationProto} from '../../../../tasks/cc/core/proto/acceleration_pb';
import {BaseOptions as BaseOptionsProto} from '../../../../tasks/cc/core/proto/base_options_pb';
import {
  Stroke as StrokeProto,
  Strokes as StrokesProto,
} from '../../../../tasks/cc/vision/interactive_segmenter/proto/stroke_pb';
import {
  CachedGraphRunner,
  TaskRunner,
} from '../../../../tasks/web/core/task_runner';
import {WasmFileset} from '../../../../tasks/web/core/wasm_fileset';
import {MPImageShaderContext} from '../../../../tasks/web/vision/core/image_shader_context';
import {MPMask} from '../../../../tasks/web/vision/core/mask';
import {
  getImageSourceSize,
  WasmModule,
} from '../../../../web/graph_runner/graph_runner';
import {WasmImage} from '../../../../web/graph_runner/graph_runner_image_lib';
import {supportsOffscreenCanvas} from '../../../../web/graph_runner/platform_utils';
// Placeholder for internal dependency on trusted resource url

interface ImageDataLike {
  width: number;
  height: number;
  data: Uint8ClampedArray | Uint8Array;
}

function isImageDataLike(img: unknown): img is ImageDataLike {
  if (typeof img !== 'object' || img === null) return false;
  // Safe cast as we have verified that 'img' is an object and not null,
  // and we verify its structural properties below.
  const {data, width, height} = img as ImageDataLike;
  return (
    Number.isInteger(width) &&
    width > 0 &&
    Number.isInteger(height) &&
    height > 0 &&
    (data instanceof Uint8ClampedArray || data instanceof Uint8Array)
  );
}

import {
  BrushMode,
  InteractiveSegmenterOptions,
  Stroke,
} from './interactive_segmenter_options';
export type {
  InteractiveSegmenterOptions,
  Stroke,
} from './interactive_segmenter_options';
export {BrushMode};

const BRUSH_MODE_MAP: Record<BrushMode, 0 | 1 | 2 | 3> = {
  [BrushMode.UNSPECIFIED]: 0,
  [BrushMode.POSITIVE]: 1,
  [BrushMode.NEGATIVE]: 2,
  [BrushMode.LASSO]: 3,
};

// The OSS JS API does not support the builder pattern.
// tslint:disable:jspb-use-builder-pattern

/** Serializes the user strokes into a binary protocol buffer. */
function serializeStrokes(strokes: readonly Stroke[]): Uint8Array {
  const strokeList = strokes.map(({isCompleted, brushMode, point}) => {
    const brushModeProto = BRUSH_MODE_MAP[brushMode] ?? 0;

    const pointList = point.map(({x, y}) => {
      const p = new StrokeProto.Point();
      p.setX(x);
      p.setY(y);
      return p;
    });

    const stroke = new StrokeProto();
    stroke.setIsCompleted(isCompleted);
    stroke.setBrushMode(brushModeProto);
    stroke.setPointList(pointList);
    return stroke;
  });

  const strokesProto = new StrokesProto();
  strokesProto.setStrokeList(strokeList);
  return strokesProto.serializeBinary();
}

/**
 * The WasmModule interface for Interactive Segmenter, defining the native
 * methods exported by the C++ WebAssembly wrapper.
 */
// Exposes raw snake_case WebAssembly symbol exports that cannot use camelCase.
// tslint:disable:name-casing
/* eslint-disable @typescript-eslint/naming-convention */
export declare interface InteractiveSegmenterWasmModule extends WasmModule {
  /** Creates the native C++ InteractiveSegmenter engine instance. */
  _interactive_segmenter_create: (
    baseOptionsPtr: number,
    baseOptionsSize: number,
  ) => number;
  /** Sets the input image for segmentation, executing the encoder model. */
  _interactive_segmenter_set_image: (
    handle: number,
    pixelPtr: number,
    width: number,
    height: number,
    channels: number,
  ) => boolean;
  /** Executes the lightweight decoder model with user strokes. */
  _interactive_segmenter_segment: (
    handle: number,
    strokesPtr: number,
    strokesSize: number,
    outWidthPtr: number,
    outHeightPtr: number,
    outSizePtr: number,
  ) => number;
  /** Safely deletes the native C++ engine instance. */
  _interactive_segmenter_close: (handle: number) => void;
}
/* eslint-enable @typescript-eslint/naming-convention */
// tslint:enable:name-casing

/**
 * Calculates the number of channels from pixel length and dimensions,
 * performing safety checks.
 */
function calculateNumChannels({
  pixelsLength,
  width,
  height,
}: {
  pixelsLength: number;
  width: number;
  height: number;
}): number {
  if (width <= 0 || height <= 0) {
    throw new Error(
      `Invalid image dimensions: ${width}x${height}. ` +
        `Dimensions must be positive.`,
    );
  }

  if (pixelsLength % (width * height) !== 0) {
    throw new Error(
      `Invalid image dimensions or pixel data length. ` +
        `Pixel data length ${pixelsLength} is not a multiple of the number ` +
        `of pixels (${width * height}).`,
    );
  }

  const numChannels = pixelsLength / (width * height);
  if (numChannels !== 4 && numChannels !== 3 && numChannels !== 1) {
    throw new Error(
      `Invalid image dimensions or pixel data length. ` +
        `Calculated channels: ${numChannels}. Expected 1, 3, or 4.`,
    );
  }
  return numChannels;
}

/** Helper to create canvas. */
function createCanvas(): HTMLCanvasElement | OffscreenCanvas | undefined {
  return supportsOffscreenCanvas()
    ? undefined
    : document.createElement('canvas');
}

/**
 * Performs interactive segmentation on images using split mode architecture.
 *
 * Interactive Segmenter splits segmentation into two distinct steps:
 * 1. Set an image with `setImage()` (executes heavy feature extraction once).
 * 2. Segment with `segment()` one or more times efficiently providing strokes,
 *    allowing the user to fine tune the produced segmentation mask.
 */
export class InteractiveSegmenter extends TaskRunner {
  private readonly shaderContext = new MPImageShaderContext();
  private delegate = 'CPU';
  private nativeSegmenterHandle = 0;
  protected override baseOptions = new BaseOptionsProto();
  private currentImagePixelPtr = 0;
  private loggerTimestamp = 0;

  /**
   * Initializes the Wasm runtime and creates a new interactive segmenter
   * from the provided options.
   * @export
   * @param wasmFileset A configuration object that provides the location of
   *     the Wasm binary and its loader.
   * @param options The options for the Interactive Segmenter. Note that
   *     either a path to the model asset or a model buffer needs to be
   *     provided (via `baseOptions`).
   * @return A new `InteractiveSegmenter`.
   */
  static createFromOptions(
    wasmFileset: WasmFileset,
    options: InteractiveSegmenterOptions,
  ): Promise<InteractiveSegmenter> {
    const canvas = options.canvas ?? createCanvas();
    return TaskRunner.createInstance(
      InteractiveSegmenter,
      canvas,
      wasmFileset,
      options,
    );
  }

  /**
   * Initializes the Wasm runtime and creates a new interactive segmenter
   * based on the provided model asset buffer.
   * @export
   * @param wasmFileset A configuration object that provides the location of
   *     the Wasm binary and its loader.
   * @param modelAssetBuffer An array or a stream containing a binary
   *    representation of the model.
   * @return A new `InteractiveSegmenter`.
   */
  static createFromModelBuffer(
    wasmFileset: WasmFileset,
    modelAssetBuffer: Uint8Array | ReadableStreamDefaultReader,
  ): Promise<InteractiveSegmenter> {
    return TaskRunner.createInstance(
      InteractiveSegmenter,
      createCanvas(),
      wasmFileset,
      {baseOptions: {modelAssetBuffer}},
    );
  }

  /**
   * Initializes the Wasm runtime and creates a new interactive segmenter
   * based on the path to the model asset.
   * @export
   * @param wasmFileset A configuration object that provides the location of
   *     the Wasm binary and its loader.
   * @param modelAssetPath The path to the model asset.
   * @return A new `InteractiveSegmenter`.
   */
  static createFromModelPath(
    wasmFileset: WasmFileset,
    modelAssetPath: string,
  ): Promise<InteractiveSegmenter> {
    return TaskRunner.createInstance(
      InteractiveSegmenter,
      createCanvas(),
      wasmFileset,
      {baseOptions: {modelAssetPath}},
    );
  }

  /** @hideconstructor */
  constructor(
    wasmModule: WasmModule,
    glCanvas?: HTMLCanvasElement | OffscreenCanvas | null,
  ) {
    super(new CachedGraphRunner(wasmModule, glCanvas));
  }

  private get wasmModule(): InteractiveSegmenterWasmModule {
    // Safe cast because this task runner is initialized with InteractiveSegmenterWasmModule.
    return this.graphRunner
      .wasmModule as unknown as InteractiveSegmenterWasmModule;
  }

  /**
   * Sets new options for the interactive segmenter.
   *
   * Calling `setOptions()` with a subset of options only affects those
   * options.
   *
   * @export
   * @param options The options for the interactive segmenter.
   * @return A Promise that resolves when the settings have been applied.
   */
  override setOptions(options: InteractiveSegmenterOptions): Promise<void> {
    this.delegate = options.baseOptions?.delegate ?? 'CPU';
    return super.applyOptions(options);
  }

  /**
   * Sets the input image for segmentation, executing the encoder model once.
   * Extracts raw pixel bytes across diverse sources (Canvas, ImageData,
   * TexImageSource, DOM elements) and copies them across Wasm heap memory into the
   * native C++ engine.
   *
   * @export
   * @param image An image to process.
   */
  setImage(image: TexImageSource): void {
    if (this.nativeSegmenterHandle === 0) {
      throw new Error('Segmenter is not initialized.');
    }

    if (this.currentImagePixelPtr !== 0) {
      this.wasmModule._free(this.currentImagePixelPtr);
      this.currentImagePixelPtr = 0;
    }

    let width = 0;
    let height = 0;
    let pixels: Uint8ClampedArray | Uint8Array | undefined;

    if (
      (typeof ImageData !== 'undefined' && image instanceof ImageData) ||
      isImageDataLike(image)
    ) {
      width = image.width;
      height = image.height;
      pixels = image.data;
    } else {
      [width, height] = getImageSourceSize(image);
      let canvas: HTMLCanvasElement | OffscreenCanvas;
      if (typeof OffscreenCanvas !== 'undefined') {
        canvas = new OffscreenCanvas(width, height);
      } else if (typeof document !== 'undefined') {
        canvas = document.createElement('canvas');
      } else {
        throw new Error('Canvas is not supported in this environment.');
      }
      canvas.width = width;
      canvas.height = height;
      // Safe cast as we are using a canvas we just created or OffscreenCanvas.
      const ctx = canvas.getContext('2d') as
        | CanvasRenderingContext2D
        | OffscreenCanvasRenderingContext2D
        | null;
      if (!ctx) {
        throw new Error(
          'Canvas 2D context is not supported in this environment.',
        );
      }
      // Safe cast as we already verified that the image source is valid and
      // has dimensions.
      ctx.drawImage(image as CanvasImageSource, 0, 0);
      pixels = ctx.getImageData(0, 0, width, height).data;
    }

    if (!pixels) {
      throw new Error(
        'Unsupported image source or failed to extract image pixels.',
      );
    }

    const numChannels = calculateNumChannels({
      pixelsLength: pixels.length,
      width,
      height,
    });

    // Allocate a raw memory block on the WASM heap, copy raw pixel bytes into
    // it, and pass the pointer to the native direct C++ InteractiveSegmenter
    // engine.
    const pixelPtr = this.wasmModule._malloc(pixels.length);
    this.wasmModule.HEAPU8.set(pixels, pixelPtr);
    this.currentImagePixelPtr = pixelPtr;
    const success = this.wasmModule._interactive_segmenter_set_image(
      this.nativeSegmenterHandle,
      pixelPtr,
      width,
      height,
      numChannels,
    );

    if (!success) {
      throw new Error('Failed to set image on native engine.');
    }
  }

  /**
   * Performs segmentation using the provided strokes, executing the
   * lightweight decoder model. Serializes user strokes to protocol buffers,
   * invokes native inference, and creates a deep copy of the resulting Wasm
   * heap mask data to safeguard against native buffer deallocation.
   *
   * @export
   * @param strokes The sequence of user strokes.
   * @return The segmentation mask as an `MPMask`.
   */
  segment(strokes: readonly Stroke[]): MPMask {
    if (this.nativeSegmenterHandle === 0) {
      throw new Error('Segmenter is not initialized.');
    }

    const binaryProto = serializeStrokes(strokes);
    const strokesPtr = this.wasmModule._malloc(binaryProto.length);
    this.wasmModule.HEAPU8.set(binaryProto, strokesPtr);

    // Consolidate three 4-byte pointer allocations into a single contiguous
    // 12-byte heap block to eliminate allocation transaction overhead.
    const outPtr = this.wasmModule._malloc(12);
    const outWidthPtr = outPtr;
    const outHeightPtr = outPtr + 4;
    const outSizePtr = outPtr + 8;

    let binaryMaskPtr = 0;
    const timestamp = this.loggerTimestamp++;

    try {
      if (this.logger) {
        if (this.delegate === 'GPU') {
          this.logger.recordGpuInputArrival(timestamp);
        } else {
          this.logger.recordCpuInputArrival(timestamp);
        }
      }

      binaryMaskPtr = this.wasmModule._interactive_segmenter_segment(
        this.nativeSegmenterHandle,
        strokesPtr,
        binaryProto.length,
        outWidthPtr,
        outHeightPtr,
        outSizePtr,
      );

      if (binaryMaskPtr === 0) {
        throw new Error('Segmentation failed.');
      }

      this.logger?.recordInvocationEnd(timestamp);

      const width = this.wasmModule.HEAPU32[outWidthPtr / 4];
      const height = this.wasmModule.HEAPU32[outHeightPtr / 4];
      const size = this.wasmModule.HEAPU32[outSizePtr / 4];

      const floatArray = new Float32Array(
        this.wasmModule.HEAPU8.buffer,
        binaryMaskPtr,
        size / 4,
      );
      const maskData = new Float32Array(floatArray);

      const wasmImage: WasmImage = {data: maskData, width, height};
      return this.convertToMPMask(wasmImage, {
        interpolateValues: true,
        shouldCopyData: false,
      });
    } finally {
      // Guarantee clean up of all WASM heap allocations in the finally block,
      // completely eliminating the risk of silent memory leak OOMs.
      if (strokesPtr !== 0) {
        this.wasmModule._free(strokesPtr);
      }
      if (outPtr !== 0) {
        this.wasmModule._free(outPtr);
      }
      if (binaryMaskPtr !== 0) {
        this.wasmModule._free(binaryMaskPtr);
      }
    }
  }

  /**
   * Configures the native segmenter engine, selecting either CPU (XNNPACK)
   * or GPU (WebGL2) acceleration delegates.
   */
  private configureRunner(): void {
    if (this.nativeSegmenterHandle !== 0) {
      this.logger?.logSessionEnd();
      this.wasmModule._interactive_segmenter_close(this.nativeSegmenterHandle);
      this.nativeSegmenterHandle = 0;
    }
    // Clean up local image copy if the runner is refreshed,
    // as the native state is lost.
    if (this.currentImagePixelPtr !== 0) {
      this.wasmModule._free(this.currentImagePixelPtr);
      this.currentImagePixelPtr = 0;
    }

    const acceleration = new AccelerationProto();
    if (this.delegate === 'GPU') {
      acceleration.setGpu(new InferenceCalculatorOptions.Delegate.Gpu());
    } else {
      const xnnpack = new InferenceCalculatorOptions.Delegate.Xnnpack();
      xnnpack.setNumThreads(4);
      acceleration.setXnnpack(xnnpack);
    }
    this.baseOptions.setAcceleration(acceleration);

    const baseOptionsBytes = this.baseOptions.serializeBinary();
    const baseOptionsPtr = this.wasmModule._malloc(baseOptionsBytes.length);
    this.wasmModule.HEAPU8.set(baseOptionsBytes, baseOptionsPtr);
    this.nativeSegmenterHandle = this.wasmModule._interactive_segmenter_create(
      baseOptionsPtr,
      baseOptionsBytes.length,
    );
    this.wasmModule._free(baseOptionsPtr);
    if (this.nativeSegmenterHandle === 0) {
      throw new Error('Failed to create native InteractiveSegmenter engine.');
    }
    this.logger?.logSessionStart();
  }

  /**
   * Overrides the base class `refreshGraph` lifecycle hook. Due to the
   * stateful split-runner architecture, we route this hook directly
   * to `configureRunner()` to instantiate the native C++ engine.
   */
  protected override refreshGraph(): void {
    this.configureRunner();
  }

  /**
   * Converts a WebAssembly image (WasmImage) into a Multi-Platform Mask
   * (MPMask) representation.
   */
  protected convertToMPMask(
    wasmImage: WasmImage,
    options: {interpolateValues: boolean; shouldCopyData: boolean},
  ): MPMask {
    const {data, width, height} = wasmImage;
    const pixels = width * height;

    if (
      (data instanceof Uint8Array || data instanceof Float32Array) &&
      data.length !== pixels
    ) {
      throw new Error(`Unsupported channel count: ${data.length / pixels}`);
    }
    const container = data;

    const mask = new MPMask(
      [container],
      options.interpolateValues,
      /* ownsWebGLTexture= */ false,
      this.graphRunner.wasmModule.canvas ?? undefined,
      this.shaderContext,
      width,
      height,
    );
    return options.shouldCopyData ? mask.clone() : mask;
  }

  /**
   * Closes and cleans up the resources held by this task.
   * @export
   */
  override close(): void {
    if (this.nativeSegmenterHandle !== 0) {
      this.wasmModule._interactive_segmenter_close(this.nativeSegmenterHandle);
      this.nativeSegmenterHandle = 0;
    }
    if (this.currentImagePixelPtr !== 0) {
      this.wasmModule._free(this.currentImagePixelPtr);
      this.currentImagePixelPtr = 0;
    }
    this.shaderContext.close();
    super.close();
  }
}


