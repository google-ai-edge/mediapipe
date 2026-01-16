// Placeholder for internal dependency on assertTruthy
import {supportsOffscreenCanvas} from '../../web/graph_runner/platform_utils';
import {runScript} from '../../web/graph_runner/run_script_helper';
// Placeholder for internal dependency on trusted resource url

import {GraphRunnerApi, ImageSource} from './graph_runner_api';
import {CreateGraphRunnerApi, CreateMediaPipeLibApi, FileLocator, WasmMediaPipeConstructor} from './graph_runner_factory_api';
import {EmptyPacketListener, ErrorListener, SimpleListener, VectorListener} from './listener_types';
import {WasmModule} from './wasm_module';

// This file contains the internal implementations behind the public APIs
// declared in "graph_runner_api.d.ts" and "graph_runner_factory_api.d.ts".

// First we re-export all of our imported public types/defines, so that users
// can import everything they need from here directly.
export {
  type EmptyPacketListener,
  type ErrorListener,
  type FileLocator,
  type ImageSource,
  type SimpleListener,
  type VectorListener,
  type WasmMediaPipeConstructor,
  type WasmModule,
};

/**
 * A listener that receives the CalculatorGraphConfig in binary encoding.
 */
export type CalculatorGraphConfigListener = (graphConfig: Uint8Array) => void;

/**
 * The name of the internal listener that we use to obtain the calculator graph
 * config. Intended for internal usage. Exported for testing only.
 */
export const CALCULATOR_GRAPH_CONFIG_LISTENER_NAME = '__graph_config__';

// Global declarations, for tapping into Window for Wasm blob running
declare global {
  interface Window {
    // Created by us using wasm-runner script
    Module?: WasmModule|FileLocator;
    // Created by wasm-runner script
    ModuleFactory?: (fileLocator: FileLocator) => Promise<WasmModule>;
  }
}

/**
 * Fetches each URL in urls, executes them one-by-one in the order they are
 * passed, and then returns (or throws if something went amiss).
 */
declare function importScripts(...urls: Array<string|URL>): void;

/**
 * Detects image source size.
 */
export function getImageSourceSize(imageSource: TexImageSource):
    [number, number] {
  if ((imageSource as HTMLVideoElement).videoWidth !== undefined) {
    const videoElement = imageSource as HTMLVideoElement;
    return [videoElement.videoWidth, videoElement.videoHeight];
  } else if ((imageSource as HTMLImageElement).naturalWidth !== undefined) {
    // TODO: Ensure this works with SVG images
    const imageElement = imageSource as HTMLImageElement;
    return [imageElement.naturalWidth, imageElement.naturalHeight];
  } else if ((imageSource as VideoFrame).displayWidth !== undefined) {
    const videoFrame = imageSource as VideoFrame;
    return [videoFrame.displayWidth, videoFrame.displayHeight];
  } else {
    const notVideoFrame = imageSource as Exclude<TexImageSource, VideoFrame>;
    return [notVideoFrame.width, notVideoFrame.height];
  }
}

/**
 * Simple class to run an arbitrary image-in/image-out MediaPipe graph (i.e.
 * as created by wasm_mediapipe_demo BUILD macro), and either render results
 * into canvas, or else return the output WebGLTexture. Takes a WebAssembly
 * Module.
 */
export class GraphRunner implements GraphRunnerApi {
  // TODO: These should be protected/private, but are left exposed for
  //   now so that we can use proper TS mixins with this class as a base. This
  //   should be somewhat fixed when we create our .d.ts files.
  readonly wasmModule: WasmModule;
  readonly hasMultiStreamSupport: boolean;
  autoResizeCanvas = true;
  audioPtr: number|null;
  audioSize: number;

  /**
   * Creates a new MediaPipe WASM module. Must be called *after* wasm Module has
   * initialized. Note that we take control of the GL canvas from here on out,
   * and will resize it to fit input.
   *
   * @param module The underlying Wasm Module to use.
   * @param glCanvas The type of the GL canvas to use, or `null` if no GL
   *    canvas should be initialzed. Initializes an offscreen canvas if not
   *    provided.
   */
  constructor(
      module: WasmModule, glCanvas?: HTMLCanvasElement|OffscreenCanvas|null) {
    this.wasmModule = module;
    this.audioPtr = null;
    this.audioSize = 0;
    this.hasMultiStreamSupport =
        (typeof this.wasmModule._addIntToInputStream === 'function');

    if (glCanvas !== undefined) {
      this.wasmModule.canvas = glCanvas;
    } else if (supportsOffscreenCanvas()) {
      // If no canvas is provided, assume Chrome/Firefox and just make an
      // OffscreenCanvas for GPU processing. Note that we exclude older Safari
      // versions that not support WebGL for OffscreenCanvas.
      this.wasmModule.canvas = new OffscreenCanvas(1, 1);
    } else {
      console.warn(
          'OffscreenCanvas not supported and GraphRunner constructor ' +
          'glCanvas parameter is undefined. Creating backup canvas.');
      this.wasmModule.canvas = document.createElement('canvas');
    }
  }

  /** {@override GraphRunnerApi} */
  async initializeGraph(graphFile: string): Promise<void> {
    // Fetch and set graph
    const response = await fetch(graphFile);
    const graphData = await response.arrayBuffer();
    const isBinary =
        !(graphFile.endsWith('.pbtxt') || graphFile.endsWith('.textproto'));
    this.setGraph(new Uint8Array(graphData), isBinary);
  }

  /** {@override GraphRunnerApi} */
  setGraphFromString(graphConfig: string): void {
    this.setGraph((new TextEncoder()).encode(graphConfig), false);
  }

  /** {@override GraphRunnerApi} */
  setGraph(graphData: Uint8Array, isBinary: boolean): void {
    const size = graphData.length;
    const dataPtr = this.wasmModule._malloc(size);
    this.wasmModule.HEAPU8.set(graphData, dataPtr);
    if (isBinary) {
      this.wasmModule._changeBinaryGraph(size, dataPtr);
    } else {
      this.wasmModule._changeTextGraph(size, dataPtr);
    }
    this.wasmModule._free(dataPtr);
  }

  /** {@override GraphRunnerApi} */
  configureAudio(numChannels: number, numSamples: number | null, sampleRate: number,
      streamName?: string, headerName?: string) {
    if (!this.wasmModule._configureAudio) {
      console.warn(
          'Attempting to use configureAudio without support for input audio. ' +
          'Is build dep ":gl_graph_runner_audio" missing?');
    }
    streamName = streamName || 'input_audio';
    this.wrapStringPtr(streamName, (streamNamePtr: number) => {
      headerName = headerName || 'audio_header';
      this.wrapStringPtr(headerName, (headerNamePtr: number) => {
        this.wasmModule._configureAudio(streamNamePtr, headerNamePtr,
          numChannels, numSamples ?? 0, sampleRate);
      });
    });
  }

  /** {@override GraphRunnerApi} */
  setAutoResizeCanvas(resize: boolean): void {
    this.autoResizeCanvas = resize;
  }

  /** {@override GraphRunnerApi} */
  setAutoRenderToScreen(enabled: boolean): void {
    this.wasmModule._setAutoRenderToScreen(enabled);
  }

  /** {@override GraphRunnerApi} */
  setGpuBufferVerticalFlip(bottomLeftIsOrigin: boolean): void {
    this.wasmModule.gpuOriginForWebTexturesIsBottomLeft = bottomLeftIsOrigin;
  }

  /**
   * Bind texture to our internal canvas, and upload image source to GPU.
   * Returns tuple [width, height] of texture.  Intended for internal usage.
   */
  bindTextureToStream(imageSource: TexImageSource, streamNamePtr?: number):
      [number, number] {
    if (!this.wasmModule.canvas) {
      throw new Error('No OpenGL canvas configured.');
    }

    if (!streamNamePtr) {
      // TODO: Remove this path once completely refactored away.
      console.assert(this.wasmModule._bindTextureToCanvas());
    } else {
      this.wasmModule._bindTextureToStream(streamNamePtr);
    }
    const gl =
        (this.wasmModule.canvas.getContext('webgl2') ||
         this.wasmModule.canvas.getContext('webgl')) as WebGL2RenderingContext |
        WebGLRenderingContext | null;
    if (!gl) {
      throw new Error(
          'Failed to obtain WebGL context from the provided canvas. ' +
          '`getContext()` should only be invoked with `webgl` or `webgl2`.');
    }
    if (this.wasmModule.gpuOriginForWebTexturesIsBottomLeft) {
      gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, true);
    }
    gl.texImage2D(
        gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, imageSource);
    if (this.wasmModule.gpuOriginForWebTexturesIsBottomLeft) {
      gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, false);
    }

    const [width, height] = getImageSourceSize(imageSource);

    if (this.autoResizeCanvas &&
        (width !== this.wasmModule.canvas.width ||
         height !== this.wasmModule.canvas.height)) {
      this.wasmModule.canvas.width = width;
      this.wasmModule.canvas.height = height;
    }

    return [width, height];
  }

  /**
   * Takes the raw data from a JS image source, and sends it to C++ to be
   * processed, waiting synchronously for the response. Note that we will resize
   * our GL canvas to fit the input, so input size should only change
   * infrequently. NOTE: This call has been deprecated in favor of
   * `addGpuBufferToStream`.
   * @param imageSource An image source to process.
   * @param timestamp The timestamp of the current frame, in ms.
   * @return texture? The WebGL texture reference, if one was produced.
   */
  processGl(imageSource: TexImageSource, timestamp: number): WebGLTexture
      |undefined {
    // Bind to default input stream
    const [width, height] = this.bindTextureToStream(imageSource);

    // 2 ints and a ll (timestamp)
    const frameDataPtr = this.wasmModule._malloc(16);
    this.wasmModule.HEAPU32[frameDataPtr / 4] = width;
    this.wasmModule.HEAPU32[(frameDataPtr / 4) + 1] = height;
    this.wasmModule.HEAPF64[(frameDataPtr / 8) + 1] = timestamp;
    // outputPtr points in HEAPF32-space to running mspf calculations, which we
    // don't use at the moment.
    // tslint:disable-next-line:no-unused-variable
    const outputPtr = this.wasmModule._processGl(frameDataPtr) / 4;
    this.wasmModule._free(frameDataPtr);

    // TODO: Hook up WebGLTexture output, when given.
    // TODO: Allow user to toggle whether or not to render output into canvas.
    return undefined;
  }

  /**
   * Converts JavaScript string input parameters into C++ c-string pointers.
   * See b/204830158 for more details. Intended for internal usage.
   */
  wrapStringPtr(stringData: string, stringPtrFunc: (ptr: number) => void):
      void {
    if (!this.hasMultiStreamSupport) {
      console.error(
          'No wasm multistream support detected: ensure dependency ' +
          'inclusion of :gl_graph_runner_internal_multi_input target');
    }
    const stringDataPtr = this.wasmModule.stringToNewUTF8(stringData);
    stringPtrFunc(stringDataPtr);
    this.wasmModule._free(stringDataPtr);
  }

  /**
   * Converts JavaScript string input parameters into C++ c-string pointers.
   * See b/204830158 for more details. Intended for internal usage.
   */
  async wrapStringPtrAsync(stringData: string,
                           stringPtrFunc: (ptr: number) => Promise<void>):
      Promise<void> {
    if (!this.hasMultiStreamSupport) {
      console.error(
          'No wasm multistream support detected: ensure dependency ' +
          'inclusion of :gl_graph_runner_internal_multi_input target');
    }
    const stringDataPtr = this.wasmModule.stringToNewUTF8(stringData);
    await stringPtrFunc(stringDataPtr);
    this.wasmModule._free(stringDataPtr);
  }

  /**
   * Converts JavaScript string input parameters into C++ c-string pointers.
   * See b/204830158 for more details.
   */
  wrapStringPtrPtr(stringData: string[], ptrFunc: (ptr: number) => void): void {
    if (!this.hasMultiStreamSupport) {
      console.error(
          'No wasm multistream support detected: ensure dependency ' +
          'inclusion of :gl_graph_runner_internal_multi_input target');
    }
    const uint32Array = new Uint32Array(stringData.length);
    for (let i = 0; i < stringData.length; i++) {
      uint32Array[i] = this.wasmModule.stringToNewUTF8(stringData[i]);
    }
    const heapSpace = this.wasmModule._malloc(uint32Array.length * 4);
    this.wasmModule.HEAPU32.set(uint32Array, heapSpace >> 2);

    ptrFunc(heapSpace);
    for (const uint32ptr of uint32Array) {
      this.wasmModule._free(uint32ptr);
    }
    this.wasmModule._free(heapSpace);
  }

  /**
   * Invokes the callback with the current calculator configuration (in binary
   * format).
   *
   * Consumers must deserialize the binary representation themselves as this
   * avoids adding a direct dependency on the Protobuf JSPB target in the graph
   * library.
   */
  getCalculatorGraphConfig(
      callback: CalculatorGraphConfigListener, makeDeepCopy?: boolean): void {
    const listener = CALCULATOR_GRAPH_CONFIG_LISTENER_NAME;

    // Create a short-lived listener to receive the binary encoded proto
    this.setListener(listener, (data: Uint8Array) => {
      callback(data);
    });
    this.wrapStringPtr(listener, (outputStreamNamePtr: number) => {
      this.wasmModule._getGraphConfig(outputStreamNamePtr, makeDeepCopy);
    });

    delete this.wasmModule.simpleListeners![listener];
  }

  /**
   * Ensures existence of the simple listeners table and registers the callback.
   * Intended for internal usage.
   */
  setListener<T>(outputStreamName: string, callbackFcn: SimpleListener<T>) {
    this.wasmModule.simpleListeners = this.wasmModule.simpleListeners || {};
    this.wasmModule.simpleListeners[outputStreamName] =
        callbackFcn as SimpleListener<unknown>;
  }

  /**
   * Ensures existence of the vector listeners table and registers the callback.
   * Intended for internal usage.
   */
  setVectorListener<T>(
      outputStreamName: string, callbackFcn: SimpleListener<T[]>) {
    let buffer: T[] = [];
    this.wasmModule.simpleListeners = this.wasmModule.simpleListeners || {};
    this.wasmModule.simpleListeners[outputStreamName] =
        (data: unknown, done: boolean, timestamp: number) => {
          if (done) {
            callbackFcn(buffer, timestamp);
            buffer = [];
          } else {
            buffer.push(data as T);
          }
        };
  }

  /** {@override GraphRunnerApi} */
  attachErrorListener(callbackFcn: (code: number, message: string) => void) {
    this.wasmModule.errorListener = callbackFcn;
  }

  /** {@override GraphRunnerApi} */
  attachEmptyPacketListener(
      outputStreamName: string, callbackFcn: EmptyPacketListener) {
    this.wasmModule.emptyPacketListeners =
        this.wasmModule.emptyPacketListeners || {};
    this.wasmModule.emptyPacketListeners[outputStreamName] = callbackFcn;
  }

  /** {@override GraphRunnerApi} */
  addAudioToStream(
      audioData: Float32Array, streamName: string, timestamp: number) {
    // numChannels and numSamples being 0 will cause defaults to be used,
    // which will reflect values from last call to configureAudio.
    this.addAudioToStreamWithShape(audioData, 0, 0, streamName, timestamp);
  }

  /** {@override GraphRunnerApi} */
  addAudioToStreamWithShape(
      audioData: Float32Array, numChannels: number, numSamples: number,
      streamName: string, timestamp: number) {
    // 4 bytes for each F32
    const size = audioData.length * 4;
    if (this.audioSize !== size) {
      if (this.audioPtr) {
        this.wasmModule._free(this.audioPtr);
      }
      this.audioPtr = this.wasmModule._malloc(size);
      this.audioSize = size;
    }
    this.wasmModule.HEAPF32.set(audioData, this.audioPtr! / 4);

    this.wrapStringPtr(streamName, (streamNamePtr: number) => {
      this.wasmModule._addAudioToInputStream(
          this.audioPtr!, numChannels, numSamples, streamNamePtr, timestamp);
    });
  }

  /** {@override GraphRunnerApi} */
  addGpuBufferToStream(
      imageSource: TexImageSource, streamName: string,
      timestamp: number): void {
    this.wrapStringPtr(streamName, (streamNamePtr: number) => {
      const [width, height] =
          this.bindTextureToStream(imageSource, streamNamePtr);
      this.wasmModule._addBoundTextureToStream(
          streamNamePtr, width, height, timestamp);
    });
  }

  /** {@override GraphRunnerApi} */
  addBoolToStream(data: boolean, streamName: string, timestamp: number): void {
    this.wrapStringPtr(streamName, (streamNamePtr: number) => {
      this.wasmModule._addBoolToInputStream(data, streamNamePtr, timestamp);
    });
  }

  /** {@override GraphRunnerApi} */
  addDoubleToStream(data: number, streamName: string, timestamp: number): void {
    this.wrapStringPtr(streamName, (streamNamePtr: number) => {
      this.wasmModule._addDoubleToInputStream(data, streamNamePtr, timestamp);
    });
  }

  /** {@override GraphRunnerApi} */
  addFloatToStream(data: number, streamName: string, timestamp: number): void {
    this.wrapStringPtr(streamName, (streamNamePtr: number) => {
      // NOTE: _addFloatToStream and _addIntToStream are reserved for JS
      // Calculators currently; we may want to revisit this naming scheme in the
      // future.
      this.wasmModule._addFloatToInputStream(data, streamNamePtr, timestamp);
    });
  }

  /** {@override GraphRunnerApi} */
  addIntToStream(data: number, streamName: string, timestamp: number): void {
    this.wrapStringPtr(streamName, (streamNamePtr: number) => {
      this.wasmModule._addIntToInputStream(data, streamNamePtr, timestamp);
    });
  }

  /** {@override GraphRunnerApi} */
  addUintToStream(data: number, streamName: string, timestamp: number): void {
    this.wrapStringPtr(streamName, (streamNamePtr: number) => {
      this.wasmModule._addUintToInputStream(
          data, streamNamePtr, timestamp);
    });
  }

  /** {@override GraphRunnerApi} */
  addStringToStream(data: string, streamName: string, timestamp: number): void {
    this.wrapStringPtr(streamName, (streamNamePtr: number) => {
      this.wrapStringPtr(data, (dataPtr: number) => {
        this.wasmModule._addStringToInputStream(
            dataPtr, streamNamePtr, timestamp);
      });
    });
  }

  /** {@override GraphRunnerApi} */
  addStringRecordToStream(
      data: Record<string, string>, streamName: string,
      timestamp: number): void {
    this.wrapStringPtr(streamName, (streamNamePtr: number) => {
      this.wrapStringPtrPtr(Object.keys(data), (keyList: number) => {
        this.wrapStringPtrPtr(Object.values(data), (valueList: number) => {
          this.wasmModule._addFlatHashMapToInputStream(
              keyList, valueList, Object.keys(data).length, streamNamePtr,
              timestamp);
        });
      });
    });
  }

  /** {@override GraphRunnerApi} */
  addProtoToStream(
      data: Uint8Array, protoType: string, streamName: string,
      timestamp: number): void {
    this.wrapStringPtr(streamName, (streamNamePtr: number) => {
      this.wrapStringPtr(protoType, (protoTypePtr: number) => {
        // Deep-copy proto data into Wasm heap
        const dataPtr = this.wasmModule._malloc(data.length);
        // TODO: Ensure this is the fastest way to copy this data.
        this.wasmModule.HEAPU8.set(data, dataPtr);
        this.wasmModule._addProtoToInputStream(
            dataPtr, data.length, protoTypePtr, streamNamePtr, timestamp);
        this.wasmModule._free(dataPtr);
      });
    });
  }

  /** {@override GraphRunnerApi} */
  addEmptyPacketToStream(streamName: string, timestamp: number): void {
    this.wrapStringPtr(streamName, (streamNamePtr: number) => {
      this.wasmModule._addEmptyPacketToInputStream(streamNamePtr, timestamp);
    });
  }

  /** {@override GraphRunnerApi} */
  addBoolVectorToStream(data: boolean[], streamName: string, timestamp: number):
      void {
    this.wrapStringPtr(streamName, (streamNamePtr: number) => {
      const vecPtr = this.wasmModule._allocateBoolVector(data.length);
      if (!vecPtr) {
        throw new Error('Unable to allocate new bool vector on heap.');
      }
      for (const entry of data) {
        this.wasmModule._addBoolVectorEntry(vecPtr, entry);
      }
      this.wasmModule._addBoolVectorToInputStream(
          vecPtr, streamNamePtr, timestamp);
    });
  }

  /** {@override GraphRunnerApi} */
  addDoubleVectorToStream(
      data: number[], streamName: string, timestamp: number): void {
    this.wrapStringPtr(streamName, (streamNamePtr: number) => {
      const vecPtr = this.wasmModule._allocateDoubleVector(data.length);
      if (!vecPtr) {
        throw new Error('Unable to allocate new double vector on heap.');
      }
      for (const entry of data) {
        this.wasmModule._addDoubleVectorEntry(vecPtr, entry);
      }
      this.wasmModule._addDoubleVectorToInputStream(
          vecPtr, streamNamePtr, timestamp);
    });
  }

  /** {@override GraphRunnerApi} */
  addFloatVectorToStream(data: number[], streamName: string, timestamp: number):
      void {
    this.wrapStringPtr(streamName, (streamNamePtr: number) => {
      const vecPtr = this.wasmModule._allocateFloatVector(data.length);
      if (!vecPtr) {
        throw new Error('Unable to allocate new float vector on heap.');
      }
      for (const entry of data) {
        this.wasmModule._addFloatVectorEntry(vecPtr, entry);
      }
      this.wasmModule._addFloatVectorToInputStream(
          vecPtr, streamNamePtr, timestamp);
    });
  }

  /** {@override GraphRunnerApi} */
  addIntVectorToStream(data: number[], streamName: string, timestamp: number):
      void {
    this.wrapStringPtr(streamName, (streamNamePtr: number) => {
      const vecPtr = this.wasmModule._allocateIntVector(data.length);
      if (!vecPtr) {
        throw new Error('Unable to allocate new int vector on heap.');
      }
      for (const entry of data) {
        this.wasmModule._addIntVectorEntry(vecPtr, entry);
      }
      this.wasmModule._addIntVectorToInputStream(
          vecPtr, streamNamePtr, timestamp);
    });
  }

  /** {@override GraphRunnerApi} */
  addUintVectorToStream(data: number[], streamName: string, timestamp: number):
      void {
    this.wrapStringPtr(streamName, (streamNamePtr: number) => {
      const vecPtr = this.wasmModule._allocateUintVector(data.length);
      if (!vecPtr) {
        throw new Error('Unable to allocate new unsigned int vector on heap.');
      }
      for (const entry of data) {
        this.wasmModule._addUintVectorEntry(vecPtr, entry);
      }
      this.wasmModule._addUintVectorToInputStream(
          vecPtr, streamNamePtr, timestamp);
    });
  }

  /** {@override GraphRunnerApi} */
  addStringVectorToStream(
      data: string[], streamName: string, timestamp: number): void {
    this.wrapStringPtr(streamName, (streamNamePtr: number) => {
      const vecPtr = this.wasmModule._allocateStringVector(data.length);
      if (!vecPtr) {
        throw new Error('Unable to allocate new string vector on heap.');
      }
      for (const entry of data) {
        this.wrapStringPtr(entry, (entryStringPtr: number) => {
          this.wasmModule._addStringVectorEntry(vecPtr, entryStringPtr);
        });
      }
      this.wasmModule._addStringVectorToInputStream(
          vecPtr, streamNamePtr, timestamp);
    });
  }

  /** {@override GraphRunnerApi} */
  addBoolToInputSidePacket(data: boolean, sidePacketName: string): void {
    this.wrapStringPtr(sidePacketName, (sidePacketNamePtr: number) => {
      this.wasmModule._addBoolToInputSidePacket(data, sidePacketNamePtr);
    });
  }

  /** {@override GraphRunnerApi} */
  addDoubleToInputSidePacket(data: number, sidePacketName: string): void {
    this.wrapStringPtr(sidePacketName, (sidePacketNamePtr: number) => {
      this.wasmModule._addDoubleToInputSidePacket(data, sidePacketNamePtr);
    });
  }

  /** {@override GraphRunnerApi} */
  addFloatToInputSidePacket(data: number, sidePacketName: string): void {
    this.wrapStringPtr(sidePacketName, (sidePacketNamePtr: number) => {
      this.wasmModule._addFloatToInputSidePacket(data, sidePacketNamePtr);
    });
  }

  /** {@override GraphRunnerApi} */
  addIntToInputSidePacket(data: number, sidePacketName: string): void {
    this.wrapStringPtr(sidePacketName, (sidePacketNamePtr: number) => {
      this.wasmModule._addIntToInputSidePacket(data, sidePacketNamePtr);
    });
  }

  /** {@override GraphRunnerApi} */
  addUintToInputSidePacket(data: number, sidePacketName: string): void {
    this.wrapStringPtr(sidePacketName, (sidePacketNamePtr: number) => {
      this.wasmModule._addUintToInputSidePacket(data, sidePacketNamePtr);
    });
  }

  /** {@override GraphRunnerApi} */
  addStringToInputSidePacket(data: string, sidePacketName: string): void {
    this.wrapStringPtr(sidePacketName, (sidePacketNamePtr: number) => {
      this.wrapStringPtr(data, (dataPtr: number) => {
        this.wasmModule._addStringToInputSidePacket(dataPtr, sidePacketNamePtr);
      });
    });
  }

  /** {@override GraphRunnerApi} */
  addProtoToInputSidePacket(
      data: Uint8Array, protoType: string, sidePacketName: string): void {
    this.wrapStringPtr(sidePacketName, (sidePacketNamePtr: number) => {
      this.wrapStringPtr(protoType, (protoTypePtr: number) => {
        // Deep-copy proto data into Wasm heap
        const dataPtr = this.wasmModule._malloc(data.length);
        // TODO: Ensure this is the fastest way to copy this data.
        this.wasmModule.HEAPU8.set(data, dataPtr);
        this.wasmModule._addProtoToInputSidePacket(
            dataPtr, data.length, protoTypePtr, sidePacketNamePtr);
        this.wasmModule._free(dataPtr);
      });
    });
  }

  /** {@override GraphRunnerApi} */
  addBoolVectorToInputSidePacket(data: boolean[], sidePacketName: string):
      void {
    this.wrapStringPtr(sidePacketName, (sidePacketNamePtr: number) => {
      const vecPtr = this.wasmModule._allocateBoolVector(data.length);
      if (!vecPtr) {
        throw new Error('Unable to allocate new bool vector on heap.');
      }
      for (const entry of data) {
        this.wasmModule._addBoolVectorEntry(vecPtr, entry);
      }
      this.wasmModule._addBoolVectorToInputSidePacket(
          vecPtr, sidePacketNamePtr);
    });
  }

  /** {@override GraphRunnerApi} */
  addDoubleVectorToInputSidePacket(data: number[], sidePacketName: string):
      void {
    this.wrapStringPtr(sidePacketName, (sidePacketNamePtr: number) => {
      const vecPtr = this.wasmModule._allocateDoubleVector(data.length);
      if (!vecPtr) {
        throw new Error('Unable to allocate new double vector on heap.');
      }
      for (const entry of data) {
        this.wasmModule._addDoubleVectorEntry(vecPtr, entry);
      }
      this.wasmModule._addDoubleVectorToInputSidePacket(
          vecPtr, sidePacketNamePtr);
    });
  }

  /** {@override GraphRunnerApi} */
  addFloatVectorToInputSidePacket(data: number[], sidePacketName: string):
      void {
    this.wrapStringPtr(sidePacketName, (sidePacketNamePtr: number) => {
      const vecPtr = this.wasmModule._allocateFloatVector(data.length);
      if (!vecPtr) {
        throw new Error('Unable to allocate new float vector on heap.');
      }
      for (const entry of data) {
        this.wasmModule._addFloatVectorEntry(vecPtr, entry);
      }
      this.wasmModule._addFloatVectorToInputSidePacket(
          vecPtr, sidePacketNamePtr);
    });
  }

  /** {@override GraphRunnerApi} */
  addIntVectorToInputSidePacket(data: number[], sidePacketName: string): void {
    this.wrapStringPtr(sidePacketName, (sidePacketNamePtr: number) => {
      const vecPtr = this.wasmModule._allocateIntVector(data.length);
      if (!vecPtr) {
        throw new Error('Unable to allocate new int vector on heap.');
      }
      for (const entry of data) {
        this.wasmModule._addIntVectorEntry(vecPtr, entry);
      }
      this.wasmModule._addIntVectorToInputSidePacket(vecPtr, sidePacketNamePtr);
    });
  }

  /** {@override GraphRunnerApi} */
  addUintVectorToInputSidePacket(data: number[], sidePacketName: string): void {
    this.wrapStringPtr(sidePacketName, (sidePacketNamePtr: number) => {
      const vecPtr = this.wasmModule._allocateUintVector(data.length);
      if (!vecPtr) {
        throw new Error('Unable to allocate new unsigned int vector on heap.');
      }
      for (const entry of data) {
        this.wasmModule._addUintVectorEntry(vecPtr, entry);
      }
      this.wasmModule._addUintVectorToInputSidePacket(
          vecPtr, sidePacketNamePtr);
    });
  }

  /** {@override GraphRunnerApi} */
  addStringVectorToInputSidePacket(data: string[], sidePacketName: string):
      void {
    this.wrapStringPtr(sidePacketName, (sidePacketNamePtr: number) => {
      const vecPtr = this.wasmModule._allocateStringVector(data.length);
      if (!vecPtr) {
        throw new Error('Unable to allocate new string vector on heap.');
      }
      for (const entry of data) {
        this.wrapStringPtr(entry, (entryStringPtr: number) => {
          this.wasmModule._addStringVectorEntry(vecPtr, entryStringPtr);
        });
      }
      this.wasmModule._addStringVectorToInputSidePacket(
          vecPtr, sidePacketNamePtr);
    });
  }

  /** {@override GraphRunnerApi} */
  attachBoolListener(
      outputStreamName: string, callbackFcn: SimpleListener<boolean>): void {
    // Set up our TS listener to receive any packets for this stream.
    this.setListener(outputStreamName, callbackFcn);

    // Tell our graph to listen for bool packets on this stream.
    this.wrapStringPtr(outputStreamName, (outputStreamNamePtr: number) => {
      this.wasmModule._attachBoolListener(outputStreamNamePtr);
    });
  }

  /** {@override GraphRunnerApi} */
  attachBoolVectorListener(
      outputStreamName: string, callbackFcn: SimpleListener<boolean[]>): void {
    // Set up our TS listener to receive any packets for this stream.
    this.setVectorListener(outputStreamName, callbackFcn);

    // Tell our graph to listen for std::vector<bool> packets on this stream.
    this.wrapStringPtr(outputStreamName, (outputStreamNamePtr: number) => {
      this.wasmModule._attachBoolVectorListener(outputStreamNamePtr);
    });
  }

  /** {@override GraphRunnerApi} */
  attachIntListener(
      outputStreamName: string, callbackFcn: SimpleListener<number>): void {
    // Set up our TS listener to receive any packets for this stream.
    this.setListener(outputStreamName, callbackFcn);

    // Tell our graph to listen for int packets on this stream.
    this.wrapStringPtr(outputStreamName, (outputStreamNamePtr: number) => {
      this.wasmModule._attachIntListener(outputStreamNamePtr);
    });
  }

  /** {@override GraphRunnerApi} */
  attachIntVectorListener(
      outputStreamName: string, callbackFcn: SimpleListener<number[]>): void {
    // Set up our TS listener to receive any packets for this stream.
    this.setVectorListener(outputStreamName, callbackFcn);

    // Tell our graph to listen for std::vector<int> packets on this stream.
    this.wrapStringPtr(outputStreamName, (outputStreamNamePtr: number) => {
      this.wasmModule._attachIntVectorListener(outputStreamNamePtr);
    });
  }

  /** {@override GraphRunnerApi} */
  attachUintListener(
      outputStreamName: string, callbackFcn: SimpleListener<number>): void {
    // Set up our TS listener to receive any packets for this stream.
    this.setListener(outputStreamName, callbackFcn);

    // Tell our graph to listen for uint32_t packets on this stream.
    this.wrapStringPtr(outputStreamName, (outputStreamNamePtr: number) => {
      this.wasmModule._attachUintListener(outputStreamNamePtr);
    });
  }

  /** {@override GraphRunnerApi} */
  attachUintVectorListener(
      outputStreamName: string, callbackFcn: SimpleListener<number[]>): void {
    // Set up our TS listener to receive any packets for this stream.
    this.setVectorListener(outputStreamName, callbackFcn);

    // Tell our graph to listen for std::vector<uint32_t> packets on this
    // stream.
    this.wrapStringPtr(outputStreamName, (outputStreamNamePtr: number) => {
      this.wasmModule._attachUintVectorListener(outputStreamNamePtr);
    });
  }

  /** {@override GraphRunnerApi} */
  attachDoubleListener(
      outputStreamName: string, callbackFcn: SimpleListener<number>): void {
    // Set up our TS listener to receive any packets for this stream.
    this.setListener(outputStreamName, callbackFcn);

    // Tell our graph to listen for double packets on this stream.
    this.wrapStringPtr(outputStreamName, (outputStreamNamePtr: number) => {
      this.wasmModule._attachDoubleListener(outputStreamNamePtr);
    });
  }

  /** {@override GraphRunnerApi} */
  attachDoubleVectorListener(
      outputStreamName: string, callbackFcn: SimpleListener<number[]>): void {
    // Set up our TS listener to receive any packets for this stream.
    this.setVectorListener(outputStreamName, callbackFcn);

    // Tell our graph to listen for std::vector<double> packets on this stream.
    this.wrapStringPtr(outputStreamName, (outputStreamNamePtr: number) => {
      this.wasmModule._attachDoubleVectorListener(outputStreamNamePtr);
    });
  }

  /** {@override GraphRunnerApi} */
  attachFloatListener(
      outputStreamName: string, callbackFcn: SimpleListener<number>): void {
    // Set up our TS listener to receive any packets for this stream.
    this.setListener(outputStreamName, callbackFcn);

    // Tell our graph to listen for float packets on this stream.
    this.wrapStringPtr(outputStreamName, (outputStreamNamePtr: number) => {
      this.wasmModule._attachFloatListener(outputStreamNamePtr);
    });
  }

  /** {@override GraphRunnerApi} */
  attachFloatVectorListener(
      outputStreamName: string, callbackFcn: SimpleListener<number[]>): void {
    // Set up our TS listener to receive any packets for this stream.
    this.setVectorListener(outputStreamName, callbackFcn);

    // Tell our graph to listen for std::vector<float> packets on this stream.
    this.wrapStringPtr(outputStreamName, (outputStreamNamePtr: number) => {
      this.wasmModule._attachFloatVectorListener(outputStreamNamePtr);
    });
  }

  /** {@override GraphRunnerApi} */
  attachStringListener(
      outputStreamName: string, callbackFcn: SimpleListener<string>): void {
    // Set up our TS listener to receive any packets for this stream.
    this.setListener(outputStreamName, callbackFcn);

    // Tell our graph to listen for string packets on this stream.
    this.wrapStringPtr(outputStreamName, (outputStreamNamePtr: number) => {
      this.wasmModule._attachStringListener(outputStreamNamePtr);
    });
  }

  /** {@override GraphRunnerApi} */
  attachStringVectorListener(
      outputStreamName: string, callbackFcn: SimpleListener<string[]>): void {
    // Set up our TS listener to receive any packets for this stream.
    this.setVectorListener(outputStreamName, callbackFcn);

    // Tell our graph to listen for std::vector<string> packets on this stream.
    this.wrapStringPtr(outputStreamName, (outputStreamNamePtr: number) => {
      this.wasmModule._attachStringVectorListener(outputStreamNamePtr);
    });
  }

  /** {@override GraphRunnerApi} */
  attachProtoListener(
      outputStreamName: string, callbackFcn: SimpleListener<Uint8Array>,
      makeDeepCopy?: boolean): void {
    // Set up our TS listener to receive any packets for this stream.
    this.setListener(outputStreamName, callbackFcn);

    // Tell our graph to listen for binary serialized proto data packets on this
    // stream.
    this.wrapStringPtr(outputStreamName, (outputStreamNamePtr: number) => {
      this.wasmModule._attachProtoListener(
          outputStreamNamePtr, makeDeepCopy || false);
    });
  }

  /** {@override GraphRunnerApi} */
  attachProtoVectorListener(
      outputStreamName: string, callbackFcn: SimpleListener<Uint8Array[]>,
      makeDeepCopy?: boolean): void {
    // Set up our TS listener to receive any packets for this stream.
    this.setVectorListener(outputStreamName, callbackFcn);

    // Tell our graph to listen for a vector of binary serialized proto packets
    // on this stream.
    this.wrapStringPtr(outputStreamName, (outputStreamNamePtr: number) => {
      this.wasmModule._attachProtoVectorListener(
          outputStreamNamePtr, makeDeepCopy || false);
    });
  }

  /** {@override GraphRunnerApi} */
  attachAudioListener(
      outputStreamName: string, callbackFcn: SimpleListener<Float32Array>,
      makeDeepCopy?: boolean): void {
    if (!this.wasmModule._attachAudioListener) {
      console.warn(
          'Attempting to use attachAudioListener without support for ' +
          'output audio. Is build dep ":gl_graph_runner_audio_out" missing?');
    }

    // Set up our TS listener to receive any packets for this stream, and
    // additionally reformat our Uint8Array into a Float32Array for the user.
    this.setListener<Uint8Array>(outputStreamName, (data, timestamp) => {
      // Should be very fast
      const floatArray =
          new Float32Array(data.buffer, data.byteOffset, data.length / 4);
      callbackFcn(floatArray, timestamp);
    });

    // Tell our graph to listen for string packets on this stream.
    this.wrapStringPtr(outputStreamName, (outputStreamNamePtr: number) => {
      this.wasmModule._attachAudioListener(
          outputStreamNamePtr, makeDeepCopy || false);
    });
  }

  /** {@override GraphRunnerApi} */
  finishProcessing(): void {
    this.wasmModule._waitUntilIdle();
  }

  /** {@override GraphRunnerApi} */
  closeGraph(): void {
    this.wasmModule._closeGraph();
    this.wasmModule.simpleListeners = undefined;
    this.wasmModule.emptyPacketListeners = undefined;
  }
}

/** {@override CreateMediaPipeLibApi} */
export const createMediaPipeLib: CreateMediaPipeLibApi = async<LibType>(
    constructorFcn: WasmMediaPipeConstructor<LibType>,
    wasmLoaderScript?: string|null,
    assetLoaderScript?: string|null,
    glCanvas?: HTMLCanvasElement|OffscreenCanvas|null,
    fileLocator?: FileLocator): Promise<LibType> => {
  // Run wasm-loader script here
  if (wasmLoaderScript) {
    await runScript(wasmLoaderScript);
  }

  if (!self.ModuleFactory) {
    throw new Error('ModuleFactory not set.');
  }

  // Run asset-loader script here; must be run after wasm-loader script if we
  // are re-wrapping the existing MODULARIZE export.
  if (assetLoaderScript) {
    await runScript(assetLoaderScript);
    if (!self.ModuleFactory) {
      throw new Error('ModuleFactory not set.');
    }
  }

  // Until asset scripts work nicely with MODULARIZE, when we are given both
  // self.Module and a fileLocator, we manually merge them into self.Module and
  // use that. TODO: Remove this when asset scripts are fixed.
  if (self.Module && fileLocator) {
    const moduleFileLocator = self.Module as FileLocator;
    moduleFileLocator.locateFile = fileLocator.locateFile;
    if (fileLocator.mainScriptUrlOrBlob) {
      moduleFileLocator.mainScriptUrlOrBlob = fileLocator.mainScriptUrlOrBlob;
    }
  }
  // TODO: Ensure that fileLocator is passed in by all users
  // and make it required
  const module =
      await self.ModuleFactory(self.Module as FileLocator || fileLocator);
  // Don't reuse factory or module seed
  self.ModuleFactory = self.Module = undefined;
  return new constructorFcn(module, glCanvas);
};

// We extend the CreateGraphRunnerApi interface here for now so that by default
// callers of `createGraphRunner` will be given a `GraphRunner` rather than a
// `GraphRunnerApi`.
interface CreateGraphRunnerImplType extends CreateGraphRunnerApi {
  (wasmLoaderScript?: string,
   assetLoaderScript?: string,
   glCanvas?: HTMLCanvasElement|OffscreenCanvas|null,
   fileLocator?: FileLocator): Promise<GraphRunner>;
}

/** {@override CreateGraphRunnerApi} */
export const createGraphRunner: CreateGraphRunnerImplType = async(
    wasmLoaderScript?: string,
    assetLoaderScript?: string,
    glCanvas?: HTMLCanvasElement|OffscreenCanvas|null,
    fileLocator?: FileLocator): Promise<GraphRunner> => {
  return createMediaPipeLib(
      GraphRunner, wasmLoaderScript, assetLoaderScript, glCanvas, fileLocator);
};
