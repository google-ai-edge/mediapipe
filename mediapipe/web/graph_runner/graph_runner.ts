// Placeholder for internal dependency on assertTruthy
// Placeholder for internal dependency on jsloader
import {isWebKit} from '../../web/graph_runner/platform_utils';
// Placeholder for internal dependency on trusted resource url

// This file can serve as a common interface for most simple TypeScript
// libraries-- additionally, it can hook automatically into wasm_mediapipe_demo
// to autogenerate simple TS APIs from demos for instantaneous 1P integrations.

/**
 * Simple interface for allowing users to set the directory where internal
 * wasm-loading and asset-loading code looks (e.g. for .wasm and .data file
 * locations).
 */
export declare interface FileLocator {
  locateFile: (filename: string) => string;
  mainScriptUrlOrBlob?: string;
}

/**
 * A listener that receives the contents of a non-empty MediaPipe packet and
 * its timestamp.
 */
export type SimpleListener<T> = (data: T, timestamp: number) => void;

/**
 * A listener that receives the current MediaPipe packet timestamp. This is
 * invoked even for empty packet.
 */
export type EmptyPacketListener = (timestamp: number) => void;

/**
 * A listener that receives a single element of vector-returning output packet.
 * Receives one element at a time (in order). Once all elements are processed,
 * the listener is invoked with `data` set to `unknown` and `done` set to true.
 * Intended for internal usage.
 */
export type VectorListener<T> = (data: T, done: boolean, timestamp: number) =>
    void;

/**
 * A listener that receives the CalculatorGraphConfig in binary encoding.
 */
export type CalculatorGraphConfigListener = (graphConfig: Uint8Array) => void;

/**
 * The name of the internal listener that we use to obtain the calculator graph
 * config. Intended for internal usage. Exported for testing only.
 */
export const CALCULATOR_GRAPH_CONFIG_LISTENER_NAME = '__graph_config__';

/**
 * Declarations for Emscripten's WebAssembly Module behavior, so TS compiler
 * doesn't break our JS/C++ bridge.
 */
export declare interface WasmModule {
  canvas: HTMLCanvasElement|OffscreenCanvas|null;
  HEAPU8: Uint8Array;
  HEAPU32: Uint32Array;
  HEAPF32: Float32Array;
  HEAPF64: Float64Array;
  errorListener?: ErrorListener;
  _bindTextureToCanvas: () => boolean;
  _changeBinaryGraph: (size: number, dataPtr: number) => void;
  _changeTextGraph: (size: number, dataPtr: number) => void;
  _closeGraph: () => void;
  _free: (ptr: number) => void;
  _malloc: (size: number) => number;
  _processFrame: (width: number, height: number, timestamp: number) => void;
  _setAutoRenderToScreen: (enabled: boolean) => void;
  _waitUntilIdle: () => void;

  // Exposed so that clients of this lib can access this field
  dataFileDownloads?: {[url: string]: {loaded: number, total: number}};

  // Wasm Module multistream entrypoints.  Require
  // gl_graph_runner_internal_multi_input as a build dependency.
  stringToNewUTF8: (data: string) => number;
  _bindTextureToStream: (streamNamePtr: number) => void;
  _addBoundTextureToStream:
      (streamNamePtr: number, width: number, height: number,
       timestamp: number) => void;
  _addBoolToInputStream:
      (data: boolean, streamNamePtr: number, timestamp: number) => void;
  _addDoubleToInputStream:
      (data: number, streamNamePtr: number, timestamp: number) => void;
  _addFloatToInputStream:
      (data: number, streamNamePtr: number, timestamp: number) => void;
  _addIntToInputStream:
      (data: number, streamNamePtr: number, timestamp: number) => void;
  _addStringToInputStream:
      (dataPtr: number, streamNamePtr: number, timestamp: number) => void;
  _addFlatHashMapToInputStream:
      (keysPtr: number, valuesPtr: number, count: number, streamNamePtr: number,
       timestamp: number) => void;
  _addProtoToInputStream:
      (dataPtr: number, dataSize: number, protoNamePtr: number,
       streamNamePtr: number, timestamp: number) => void;
  _addEmptyPacketToInputStream:
      (streamNamePtr: number, timestamp: number) => void;

  // Input side packets
  _addBoolToInputSidePacket: (data: boolean, streamNamePtr: number) => void;
  _addDoubleToInputSidePacket: (data: number, streamNamePtr: number) => void;
  _addFloatToInputSidePacket: (data: number, streamNamePtr: number) => void;
  _addIntToInputSidePacket: (data: number, streamNamePtr: number) => void;
  _addStringToInputSidePacket: (dataPtr: number, streamNamePtr: number) => void;
  _addProtoToInputSidePacket:
      (dataPtr: number, dataSize: number, protoNamePtr: number,
       streamNamePtr: number) => void;

  // Map of output streams to packet listeners.  Also built as part of
  // gl_graph_runner_internal_multi_input.
  simpleListeners?:
      Record<string, SimpleListener<unknown>|VectorListener<unknown>>;
  // Map of output streams to empty packet listeners.
  emptyPacketListeners?: Record<string, EmptyPacketListener>;
  _attachBoolListener: (streamNamePtr: number) => void;
  _attachBoolVectorListener: (streamNamePtr: number) => void;
  _attachDoubleListener: (streamNamePtr: number) => void;
  _attachDoubleVectorListener: (streamNamePtr: number) => void;
  _attachFloatListener: (streamNamePtr: number) => void;
  _attachFloatVectorListener: (streamNamePtr: number) => void;
  _attachIntListener: (streamNamePtr: number) => void;
  _attachIntVectorListener: (streamNamePtr: number) => void;
  _attachStringListener: (streamNamePtr: number) => void;
  _attachStringVectorListener: (streamNamePtr: number) => void;
  _attachProtoListener: (streamNamePtr: number, makeDeepCopy?: boolean) => void;
  _attachProtoVectorListener:
      (streamNamePtr: number, makeDeepCopy?: boolean) => void;

  // Require dependency ":gl_graph_runner_audio_out"
  _attachAudioListener: (streamNamePtr: number, makeDeepCopy?: boolean) => void;

  // Require dependency ":gl_graph_runner_audio"
  _addAudioToInputStream: (dataPtr: number, numChannels: number,
      numSamples: number, streamNamePtr: number, timestamp: number) => void;
  _configureAudio: (channels: number, samples: number, sampleRate: number,
      streamNamePtr: number, headerNamePtr: number) => void;

  // Get the graph configuration and invoke the listener configured under
  // streamNamePtr
  _getGraphConfig: (streamNamePtr: number, makeDeepCopy?: boolean) => void;

  // TODO: Refactor to just use a few numbers (perhaps refactor away
  //   from gl_graph_runner_internal.cc entirely to use something a little more
  //   streamlined; new version is _processFrame above).
  _processGl: (frameDataPtr: number) => number;
}

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
 * Valid types of image sources which we can run our GraphRunner over.
 */
export type ImageSource =
    HTMLCanvasElement|HTMLVideoElement|HTMLImageElement|ImageData|ImageBitmap;


/** A listener that will be invoked with an absl::StatusCode and message. */
export type ErrorListener = (code: number, message: string) => void;

/**
 * Internal type of constructors used for initializing GraphRunner and
 * subclasses.
 */
export type WasmMediaPipeConstructor<LibType> =
    (new (
         module: WasmModule, canvas?: HTMLCanvasElement|OffscreenCanvas|null) =>
         LibType);

/**
 * Simple class to run an arbitrary image-in/image-out MediaPipe graph (i.e.
 * as created by wasm_mediapipe_demo BUILD macro), and either render results
 * into canvas, or else return the output WebGLTexture. Takes a WebAssembly
 * Module.
 */
export class GraphRunner {
  // TODO: These should be protected/private, but are left exposed for
  //   now so that we can use proper TS mixins with this class as a base. This
  //   should be somewhat fixed when we create our .d.ts files.
  readonly wasmModule: WasmModule;
  readonly hasMultiStreamSupport: boolean;
  autoResizeCanvas: boolean = true;
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
    } else if (typeof OffscreenCanvas !== 'undefined' && !isWebKit()) {
      // If no canvas is provided, assume Chrome/Firefox and just make an
      // OffscreenCanvas for GPU processing. Note that we exclude Safari
      // since it does not (yet) support WebGL for OffscreenCanvas.
      this.wasmModule.canvas = new OffscreenCanvas(1, 1);
    } else {
      console.warn(
          'OffscreenCanvas not supported and GraphRunner constructor ' +
          'glCanvas parameter is undefined. Creating backup canvas.');
      this.wasmModule.canvas = document.createElement('canvas');
    }
  }

  /**
   * Convenience helper to load a MediaPipe graph from a file and pass it to
   * setGraph.
   * @param graphFile The url of the MediaPipe graph file to load.
   */
  async initializeGraph(graphFile: string): Promise<void> {
    // Fetch and set graph
    const response = await fetch(graphFile);
    const graphData = await response.arrayBuffer();
    const isBinary =
        !(graphFile.endsWith('.pbtxt') || graphFile.endsWith('.textproto'));
    this.setGraph(new Uint8Array(graphData), isBinary);
  }

  /**
   * Convenience helper for calling setGraph with a string representing a text
   * proto config.
   * @param graphConfig The text proto graph config, expected to be a string in
   * default JavaScript UTF-16 format.
   */
  setGraphFromString(graphConfig: string): void {
    this.setGraph((new TextEncoder()).encode(graphConfig), false);
  }

  /**
   * Takes the raw data from a MediaPipe graph, and passes it to C++ to be run
   * over the video stream. Will replace the previously running MediaPipe graph,
   * if there is one.
   * @param graphData The raw MediaPipe graph data, either in binary
   *     protobuffer format (.binarypb), or else in raw text format (.pbtxt or
   *     .textproto).
   * @param isBinary This should be set to true if the graph is in
   *     binary format, and false if it is in human-readable text format.
   */
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

  /**
   * Configures the current graph to handle audio processing in a certain way
   * for all its audio input streams. Additionally can configure audio headers
   * (both input side packets as well as input stream headers), but these
   * configurations only take effect if called before the graph is set/started.
   * @param numChannels The number of channels of audio input. Only 1
   *     is supported for now.
   * @param numSamples The number of samples that are taken in each
   *     audio capture.
   * @param sampleRate The rate, in Hz, of the sampling.
   * @param streamName The optional name of the input stream to additionally
   *     configure with audio information. This configuration only occurs before
   *     the graph is set/started. If unset, a default stream name will be used.
   * @param headerName The optional name of the header input side packet to
   *     additionally configure with audio information. This configuration only
   *     occurs before the graph is set/started. If unset, a default header name
   *     will be used.
   */
  configureAudio(numChannels: number, numSamples: number, sampleRate: number,
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
          numChannels, numSamples, sampleRate);
      });
    });
  }

  /**
   * Allows disabling automatic canvas resizing, in case clients want to control
   * control this.
   * @param resize True will re-enable automatic canvas resizing, while false
   *     will disable the feature.
   */
  setAutoResizeCanvas(resize: boolean): void {
    this.autoResizeCanvas = resize;
  }

  /**
   * Allows disabling the automatic render-to-screen code, in case clients don't
   * need/want this. In particular, this removes the requirement for pipelines
   * to have access to GPU resources, as well as the requirement for graphs to
   * have "input_frames_gpu" and "output_frames_gpu" streams defined, so pure
   * CPU pipelines and non-video pipelines can be created.
   * NOTE: This only affects future graph initializations (via setGraph or
   *     initializeGraph), and does NOT affect the currently running graph, so
   *     calls to this should be made *before* setGraph/initializeGraph for the
   *     graph file being targeted.
   * @param enabled True will re-enable automatic render-to-screen code and
   *     cause GPU resources to once again be requested, while false will
   *     disable the feature.
   */
  setAutoRenderToScreen(enabled: boolean): void {
    this.wasmModule._setAutoRenderToScreen(enabled);
  }

  /**
   * Bind texture to our internal canvas, and upload image source to GPU.
   * Returns tuple [width, height] of texture.  Intended for internal usage.
   */
  bindTextureToStream(imageSource: ImageSource, streamNamePtr?: number):
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
    gl.texImage2D(
        gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, imageSource);

    let width, height;
    if ((imageSource as HTMLVideoElement).videoWidth) {
      width = (imageSource as HTMLVideoElement).videoWidth;
      height = (imageSource as HTMLVideoElement).videoHeight;
    } else if ((imageSource as HTMLImageElement).naturalWidth) {
      // TODO: Ensure this works with SVG images
      width = (imageSource as HTMLImageElement).naturalWidth;
      height = (imageSource as HTMLImageElement).naturalHeight;
    } else {
      width = imageSource.width;
      height = imageSource.height;
    }

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
   * infrequently.
   * @param imageSource An image source to process.
   * @param timestamp The timestamp of the current frame, in ms.
   * @return texture? The WebGL texture reference, if one was produced.
   */
  processGl(imageSource: ImageSource, timestamp: number): WebGLTexture
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
   * avoids addding a direct dependency on the Protobuf JSPB target in the graph
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

  /**
   * Attaches a listener that will be invoked when the MediaPipe framework
   * returns an error.
   */
  attachErrorListener(callbackFcn: (code: number, message: string) => void) {
    this.wasmModule.errorListener = callbackFcn;
  }

  /**
   * Attaches a listener that will be invoked when the MediaPipe framework
   * receives an empty packet on the provided output stream. This can be used
   * to receive the latest output timestamp.
   *
   * Empty packet listeners are only active if there is a corresponding packet
   * listener.
   *
   * @param outputStreamName The name of the graph output stream to receive
   *    empty packets from.
   * @param callbackFcn The callback to receive the timestamp.
   */
  attachEmptyPacketListener(
      outputStreamName: string, callbackFcn: EmptyPacketListener) {
    this.wasmModule.emptyPacketListeners =
        this.wasmModule.emptyPacketListeners || {};
    this.wasmModule.emptyPacketListeners[outputStreamName] = callbackFcn;
  }

  /**
   * Takes the raw data from a JS audio capture array, and sends it to C++ to be
   * processed.
   * @param audioData An array of raw audio capture data, like
   *     from a call to getChannelData on an AudioBuffer.
   * @param streamName The name of the MediaPipe graph stream to add the audio
   *     data to.
   * @param timestamp The timestamp of the current frame, in ms.
   */
  addAudioToStream(
      audioData: Float32Array, streamName: string, timestamp: number) {
    // numChannels and numSamples being 0 will cause defaults to be used,
    // which will reflect values from last call to configureAudio.
    this.addAudioToStreamWithShape(audioData, 0, 0, streamName, timestamp);
  }

  /**
   * Takes the raw data from a JS audio capture array, and sends it to C++ to be
   * processed, shaping the audioData array into an audio matrix according to
   * the numChannels and numSamples parameters.
   * @param audioData An array of raw audio capture data, like
   *     from a call to getChannelData on an AudioBuffer.
   * @param numChannels The number of audio channels this data represents. If 0
   *     is passed, then the value will be taken from the last call to
   *     configureAudio.
   * @param numSamples The number of audio samples captured in this data packet.
   *     If 0 is passed, then the value will be taken from the last call to
   *     configureAudio.
   * @param streamName The name of the MediaPipe graph stream to add the audio
   *     data to.
   * @param timestamp The timestamp of the current frame, in ms.
   */
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

  /**
   * Takes the relevant information from the HTML video or image element, and
   * passes it into the WebGL-based graph for processing on the given stream at
   * the given timestamp. Can be used for additional auxiliary GpuBuffer input
   * streams. Processing will not occur until a blocking call (like
   * processVideoGl or finishProcessing) is made. For use with
   * 'gl_graph_runner_internal_multi_input'.
   * @param imageSource Reference to the video frame we wish to add into our
   *     graph.
   * @param streamName The name of the MediaPipe graph stream to add the frame
   *     to.
   * @param timestamp The timestamp of the input frame, in ms.
   */
  addGpuBufferToStream(
      imageSource: ImageSource, streamName: string, timestamp: number): void {
    this.wrapStringPtr(streamName, (streamNamePtr: number) => {
      const [width, height] =
          this.bindTextureToStream(imageSource, streamNamePtr);
      this.wasmModule._addBoundTextureToStream(
          streamNamePtr, width, height, timestamp);
    });
  }

  /**
   * Sends a boolean packet into the specified stream at the given timestamp.
   * @param data The boolean data to send.
   * @param streamName The name of the graph input stream to send data into.
   * @param timestamp The timestamp of the input data, in ms.
   */
  addBoolToStream(data: boolean, streamName: string, timestamp: number): void {
    this.wrapStringPtr(streamName, (streamNamePtr: number) => {
      this.wasmModule._addBoolToInputStream(data, streamNamePtr, timestamp);
    });
  }

  /**
   * Sends a double packet into the specified stream at the given timestamp.
   * @param data The double data to send.
   * @param streamName The name of the graph input stream to send data into.
   * @param timestamp The timestamp of the input data, in ms.
   */
  addDoubleToStream(data: number, streamName: string, timestamp: number): void {
    this.wrapStringPtr(streamName, (streamNamePtr: number) => {
      this.wasmModule._addDoubleToInputStream(data, streamNamePtr, timestamp);
    });
  }

  /**
   * Sends a float packet into the specified stream at the given timestamp.
   * @param data The float data to send.
   * @param streamName The name of the graph input stream to send data into.
   * @param timestamp The timestamp of the input data, in ms.
   */
  addFloatToStream(data: number, streamName: string, timestamp: number): void {
    this.wrapStringPtr(streamName, (streamNamePtr: number) => {
      // NOTE: _addFloatToStream and _addIntToStream are reserved for JS
      // Calculators currently; we may want to revisit this naming scheme in the
      // future.
      this.wasmModule._addFloatToInputStream(data, streamNamePtr, timestamp);
    });
  }

  /**
   * Sends an integer packet into the specified stream at the given timestamp.
   * @param data The integer data to send.
   * @param streamName The name of the graph input stream to send data into.
   * @param timestamp The timestamp of the input data, in ms.
   */
  addIntToStream(data: number, streamName: string, timestamp: number): void {
    this.wrapStringPtr(streamName, (streamNamePtr: number) => {
      this.wasmModule._addIntToInputStream(data, streamNamePtr, timestamp);
    });
  }

  /**
   * Sends a string packet into the specified stream at the given timestamp.
   * @param data The string data to send.
   * @param streamName The name of the graph input stream to send data into.
   * @param timestamp The timestamp of the input data, in ms.
   */
  addStringToStream(data: string, streamName: string, timestamp: number): void {
    this.wrapStringPtr(streamName, (streamNamePtr: number) => {
      this.wrapStringPtr(data, (dataPtr: number) => {
        this.wasmModule._addStringToInputStream(
            dataPtr, streamNamePtr, timestamp);
      });
    });
  }

  /**
   * Sends a Record<string, string> packet into the specified stream at the
   * given timestamp.
   * @param data The records to send (will become a
   *             std::flat_hash_map<std::string, std::string).
   * @param streamName The name of the graph input stream to send data into.
   * @param timestamp The timestamp of the input data, in ms.
   */
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

  /**
   * Sends a serialized protobuffer packet into the specified stream at the
   *     given timestamp, to be parsed into the specified protobuffer type.
   * @param data The binary (serialized) raw protobuffer data.
   * @param protoType The C++ namespaced type this protobuffer data corresponds
   *     to. It will be converted to this type when output as a packet into the
   *     graph.
   * @param streamName The name of the graph input stream to send data into.
   * @param timestamp The timestamp of the input data, in ms.
   */
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

  /**
   * Sends an empty packet into the specified stream at the given timestamp,
   *     effectively advancing that input stream's timestamp bounds without
   *     sending additional data packets.
   * @param streamName The name of the graph input stream to send the empty
   *     packet into.
   * @param timestamp The timestamp of the empty packet, in ms.
   */
  addEmptyPacketToStream(streamName: string, timestamp: number): void {
    this.wrapStringPtr(streamName, (streamNamePtr: number) => {
      this.wasmModule._addEmptyPacketToInputStream(streamNamePtr, timestamp);
    });
  }

  /**
   * Attaches a boolean packet to the specified input_side_packet.
   * @param data The boolean data to send.
   * @param sidePacketName The name of the graph input side packet to send data
   *     into.
   */
  addBoolToInputSidePacket(data: boolean, sidePacketName: string): void {
    this.wrapStringPtr(sidePacketName, (sidePacketNamePtr: number) => {
      this.wasmModule._addBoolToInputSidePacket(data, sidePacketNamePtr);
    });
  }

  /**
   * Attaches a double packet to the specified input_side_packet.
   * @param data The double data to send.
   * @param sidePacketName The name of the graph input side packet to send data
   *     into.
   */
  addDoubleToInputSidePacket(data: number, sidePacketName: string): void {
    this.wrapStringPtr(sidePacketName, (sidePacketNamePtr: number) => {
      this.wasmModule._addDoubleToInputSidePacket(data, sidePacketNamePtr);
    });
  }

  /**
   * Attaches a float packet to the specified input_side_packet.
   * @param data The float data to send.
   * @param sidePacketName The name of the graph input side packet to send data
   *     into.
   */
  addFloatToInputSidePacket(data: number, sidePacketName: string): void {
    this.wrapStringPtr(sidePacketName, (sidePacketNamePtr: number) => {
      this.wasmModule._addFloatToInputSidePacket(data, sidePacketNamePtr);
    });
  }

  /**
   * Attaches a integer packet to the specified input_side_packet.
   * @param data The integer data to send.
   * @param sidePacketName The name of the graph input side packet to send data
   *     into.
   */
  addIntToInputSidePacket(data: number, sidePacketName: string): void {
    this.wrapStringPtr(sidePacketName, (sidePacketNamePtr: number) => {
      this.wasmModule._addIntToInputSidePacket(data, sidePacketNamePtr);
    });
  }

  /**
   * Attaches a string packet to the specified input_side_packet.
   * @param data The string data to send.
   * @param sidePacketName The name of the graph input side packet to send data
   *     into.
   */
  addStringToInputSidePacket(data: string, sidePacketName: string): void {
    this.wrapStringPtr(sidePacketName, (sidePacketNamePtr: number) => {
      this.wrapStringPtr(data, (dataPtr: number) => {
        this.wasmModule._addStringToInputSidePacket(dataPtr, sidePacketNamePtr);
      });
    });
  }

  /**
   * Attaches a serialized proto packet to the specified input_side_packet.
   * @param data The binary (serialized) raw protobuffer data.
   * @param protoType The C++ namespaced type this protobuffer data corresponds
   *     to. It will be converted to this type for use in the graph.
   * @param sidePacketName The name of the graph input side packet to send data
   *     into.
   */
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

  /**
   * Attaches a boolean packet listener to the specified output_stream.
   * @param outputStreamName The name of the graph output stream to grab boolean
   *     data from.
   * @param callbackFcn The function that will be called back with the data, as
   *     it is received.  Note that the data is only guaranteed to exist for the
   *     duration of the callback, and the callback will be called inline, so it
   *     should not perform overly complicated (or any async) behavior.
   */
  attachBoolListener(
      outputStreamName: string, callbackFcn: SimpleListener<boolean>): void {
    // Set up our TS listener to receive any packets for this stream.
    this.setListener(outputStreamName, callbackFcn);

    // Tell our graph to listen for bool packets on this stream.
    this.wrapStringPtr(outputStreamName, (outputStreamNamePtr: number) => {
      this.wasmModule._attachBoolListener(outputStreamNamePtr);
    });
  }

  /**
   * Attaches a bool[] packet listener to the specified output_stream.
   * @param outputStreamName The name of the graph output stream to grab
   *     std::vector<bool> data from.
   * @param callbackFcn The function that will be called back with the data, as
   *     it is received.  Note that the data is only guaranteed to exist for the
   *     duration of the callback, and the callback will be called inline, so it
   *     should not perform overly complicated (or any async) behavior.
   */
  attachBoolVectorListener(
      outputStreamName: string, callbackFcn: SimpleListener<boolean[]>): void {
    // Set up our TS listener to receive any packets for this stream.
    this.setVectorListener(outputStreamName, callbackFcn);

    // Tell our graph to listen for std::vector<bool> packets on this stream.
    this.wrapStringPtr(outputStreamName, (outputStreamNamePtr: number) => {
      this.wasmModule._attachBoolVectorListener(outputStreamNamePtr);
    });
  }

  /**
   * Attaches an int packet listener to the specified output_stream.
   * @param outputStreamName The name of the graph output stream to grab int
   *     data from.
   * @param callbackFcn The function that will be called back with the data, as
   *     it is received.  Note that the data is only guaranteed to exist for the
   *     duration of the callback, and the callback will be called inline, so it
   *     should not perform overly complicated (or any async) behavior.
   */
  attachIntListener(
      outputStreamName: string, callbackFcn: SimpleListener<number>): void {
    // Set up our TS listener to receive any packets for this stream.
    this.setListener(outputStreamName, callbackFcn);

    // Tell our graph to listen for int packets on this stream.
    this.wrapStringPtr(outputStreamName, (outputStreamNamePtr: number) => {
      this.wasmModule._attachIntListener(outputStreamNamePtr);
    });
  }

  /**
   * Attaches an int[] packet listener to the specified output_stream.
   * @param outputStreamName The name of the graph output stream to grab
   *     std::vector<int> data from.
   * @param callbackFcn The function that will be called back with the data, as
   *     it is received.  Note that the data is only guaranteed to exist for the
   *     duration of the callback, and the callback will be called inline, so it
   *     should not perform overly complicated (or any async) behavior.
   */
  attachIntVectorListener(
      outputStreamName: string, callbackFcn: SimpleListener<number[]>): void {
    // Set up our TS listener to receive any packets for this stream.
    this.setVectorListener(outputStreamName, callbackFcn);

    // Tell our graph to listen for std::vector<int> packets on this stream.
    this.wrapStringPtr(outputStreamName, (outputStreamNamePtr: number) => {
      this.wasmModule._attachIntVectorListener(outputStreamNamePtr);
    });
  }

  /**
   * Attaches a double packet listener to the specified output_stream.
   * @param outputStreamName The name of the graph output stream to grab double
   *     data from.
   * @param callbackFcn The function that will be called back with the data, as
   *     it is received.  Note that the data is only guaranteed to exist for the
   *     duration of the callback, and the callback will be called inline, so it
   *     should not perform overly complicated (or any async) behavior.
   */
  attachDoubleListener(
      outputStreamName: string, callbackFcn: SimpleListener<number>): void {
    // Set up our TS listener to receive any packets for this stream.
    this.setListener(outputStreamName, callbackFcn);

    // Tell our graph to listen for double packets on this stream.
    this.wrapStringPtr(outputStreamName, (outputStreamNamePtr: number) => {
      this.wasmModule._attachDoubleListener(outputStreamNamePtr);
    });
  }

  /**
   * Attaches a double[] packet listener to the specified output_stream.
   * @param outputStreamName The name of the graph output stream to grab
   *     std::vector<double> data from.
   * @param callbackFcn The function that will be called back with the data, as
   *     it is received.  Note that the data is only guaranteed to exist for the
   *     duration of the callback, and the callback will be called inline, so it
   *     should not perform overly complicated (or any async) behavior.
   */
  attachDoubleVectorListener(
      outputStreamName: string, callbackFcn: SimpleListener<number[]>): void {
    // Set up our TS listener to receive any packets for this stream.
    this.setVectorListener(outputStreamName, callbackFcn);

    // Tell our graph to listen for std::vector<double> packets on this stream.
    this.wrapStringPtr(outputStreamName, (outputStreamNamePtr: number) => {
      this.wasmModule._attachDoubleVectorListener(outputStreamNamePtr);
    });
  }

  /**
   * Attaches a float packet listener to the specified output_stream.
   * @param outputStreamName The name of the graph output stream to grab float
   *     data from.
   * @param callbackFcn The function that will be called back with the data, as
   *     it is received.  Note that the data is only guaranteed to exist for the
   *     duration of the callback, and the callback will be called inline, so it
   *     should not perform overly complicated (or any async) behavior.
   */
  attachFloatListener(
      outputStreamName: string, callbackFcn: SimpleListener<number>): void {
    // Set up our TS listener to receive any packets for this stream.
    this.setListener(outputStreamName, callbackFcn);

    // Tell our graph to listen for float packets on this stream.
    this.wrapStringPtr(outputStreamName, (outputStreamNamePtr: number) => {
      this.wasmModule._attachFloatListener(outputStreamNamePtr);
    });
  }

  /**
   * Attaches a float[] packet listener to the specified output_stream.
   * @param outputStreamName The name of the graph output stream to grab
   *     std::vector<float> data from.
   * @param callbackFcn The function that will be called back with the data, as
   *     it is received.  Note that the data is only guaranteed to exist for the
   *     duration of the callback, and the callback will be called inline, so it
   *     should not perform overly complicated (or any async) behavior.
   */
  attachFloatVectorListener(
      outputStreamName: string, callbackFcn: SimpleListener<number[]>): void {
    // Set up our TS listener to receive any packets for this stream.
    this.setVectorListener(outputStreamName, callbackFcn);

    // Tell our graph to listen for std::vector<float> packets on this stream.
    this.wrapStringPtr(outputStreamName, (outputStreamNamePtr: number) => {
      this.wasmModule._attachFloatVectorListener(outputStreamNamePtr);
    });
  }

  /**
   * Attaches a string packet listener to the specified output_stream.
   * @param outputStreamName The name of the graph output stream to grab string
   *     data from.
   * @param callbackFcn The function that will be called back with the data, as
   *     it is received.  Note that the data is only guaranteed to exist for the
   *     duration of the callback, and the callback will be called inline, so it
   *     should not perform overly complicated (or any async) behavior.
   */
  attachStringListener(
      outputStreamName: string, callbackFcn: SimpleListener<string>): void {
    // Set up our TS listener to receive any packets for this stream.
    this.setListener(outputStreamName, callbackFcn);

    // Tell our graph to listen for string packets on this stream.
    this.wrapStringPtr(outputStreamName, (outputStreamNamePtr: number) => {
      this.wasmModule._attachStringListener(outputStreamNamePtr);
    });
  }

  /**
   * Attaches a string[] packet listener to the specified output_stream.
   * @param outputStreamName The name of the graph output stream to grab
   *     std::vector<std::string> data from.
   * @param callbackFcn The function that will be called back with the data, as
   *     it is received.  Note that the data is only guaranteed to exist for the
   *     duration of the callback, and the callback will be called inline, so it
   *     should not perform overly complicated (or any async) behavior.
   */
  attachStringVectorListener(
      outputStreamName: string, callbackFcn: SimpleListener<string[]>): void {
    // Set up our TS listener to receive any packets for this stream.
    this.setVectorListener(outputStreamName, callbackFcn);

    // Tell our graph to listen for std::vector<string> packets on this stream.
    this.wrapStringPtr(outputStreamName, (outputStreamNamePtr: number) => {
      this.wasmModule._attachStringVectorListener(outputStreamNamePtr);
    });
  }

  /**
   * Attaches a serialized proto packet listener to the specified output_stream.
   * @param outputStreamName The name of the graph output stream to grab binary
   *     serialized proto data from (in Uint8Array format).
   * @param callbackFcn The function that will be called back with the data, as
   *     it is received.  Note that by default the data is only guaranteed to
   *     exist for the duration of the callback, and the callback will be called
   *     inline, so it should not perform overly complicated (or any async)
   *     behavior. If the proto data needs to be able to outlive the call, you
   *     may set the optional makeDeepCopy parameter to true, or can manually
   *     deep-copy the data yourself.
   * @param makeDeepCopy Optional convenience parameter which, if set to true,
   *     will override the default memory management behavior and make a deep
   *     copy of the underlying data, rather than just returning a view into the
   *     C++-managed memory. At the cost of a data copy, this allows the
   *     returned data to outlive the callback lifetime (and it will be cleaned
   *     up automatically by JS garbage collection whenever the user is finished
   *     with it).
   */
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

  /**
   * Attaches a listener for an array of serialized proto packets to the
   * specified output_stream.
   * @param outputStreamName The name of the graph output stream to grab a
   *     vector of binary serialized proto data from (in Uint8Array[] format).
   * @param callbackFcn The function that will be called back with the data, as
   *     it is received.  Note that by default the data is only guaranteed to
   *     exist for the duration of the callback, and the callback will be called
   *     inline, so it should not perform overly complicated (or any async)
   *     behavior. If the proto data needs to be able to outlive the call, you
   *     may set the optional makeDeepCopy parameter to true, or can manually
   *     deep-copy the data yourself.
   * @param makeDeepCopy Optional convenience parameter which, if set to true,
   *     will override the default memory management behavior and make a deep
   *     copy of the underlying data, rather than just returning a view into the
   *     C++-managed memory. At the cost of a data copy, this allows the
   *     returned data to outlive the callback lifetime (and it will be cleaned
   *     up automatically by JS garbage collection whenever the user is finished
   *     with it).
   */
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

  /**
   * Attaches an audio packet listener to the specified output_stream, to be
   * given a Float32Array as output.
   * @param outputStreamName The name of the graph output stream to grab audio
   *     data from.
   * @param callbackFcn The function that will be called back with the data, as
   *     it is received. Note that the data is only guaranteed to exist for the
   *     duration of the callback, and the callback will be called inline, so it
   *     should not perform overly complicated (or any async) behavior. If the
   *     audio data needs to be able to outlive the call, you may set the
   *     optional makeDeepCopy parameter to true, or can manually deep-copy the
   *     data yourself.
   * @param makeDeepCopy Optional convenience parameter which, if set to true,
   *     will override the default memory management behavior and make a deep
   *     copy of the underlying data, rather than just returning a view into the
   *     C++-managed memory. At the cost of a data copy, this allows the
   *     returned data to outlive the callback lifetime (and it will be cleaned
   *     up automatically by JS garbage collection whenever the user is finished
   *     with it).
   */
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

  /**
   * Forces all queued-up packets to be pushed through the MediaPipe graph as
   * far as possible, performing all processing until no more processing can be
   * done.
   */
  finishProcessing(): void {
    this.wasmModule._waitUntilIdle();
  }

  /**
   * Closes the input streams and all calculators for this graph and frees up
   * any C++ resources. The graph will not be usable once closed.
   */
  closeGraph(): void {
    this.wasmModule._closeGraph();
    this.wasmModule.simpleListeners = undefined;
    this.wasmModule.emptyPacketListeners = undefined;
  }
}

// Quick private helper to run the given script safely
async function runScript(scriptUrl: string) {
  if (typeof importScripts === 'function') {
    importScripts(scriptUrl.toString());
  } else {
    const script = document.createElement('script');
    script.setAttribute('src', scriptUrl);
    script.setAttribute('crossorigin', 'anonymous');
    return new Promise<void>((resolve) => {
      script.addEventListener('load', () => {
        resolve();
      }, false);
      script.addEventListener('error', () => {
        resolve();
      }, false);
      document.body.appendChild(script);
    });
  }
}

/**
 * Helper type macro for use with createMediaPipeLib. Allows us to easily infer
 * the type of a mixin-extended GraphRunner. Example usage:
 * const GraphRunnerConstructor =
 *     SupportImage(SupportSerialization(GraphRunner));
 * let mediaPipe: ReturnType<typeof GraphRunnerConstructor>;
 * ...
 * mediaPipe = await createMediaPipeLib(GraphRunnerConstructor, ...);
 */
// tslint:disable-next-line:no-any
export type ReturnType<T> = T extends (...args: unknown[]) => infer R ? R : any;

/**
 * Global function to initialize Wasm blob and load runtime assets for a
 *     specialized MediaPipe library. This allows us to create a requested
 *     subclass inheriting from GraphRunner.
 * @param constructorFcn The name of the class to instantiate via "new".
 * @param wasmLoaderScript Url for the wasm-runner script; produced by the build
 *     process.
 * @param assetLoaderScript Url for the asset-loading script; produced by the
 *     build process.
 * @param fileLocator A function to override the file locations for assets
 *     loaded by the MediaPipe library.
 * @return promise A promise which will resolve when initialization has
 *     completed successfully.
 */
export async function createMediaPipeLib<LibType>(
    constructorFcn: WasmMediaPipeConstructor<LibType>,
    wasmLoaderScript?: string|null,
    assetLoaderScript?: string|null,
    glCanvas?: HTMLCanvasElement|OffscreenCanvas|null,
    fileLocator?: FileLocator): Promise<LibType> {
  const scripts = [];
  // Run wasm-loader script here
  if (wasmLoaderScript) {
    scripts.push(wasmLoaderScript);
  }
  // Run asset-loader script here
  if (assetLoaderScript) {
    scripts.push(assetLoaderScript);
  }
  // Load scripts in parallel, browser will execute them in sequence.
  if (scripts.length) {
    await Promise.all(scripts.map(runScript));
  }
  if (!self.ModuleFactory) {
    throw new Error('ModuleFactory not set.');
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
}

/**
 * Global function to initialize Wasm blob and load runtime assets for a generic
 *     MediaPipe library.
 * @param wasmLoaderScript Url for the wasm-runner script; produced by the build
 *     process.
 * @param assetLoaderScript Url for the asset-loading script; produced by the
 *     build process.
 * @param fileLocator A function to override the file locations for assets
 *     loaded by the MediaPipe library.
 * @return promise A promise which will resolve when initialization has
 *     completed successfully.
 */
export async function createGraphRunner(
    wasmLoaderScript?: string,
    assetLoaderScript?: string,
    glCanvas?: HTMLCanvasElement|OffscreenCanvas|null,
    fileLocator?: FileLocator): Promise<GraphRunner> {
  return createMediaPipeLib(
      GraphRunner, wasmLoaderScript, assetLoaderScript, glCanvas,
      fileLocator);
}
