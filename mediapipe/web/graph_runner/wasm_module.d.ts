import {EmptyPacketListener, ErrorListener, SimpleListener, VectorListener} from './listener_types';

/**
 * Declarations for Emscripten's WebAssembly Module behavior, so TS compiler
 * doesn't break our various JS/C++ bridges. For internal usage.
 */
export declare interface WasmModule {
  canvas: HTMLCanvasElement|OffscreenCanvas|null;
  HEAPU8: Uint8Array;
  HEAPU32: Uint32Array;
  HEAPF32: Float32Array;
  HEAPF64: Float64Array;
  FS_createDataFile:
      (parent: string, name: string, data: Uint8Array, canRead: boolean,
       canWrite: boolean, canOwn: boolean) => void;
  FS_createPath:
      (parent: string, name: string, canRead: boolean,
       canWrite: boolean) => void;
  FS_unlink(path: string): void;
  gpuOriginForWebTexturesIsBottomLeft?: boolean;

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
  _addUintToInputStream:
      (data: number, streamNamePtr: number, timestamp: number) => void;
  _addStringToInputStream:
      (dataPtr: number, streamNamePtr: number, timestamp: number) => void;
  _addBoolVectorToInputStream:
      (vecPtr: number, streamNamePtr: number, timestamp: number) => void;
  _addDoubleVectorToInputStream:
      (vecPtr: number, streamNamePtr: number, timestamp: number) => void;
  _addFloatVectorToInputStream:
      (vecPtr: number, streamNamePtr: number, timestamp: number) => void;
  _addIntVectorToInputStream:
      (vecPtr: number, streamNamePtr: number, timestamp: number) => void;
  _addUintVectorToInputStream:
      (vecPtr: number, streamNamePtr: number, timestamp: number) => void;
  _addStringVectorToInputStream:
      (vecPtr: number, streamNamePtr: number, timestamp: number) => void;
  _addFlatHashMapToInputStream:
      (keysPtr: number, valuesPtr: number, count: number, streamNamePtr: number,
       timestamp: number) => void;
  _addRawDataSpanToInputStream:
      (dataPtr: number, dataSize: number, streamNamePtr: number,
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
  _addUintToInputSidePacket:
      (data: number, streamNamePtr: number) => void;
  _addStringToInputSidePacket: (dataPtr: number, streamNamePtr: number) => void;
  _addRawDataSpanToInputSidePacket:
      (dataPtr: number, dataSize: number, streamNamePtr: number) => void;
  _addProtoToInputSidePacket:
      (dataPtr: number, dataSize: number, protoNamePtr: number,
       streamNamePtr: number) => void;
  _addBoolVectorToInputSidePacket:
      (vecPtr: number, streamNamePtr: number) => void;
  _addDoubleVectorToInputSidePacket:
      (vecPtr: number, streamNamePtr: number) => void;
  _addFloatVectorToInputSidePacket:
      (vecPtr: number, streamNamePtr: number) => void;
  _addIntVectorToInputSidePacket:
      (vecPtr: number, streamNamePtr: number) => void;
  _addUintVectorToInputSidePacket:
      (vecPtr: number, streamNamePtr: number) => void;
  _addStringVectorToInputSidePacket:
      (vecPtr: number, streamNamePtr: number) => void;

  // Vector input creation
  _allocateBoolVector: (size: number) => number;
  _allocateDoubleVector: (size: number) => number;
  _allocateFloatVector: (size: number) => number;
  _allocateIntVector: (size: number) => number;
  _allocateUintVector: (size: number) => number;
  _allocateStringVector: (size: number) => number;
  _addBoolVectorEntry: (vecPtr: number, entry: boolean) => void;
  _addDoubleVectorEntry: (vecPtr: number, entry: number) => void;
  _addFloatVectorEntry: (vecPtr: number, entry: number) => void;
  _addIntVectorEntry: (vecPtr: number, entry: number) => void;
  _addUintVectorEntry: (vecPtr: number, entry: number) => void;
  _addStringVectorEntry: (vecPtr: number, entryStringPtr: number) => void;

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
  _attachUintListener: (streamNamePtr: number) => void;
  _attachUintVectorListener: (streamNamePtr: number) => void;
  _attachStringListener: (streamNamePtr: number) => void;
  _attachStringVectorListener: (streamNamePtr: number) => void;
  _attachProtoListener: (streamNamePtr: number, makeDeepCopy?: boolean) => void;
  _attachProtoVectorListener:
      (streamNamePtr: number, makeDeepCopy?: boolean) => void;

  // Require dependency ":gl_graph_runner_audio_out"
  _attachAudioListener: (streamNamePtr: number, makeDeepCopy?: boolean) => void;

  // Require dependency ":gl_graph_runner_audio"
  _addAudioToInputStream:
      (dataPtr: number, numChannels: number, numSamples: number,
       streamNamePtr: number, timestamp: number) => void;
  _configureAudio:
      (channels: number, samples: number, sampleRate: number,
       streamNamePtr: number, headerNamePtr: number) => void;

  // Get the graph configuration and invoke the listener configured under
  // streamNamePtr
  _getGraphConfig: (streamNamePtr: number, makeDeepCopy?: boolean) => void;

  // TODO: Refactor to just use a few numbers (perhaps refactor away
  //   from gl_graph_runner_internal.cc entirely to use something a little more
  //   streamlined; new version is _processFrame above).
  _processGl: (frameDataPtr: number) => number;
}
