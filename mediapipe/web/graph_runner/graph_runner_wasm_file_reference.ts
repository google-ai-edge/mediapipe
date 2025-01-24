import {GraphRunner, WasmModule} from '../../web/graph_runner/graph_runner';
// Placeholder for internal dependency on trusted resource url

/**
 * We extend from a GraphRunner constructor. This ensures our mixin has
 * access to the wasmModule, among other things. The `any` type is required for
 * mixin constructors.
 */
// tslint:disable-next-line:no-any
type LibConstructor = new (...args: any[]) => GraphRunner;

/**
 * WasmFileReference represents a reference to a span of raw data on the Wasm
 * heap, containing the data from a file. This can be useful for handling very
 * large files efficiently.
 *
 * The reference must be closed with `.free()` when the user is finished with
 * it, or else the memory will be retained indefinitely.
 */
export class WasmFileReference {
  private readonly wasmHeapOffset: number;
  private readonly wasmHeapView: Uint8Array;
  private isReferenceValid: boolean;

  constructor(
      private readonly wasmModule: WasmModule,
      private readonly fileSize: number) {
    // Use >>> 0 to convert the data type from signed int to unsigned int,
    // because Wasm malloc returns a signed int. This is needed if Wasm heap is
    // larger than 2GB.
    this.wasmHeapOffset = this.wasmModule._malloc(fileSize) >>> 0;
    this.wasmHeapView = this.wasmModule.HEAPU8;
    this.isReferenceValid = !!this.wasmHeapOffset;
  }

  /*
   * Get the offset of the file in Wasm heap.
   */
  get offset() {
    if (!this.isReferenceValid) {
      throw new Error('WasmFileReference has been freed.');
    }
    return this.wasmHeapOffset;
  }

  /*
   * Get the size of the file.
   */
  get size() {
    if (!this.isReferenceValid) {
      throw new Error('WasmFileReference has been freed.');
    }
    return this.fileSize;
  }

  private static async loadFromReadableStream(
    wasmModule: WasmModule,
    readableStream: ReadableStream<Uint8Array>,
    fileSize: number,
  ): Promise<WasmFileReference> {
    const fileReference = new WasmFileReference(wasmModule, fileSize);
    let offset = 0;
    const reader = readableStream.getReader();
    while (true) {
      const {value, done} =
          await (reader as ReadableStreamDefaultReader).read();
      if (done) {
        break;
      }
      fileReference.set(value, offset);
      offset += value.byteLength;
    }
    if (fileSize !== offset) {
      fileReference.free();
      throw new Error(
          `File could not be fully loaded to memory, so was not retained. ` +
          `Loaded ${offset}/${fileSize} bytes before failure`);
    }
    return fileReference;
  }

  /*
   * Creates a WasmFileReference from a URL.
   * @param wasmModule The underlying Wasm Module to use.
   * @param url The URL to request the file.
   */
  static async loadFromUrl(wasmModule: WasmModule, url: string):
      Promise<WasmFileReference> {
    const response = await fetch(url.toString());
    const fileSize = Number(response.headers.get('content-length'));
    if (!response.body) {
      throw new Error('Response body is not available.');
    }
    if (!fileSize) {
      throw new Error('File size is 0.');
    }
    return WasmFileReference.loadFromReadableStream(
      wasmModule,
      response.body,
      fileSize,
    );
  }

  /*
   * Creates a WasmFileReference from a blob.
   * @param wasmModule The underlying Wasm Module to use.
   * @param blob The file data.
   */
  static async loadFromBlob(wasmModule: WasmModule, blob: Blob):
      Promise<WasmFileReference> {
    return WasmFileReference.loadFromReadableStream(
        wasmModule,
        blob.stream(),
        blob.size,
    );
  }

  /*
   * Creates a WasmFileReference from an array.
   * @param wasmModule The underlying Wasm Module to use.
   * @param array The file data.
   */
  static loadFromArray(wasmModule: WasmModule, array: Uint8Array):
      WasmFileReference {
    const fileReference = new WasmFileReference(wasmModule, array.length);
    fileReference.set(array);
    return fileReference;
  }

  /**
   * Sets a value or an array of values into the file content.
   * @param array A typed or untyped array of values to set.
   * @param offset The index to file content at which the values are to be
   *     written.
   */
  private set(array: Uint8Array, offset?: number) {
    this.wasmHeapView.set(array, this.wasmHeapOffset + (offset ?? 0));
  }

  /**
   * Release the file's memory in Wasm heap. This must be called after the user
   * has finished using this file.
   */
  free() {
    if (!this.isReferenceValid) {
      return;
    }
    try {
      this.wasmModule._free(this.wasmHeapOffset);
    } catch {
    } finally {
      this.isReferenceValid = false;
    }
  }
}

/**
 * An implementation of GraphRunner that supports file references in the Wasm
 * heap as an input to MediaPipe graph.
 * Example usage: `const MediaPipeLib = SupportWasmFileReference(GraphRunner);`
 */
// tslint:disable-next-line:enforce-name-casing
export function SupportWasmFileReference<TBase extends LibConstructor>(
    // tslint:disable-next-line:enforce-name-casing
    Base: TBase) {
  return class extends Base {
    /*
     * Add a WasmFileReference object to an input side packet. The reference
     * must not have its `.free()` method called until the graph has finished
     * using it.
     * @param wasmFileReference The WasmFileReference object to be added.
     * @param sidePacketName The side packet name to add the object.
     */
    addWasmFileReferenceToInputSidePacket(
        wasmFileReference: WasmFileReference, sidePacketName: string): void {
      this.wrapStringPtr(sidePacketName, (sidePacketNamePtr: number) => {
        this.wasmModule._addRawDataSpanToInputSidePacket(
            wasmFileReference.offset, wasmFileReference.size,
            sidePacketNamePtr);
      });
    }

    /*
     * Add a WasmFileReference object to an input stream. The reference
     * must not have its `.free()` method called until the graph has finished
     * using it.
     * @param wasmFileReference The WasmFileReference object to be added.
     * @param streamName The stream name to add the object.
     * @param timestamp The timestamp to add the object.
     */
    addWasmFileReferenceToStream(
        wasmFileReference: WasmFileReference, streamName: string,
        timestamp: number): void {
      this.wrapStringPtr(streamName, (streamNamePtr: number) => {
        this.wasmModule._addRawDataSpanToInputStream(
            wasmFileReference.offset, wasmFileReference.size, streamNamePtr,
            timestamp);
      });
    }
  };
}
