import {
  GraphRunner,
  WasmModule,
} from '../../web/graph_runner/graph_runner';
// Placeholder for internal dependency on trusted resource url

/**
 * We extend from a GraphRunner constructor. This ensures our mixin has
 * access to the wasmModule, among other things. The `any` type is required for
 * mixin constructors.
 */
// tslint:disable-next-line:no-any
type LibConstructor = new (...args: any[]) => GraphRunner;

/**
 * Options for how data should be treated after a read.
 */
export enum ReadMode {
  // Data will be kept in memory after read.
  KEEP = 0,
  // Data will not be accessed again and can be discarded.
  DISCARD = 1,
  // All data has been used and can be discarded.
  DISCARD_ALL = 2,
}

/**
 * Holds a chunk of data that may be discarded after use. The length will stay
 * constant after creation to allow correctly computing offsets.
 */
class DiscardableDataChunk {
  readonly length: number;
  private readonly discardList: number[][] = [];
  private data: Uint8Array | undefined;

  constructor(data: Uint8Array) {
    this.data = data;
    this.length = data.length;
  }

  getData(
    offset: number,
    size: number,
    mode: ReadMode,
  ): Uint8Array | undefined {
    if (this.data === undefined) {
      return undefined;
    }
    const dataView = new Uint8Array(this.data.buffer, offset, size);
    if (mode === ReadMode.DISCARD) {
      this.checkDiscard(offset, size);
    }
    return dataView;
  }

  private checkDiscard(offset: number, size: number): void {
    this.discardList.push([offset, size]);
    this.discardList.sort((a, b) => a[0] - b[0]);

    let curOffset = 0;
    for (const [offset, size] of this.discardList) {
      if (offset <= curOffset) {
        curOffset = Math.max(curOffset, offset + size);
      }
    }
    // All data has been used, release the reference on the underlying array.
    if (curOffset === this.length) {
      this.data = undefined;
    }
  }
}

/**
 * Declarations for Emscripten's WebAssembly Module behavior, so TS compiler
 * doesn't break our JS/C++ bridge.
 */
export declare interface WasmStreamingReaderModule {
  addStreamingReaderToInputSidePacket: (
    fn: (offset: number, size: number, mode: number) => Promise<number>,
    streamNamePtr: string,
  ) => void;
}

/**
 * StreamingReader allows reading chunks of a large buffer into the wasm heap.
 * This lets the caller free the data when it's no longer needed, keeping wasm
 * heap usage low (and potentially also doing the same for JS CPU memory usage).
 * This can be useful for handling very large files efficiently.
 *
 * ArrayBuffers have a max size of 2GB in Chrome, so the data is stored as an
 * array of Uint8Array to allow holding files >2GB.
 */
export class StreamingReader {
  constructor(
    private dataArray: DiscardableDataChunk[],
    private fetchMoreData: () => Promise<Uint8Array | undefined>,
    private readonly onFinished: () => void,
  ) {}

  /*
   * Creates a StreamingReader from a ReadableStreamDefaultReader.
   * @param reader The reader
   */
  static loadFromReader(
    reader: ReadableStreamDefaultReader,
    onFinished: () => void,
  ): StreamingReader {
    const fetchMore = async () => {
      const {value, done} = await reader.read();
      if (done) {
        return undefined;
      }
      return value;
    };
    return new StreamingReader([], fetchMore, onFinished);
  }

  /*
   * Creates a StreamingReader from a URL.
   * @param url The URL to request the file.
   */
  static loadFromUrl(
    url: string,
    onFinished: () => void,
  ): StreamingReader {
    const readerPromise = fetch(url.toString()).then(
      (response) => response?.body?.getReader() as ReadableStreamDefaultReader,
    );
    const fetchMore = async () => {
      let reader: ReadableStreamDefaultReader;
      try {
        reader = await readerPromise;
      } catch (e) {
        throw new Error(`Error loading model from "${url.toString()}": ${e}`);
      }
      const {value, done} = await reader.read();
      if (done) {
        return undefined;
      }
      return value;
    };
    return new StreamingReader([], fetchMore, onFinished);
  }

  /*
   * Get the size of the data.
   */
  private get size() {
    let size = 0;
    for (let i = 0; i < this.dataArray.length; i++) {
      size += this.dataArray[i].length;
    }
    return size;
  }

  /*
   * Adds the requested data to the wasm heap.
   * @param wasmModule The module to use.
   * @param offset The offset the copy should start at.
   * @param size The data size to copy.
   */
  async addToHeap(
    wasmModule: WasmModule,
    offset: number,
    size: number,
    mode: ReadMode,
  ): Promise<number> {
    if (mode === ReadMode.DISCARD_ALL) {
      this.dataArray = [];
      this.fetchMoreData = () => Promise.resolve(undefined);
      // Signal asynchronously that we're done with the data. This ensures that
      // we finish the rest of our (synchronous from this point on)
      // initialization before we resolve the data loading promise, avoiding
      // infinite stalls in some cases.
      setTimeout(() => {
        this.onFinished();
      }, 0);
      return Promise.resolve(0);
    }

    // Fetch more data if needed.
    while (this.size < offset + size) {
      const data = await this.fetchMoreData();
      if (data === undefined) {
        break;
      }
      this.dataArray.push(new DiscardableDataChunk(data));
    }
    if (this.size < offset + size) {
      throw new Error(
        `Data size is too small: ${this.size}, expected at least ${
          offset + size
        }.`,
      );
    }

    const heapOffset = wasmModule._malloc(size) >>> 0;

    // The size that has been copied so far. The requested data may be split
    // between chunks. This keeps track of what the offset should be in the
    // heap.
    let copiedSize = 0;
    for (let i = 0; i < this.dataArray.length; i++) {
      const data = this.dataArray[i];
      if (offset >= data.length) {
        offset -= data.length;
        continue;
      }

      // The full requested size may not be in this data chunk.
      const sizeToCopy = Math.min(size, data.length - offset);
      const dataView = data.getData(offset, sizeToCopy, mode);
      if (dataView === undefined) {
        throw new Error('Data has already been released.');
      }
      wasmModule.HEAPU8.set(dataView, heapOffset + copiedSize);

      // Offset is 0 now since we've found the start of the data.
      offset = 0;
      size -= sizeToCopy;
      copiedSize += sizeToCopy;

      // If size is 0, copying is done.
      if (size === 0) {
        break;
      }
    }
    if (size !== 0) {
      throw new Error('Data not found.');
    }
    return Promise.resolve(heapOffset);
  }

  /*
   * Creates a StreamingReader from an array.
   * @param array The file data.
   */
  static loadFromArray(
    array: Uint8Array,
    onFinished: () => void,
  ): StreamingReader {
    return new StreamingReader(
      [new DiscardableDataChunk(array)],
      () => Promise.resolve(undefined),
      onFinished,
    );
  }
}

/**
 * An implementation of GraphRunner that supports streaming reads as an input to
 * MediaPipe graph.
 * Example usage: `const MediaPipeLib = SupportStreamingReader(GraphRunner);`
 */
// tslint:disable-next-line:enforce-name-casing
export function SupportStreamingReader<TBase extends LibConstructor>(
  // tslint:disable-next-line:enforce-name-casing
  Base: TBase,
) {
  return class extends Base {
    /*
     * Add a StreamingReader object to an input side packet.
     * @param streamingReader The StreamingReader object to be added.
     * @param sidePacketName The side packet name to add the object.
     */
    addStreamingReaderToInputSidePacket(
      streamingReader: StreamingReader,
      sidePacketName: string,
    ): void {
      (
        this.wasmModule as unknown as WasmStreamingReaderModule
      ).addStreamingReaderToInputSidePacket(
        (offset: number, size: number, mode: number) => {
          return streamingReader.addToHeap(
            this.wasmModule,
            offset,
            size,
            mode as ReadMode,
          );
        },
        sidePacketName,
      );
    }
  };
}
