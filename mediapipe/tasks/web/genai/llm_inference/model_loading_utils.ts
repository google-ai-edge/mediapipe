/**
 * Copyright 2025 The MediaPipe Authors.
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

/**
 * Turn a ReadableStreamDefaultReader into two streams that are identical.
 *
 * This implementation is the same as the native implementation where when one stream is read, the
 * data is cached so the other can read it as well. However, the key difference
 * is that when one stream is cancelled, it stops caching data.
 */
export function tee<T>(
  mainStream: ReadableStreamDefaultReader<T>,
): [ReadableStreamDefaultReader<T>, ReadableStreamDefaultReader<T>] {
  interface StreamCache {
    cache: T[];
    active: boolean;
  }
  const cache1: StreamCache = {cache: [], active: true};
  const cache2: StreamCache = {cache: [], active: true};
  let readLock: Promise<void> = Promise.resolve();

  function makeTeedStream(myCache: StreamCache, otherCache: StreamCache) {
    return new ReadableStream<T>({
      start(controller) {},
      async pull(controller) {
        readLock = readLock.then(async () => {
          if (myCache.cache.length > 0) {
            controller.enqueue(myCache.cache.shift()!);
            return;
          }
          const {value, done} = await mainStream.read();
          if (value) {
            if (otherCache.active) {
              otherCache.cache.push(value);
            }
            if (myCache.active) {
              // cancel() may be called at any time, so we need to check if this
              // stream is still active before enqueuing.
              controller.enqueue(value);
            }
          }
          if (done) {
            controller.close();
          }
        });
        await readLock;
      },
      cancel() {
        myCache.active = false;
        myCache.cache.length = 0;
        if (!otherCache.active) {
          mainStream.cancel();
        }
      },
    });
  }

  const stream1 = makeTeedStream(cache1, cache2);
  const stream2 = makeTeedStream(cache2, cache1);
  return [stream1.getReader(), stream2.getReader()];
}

/**
 * Reads a number of bytes from a stream.
 *
 * If the stream ends before the number of bytes is read, an error is thrown.
 *
 * This closes the stream when it finishes reading because otherwise, the stream
 * may be left in an unknown state as the last chunk may not have been fully
 * read (meaning you have no idea where the next chunk in the stream starts).
 *
 * @param stream The stream to read from.
 * @param bytes The number of bytes to read.
 * @return A Uint8Array containing the bytes read from the stream.
 */
async function readBytesAndClose(
  stream: ReadableStreamDefaultReader<Uint8Array>,
  bytes: number,
): Promise<Uint8Array> {
  const result = new Uint8Array(bytes);
  let bytesRead = 0;
  while (bytesRead < bytes) {
    const {value, done} = await stream.read();
    if (value) {
      const shortenedToMaxLength = value.subarray(0, bytes - bytesRead);
      result.set(shortenedToMaxLength, bytesRead);
      bytesRead += shortenedToMaxLength.length;
    }
    if (done) {
      throw new Error(
        `Expected ${bytes} bytes, but stream ended after reading ${bytesRead} bytes.`,
      );
    }
  }
  // A possible improvement would be to return the all the extra bytes in the
  // last chunk instead of canceling the stream.
  await stream.cancel();
  return result;
}

/** Converts a ReadableStreamDefaultReader to a Uint8Array. */
export async function streamToUint8Array(
  reader: ReadableStreamDefaultReader<Uint8Array>,
): Promise<Uint8Array> {
  const chunks: Uint8Array[] = [];
  let totalLength = 0;

  while (true) {
    const {done, value} = await reader.read();
    if (done) {
      break;
    }
    chunks.push(value);
    totalLength += value.length;
  }

  if (chunks.length === 0) {
    return new Uint8Array(0);
  } else if (chunks.length === 1) {
    return chunks[0];
  } else {
    // Merge chunks
    const combined = new Uint8Array(totalLength);
    let offset = 0;
    for (const chunk of chunks) {
      combined.set(chunk, offset);
      offset += chunk.length;
    }
    return combined;
  }
}

/**
 * The format of the model.
 */
export enum ModelFormat {
  HANDWRITTEN = 0,
  CONVERTED = 1,
  LITERTLM = 2,
}

type FormatTester = (
  model: ReadableStreamDefaultReader<Uint8Array>,
) => Promise<boolean>;

const FORMAT_TESTERS: Array<[ModelFormat, FormatTester]> = [
  [
    ModelFormat.HANDWRITTEN,
    async (model: ReadableStreamDefaultReader<Uint8Array>) => {
      const tag = 'TFL3';
      const tagDataLength = new TextEncoder().encode(tag).length;
      const offset = 4;

      const bytes = await readBytesAndClose(model, tagDataLength + offset);
      const bytesString = new TextDecoder('utf-8').decode(
        bytes.subarray(offset, tagDataLength + offset),
      );
      return bytesString === tag;
    },
  ],
  [
    ModelFormat.CONVERTED,
    async (model: ReadableStreamDefaultReader<Uint8Array>) => {
      // Converted models are always zipped, so they have the zip file header.
      const bytes = await readBytesAndClose(model, 6);
      return bytes[4] === 0x50 && bytes[5] === 0x4b;
    },
  ],
  [
    ModelFormat.LITERTLM,
    async (model: ReadableStreamDefaultReader<Uint8Array>) => {
      // Check for .litertlm header
      const bytes = await readBytesAndClose(model, 8);
      const bytesString = new TextDecoder('utf-8').decode(bytes);
      return bytesString === 'LITERTLM';
    },
  ],
];

/**
 * Wraps a ReadableStreamDefaultReader with some extra logic to allow for both
 * normal `read()` behavior as well as reading a specified number of bytes,
 * precisely "fast-forwarding" into the stream via `readBytes(numBytes)`.
 */
class FastForwardStreamReader implements ReadableStreamDefaultReader {
  private extraBytes: Uint8Array | undefined = undefined;
  readonly closed;
  constructor(private readonly stream: ReadableStreamDefaultReader) {
    this.closed = this.stream.closed;
  }
  async read() {
    if (this.extraBytes) {
      // We need to copy extraBytes when returning it, or else it gets freed.
      const toReturn = this.extraBytes.slice();
      this.extraBytes = undefined;
      return {value: toReturn, done: false};
    }
    return this.stream.read();
  }
  async readBytes(bytes: number) {
    const result = new Uint8Array(bytes);
    let bytesRead = 0;

    // First copy bytes from extraBytes if we have any.
    if (this.extraBytes) {
      const bytesToCopy = Math.min(bytes, this.extraBytes.length);
      result.set(this.extraBytes.subarray(0, bytesToCopy), 0);
      bytesRead += bytesToCopy;

      if (bytesRead < this.extraBytes.length) {
        // Truncate extraBytes if we have any left over.
        this.extraBytes = this.extraBytes.subarray(bytesRead);
      } else {
        // Remove entirely otherwise.
        this.extraBytes = undefined;
      }
    }

    // Read until we've reached (and likely passed) the target byte count.
    while (bytesRead < bytes) {
      const {value, done} = await this.stream.read();
      if (value) {
        const shortenedToMaxLength = value.subarray(0, bytes - bytesRead);
        result.set(shortenedToMaxLength, bytesRead);

        // If our final read went over, then keep only the extra bytes.
        if (value.length > shortenedToMaxLength.length) {
          this.extraBytes = value.subarray(bytes - bytesRead);
        }

        bytesRead += shortenedToMaxLength.length;
      }
      if (done) {
        throw new Error(
          `Expected ${bytes} bytes, but stream ended after reading ${bytesRead} bytes.`,
        );
      }
    }
    return result;
  }

  // We want to match the ReadableStreamDefaultReader contract, which uses `any`
  // tslint:disable-next-line:no-any
  cancel(reason?: any) {
    return this.stream.cancel(reason);
  }

  releaseLock() {
    return this.stream.releaseLock();
  }
}

/**
 * Takes a .litertlm file stream and produces a web-formatted .bin file stream.
 *
 * Takes a stream containing the contents of a .litertlm file and fast-forwards
 * it to a web-formatted .bin file. Throws an error if the .litertlm file does
 * not contain a web-formatted .bin model (denoted by model_type =
 * tf_lite_artisan_text_decoder in the .litertlm section metadata).
 * The resulting stream should be usable as if a .bin file had been streamed
 * instead.
 */
export async function fastForwardLitertLmStreamToModel(
  modelStream: ReadableStreamDefaultReader<Uint8Array>,
  getModelOffsetFromHeader: (header: Uint8Array) => number,
): Promise<ReadableStreamDefaultReader<Uint8Array>> {
  const ffStream = new FastForwardStreamReader(modelStream);

  // Read litertlm header chunk.
  const bytes = await ffStream.readBytes(32);
  // Parse last 8 bytes (24-31) as uint64 to find offset for end of header.
  const view = new DataView(bytes.buffer);
  const headerEndOffset = Number(view.getBigUint64(24, true));
  // Read entire header as bytes.
  const header = await ffStream.readBytes(headerEndOffset - 32);
  // Make wasm call to take flatbuffer header and parse out the gpu_artisan .bin
  // offset, erroring out if that does not exist.
  const modelOffset = getModelOffsetFromHeader(header);
  if (modelOffset < 0) {
    throw new Error(
      '.litertlm file could not be read or did not contain a web-formatted LLM',
    );
  }
  // Now we skip ahead in our stream until the start of our model.
  await ffStream.readBytes(modelOffset - headerEndOffset);
  // The stream should now behave as a normal .bin file stream. There may be
  // additional unused information in the stream following the .bin file, but
  // that will not affect loading.
  return ffStream;
}

/**
 * Determines the format of the model stream.
 *
 * Usage: Tee the model stream using the tee function from this file and pass
 * one of the streams to this function. Do not use the native browser tee
 * function as it will cache the entire model in memory.
 */
export async function getModelFormatAndClose(
  modelStream: ReadableStreamDefaultReader<Uint8Array>,
): Promise<ModelFormat> {
  const matchedFormats: ModelFormat[] = [];

  let testStream: ReadableStreamDefaultReader<Uint8Array>;
  for (const [format, tester] of FORMAT_TESTERS) {
    [modelStream, testStream] = tee(modelStream);
    const matched = await tester(testStream);
    await testStream.cancel();
    if (matched) {
      matchedFormats.push(format);
    }
  }
  await modelStream.cancel();

  if (matchedFormats.length === 0) {
    throw new Error('No model format matched.');
  } else if (matchedFormats.length === 1) {
    return matchedFormats[0];
  } else {
    throw new Error(`Multiple model formats matched: ${matchedFormats}`);
  }
}

/**
 * Turns the input data into a stream.
 */
export function uint8ArrayToStream(
  data: Uint8Array,
): ReadableStream<Uint8Array> {
  return new ReadableStream<Uint8Array>({
    start(controller) {},
    async pull(controller) {
      controller.enqueue(data);
      controller.close();
    },
  });
}
