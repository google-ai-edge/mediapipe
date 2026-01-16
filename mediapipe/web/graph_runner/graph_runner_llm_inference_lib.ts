import {streamToUint8Array} from '../../tasks/web/genai/llm_inference/model_loading_utils';
import {LlmInferenceGraphOptions} from '../../tasks/web/genai/llm_inference/proto/llm_inference_graph_options_pb';
import {GraphRunner} from '../../web/graph_runner/graph_runner';
import {
  ReadMode,
  StreamingReader,
} from '../../web/graph_runner/graph_runner_streaming_reader';
import {SamplerParameters} from '../../tasks/cc/genai/inference/proto/sampler_params_pb';

const DEFAULT_MAX_TOKENS = 512;
const DEFAULT_TOP_K = 40;
const DEFAULT_FORCE_F32 = false;
const DEFAULT_MAX_NUM_IMAGES = 0;
const DEFAULT_SUPPORT_AUDIO = false;

const TOKENS_PER_IMAGE = 260;

/**
 * Image type for use in multi-modal LLM queries.
 */
export declare interface Image {
  imageSource: ImageSource;
}

/**
 * Type check for Image.
 */
export function instanceOfImage(obj: unknown): obj is Image {
  // tslint:disable-next-line:ban-unsafe-reflection
  return typeof obj === 'object' && obj != null && 'imageSource' in obj;
}

/**
 * Audio type for use in multi-modal LLM queries.
 */
export declare interface Audio {
  audioSource: AudioSource;
}

/**
 * Type check for Audio.
 */
export function instanceOfAudio(obj: unknown): obj is Audio {
  // tslint:disable-next-line:ban-unsafe-reflection
  return typeof obj === 'object' && obj != null && 'audioSource' in obj;
}

function instanceOfAudioChunk(obj: unknown): obj is AudioChunk {
  // tslint:disable-next-line:ban-unsafe-reflection
  return (
    typeof obj === 'object' &&
    obj != null &&
    'audioSamples' in obj &&
    'audioSampleRateHz' in obj
  );
}

/**
 * Type for a piece of an LLM query.
 */
export type PromptPart = string | Image | Audio;

/**
 * Type for an LLM query; may be multi-modal.
 */
export type Prompt = PromptPart | PromptPart[];

// The allowable types for image sources to be used for mixed audio/text/vision
// LLM queries.
declare type ImageSource = Exclude<CanvasImageSource, SVGElement> | string;

declare interface ImageByteSource {
  image: Exclude<CanvasImageSource, SVGElement>;
  width: number;
  height: number;
}

declare interface AudioChunk {
  audioSampleRateHz: number;
  audioSamples: Float32Array;
}

// The allowable types for audio sources to be used for mixed audio/text/vision
// LLM queries. For now, we only support URLs for mono-channel audio files and
// single-channel AudioBuffer objects. We should expand this in the future.
declare type AudioSource = string | AudioBuffer | AudioChunk;

/**
 * We extend from a GraphRunner constructor. This ensures our mixin has
 * access to the wasmModule, among other things. The `any` type is required for
 * mixin constructors.
 */
// tslint:disable-next-line:no-any
type LibConstructor = new (...args: any[]) => GraphRunner;

/**
 * A listener that receives the newly generated partial result and an indication
 * whether the generation is complete.
 */
export type ProgressListener = (
  partialResult: string,
  done: boolean,
) => unknown;

/**
 * A listener that receives the newly generated partial results for multiple
 * responses and an indication whether the generation is complete.
 */
export type MultiResponseProgressListener = (
  partialResult: string[],
  done: boolean,
) => unknown;

/**
 * Declarations for Emscripten's WebAssembly Module behavior, so TS compiler
 * doesn't break our JS/C++ bridge.
 */
export declare interface WasmLlmInferenceModule {
  // TODO: b/398949555 - Support multi-response generation for converted LLM
  // models (.task format).
  _userProgressListener: ProgressListener | undefined;
  _GetSizeInTokens: (textPtr: number) => number;
  _FreeSession: (session: number) => void;
  _AddTextQueryChunk: (session: number, textPtr: number) => void;
  _AddImageQueryChunk: (
    session: number,
    imageDataPtr: number,
    width: number,
    height: number,
  ) => void;
  _AddAudioQueryChunk: (
    session: number,
    audioSampleRateHz: number,
    audioDataPtr: number,
    audioDataSize: number,
  ) => void;
  ccall: (
    name: string,
    type: string,
    inParams: unknown,
    outParams: unknown,
    options: unknown,
  ) => Promise<void>;
  ccallNum: (
    name: string,
    type: string,
    inParams: unknown,
    outParams: unknown,
    options: unknown,
  ) => Promise<number>;
  createLlmInferenceEngine: (
    maxTokens: number,
    topK: number,
    forceF32: boolean,
    maxNumImages: number,
    useAudio: boolean,
    readBufferCallback: (offset: number, size: number, mode: number) => void,
  ) => Promise<void>;
}

/**
 * An implementation of GraphRunner that provides LLM Inference Engine
 * functionality.
 * Example usage:
 * `const MediaPipeLib = SupportLlmInference(GraphRunner);`
 */
// tslint:disable-next-line:enforce-name-casing
export function SupportLlmInference<TBase extends LibConstructor>(Base: TBase) {
  return class LlmInferenceSupportedGraphRunner extends Base {
    // Some methods and properties are supposed to be private, TS has an
    // existing issue with supporting private in anonymous classes, so we use
    // old-style '_' to indicate private.
    // tslint:disable-next-line:enforce-name-casing
    _isLlmEngineProcessing = false;
    // tslint:disable-next-line:enforce-name-casing
    _audioKeepaliveSession = 0;
    // tslint:disable-next-line:enforce-name-casing
    _visionKeepaliveSession = 0;

    // tslint:disable-next-line:enforce-name-casing
    _startLlmEngineProcessing() {
      if (this._isLlmEngineProcessing) {
        throw new Error(
          'Cannot process because LLM inference engine is currently loading ' +
            'or processing.',
        );
      }
      this._isLlmEngineProcessing = true;
    }

    // tslint:disable-next-line:enforce-name-casing
    _endLlmEngineProcessing() {
      this._isLlmEngineProcessing = false;
    }

    /**
     * Create LLM Inference Engine for text generation using streaming loading.
     * This version cannot support converted models, but can support large ones,
     * via streaming loading, as well as advanced features, like multimodality.
     *
     * @param modelStream The stream object for the model to be loaded.
     * @param llmInferenceGraphOptions The settings for the LLM Inference
     *     Engine.
     */
    async createLlmInferenceEngine(
      modelStream: ReadableStreamDefaultReader,
      llmInferenceGraphOptions: LlmInferenceGraphOptions,
    ) {
      this._startLlmEngineProcessing();
      try {
        const streamingReader = StreamingReader.loadFromReader(
          modelStream,
          () => {}, // onFinished callback, if we need it
        );
        const readBufferCallback = (
          offset: number,
          size: number,
          mode: number,
        ) => {
          return streamingReader.addToHeap(
            this.wasmModule,
            offset,
            size,
            mode as ReadMode,
          );
        };
        await (
          this.wasmModule as unknown as WasmLlmInferenceModule
        ).createLlmInferenceEngine(
          llmInferenceGraphOptions.getMaxTokens() ?? DEFAULT_MAX_TOKENS,
          llmInferenceGraphOptions.getSamplerParams()?.getK() ?? DEFAULT_TOP_K,
          llmInferenceGraphOptions.getForceF32() ?? DEFAULT_FORCE_F32,
          llmInferenceGraphOptions.getMaxNumImages() ?? DEFAULT_MAX_NUM_IMAGES,
          llmInferenceGraphOptions.getSupportAudio() ?? DEFAULT_SUPPORT_AUDIO,
          readBufferCallback,
        );
      } finally {
        this._endLlmEngineProcessing();
      }
    }

    /**
     * Create LLM Inference Engine for text generation. This version can be used
     * with AI Edge converted models, but as such it does not support streaming
     * loading, nor some of the advanced features like multimodality.
     *
     * @param modelStream The stream object for the model to be loaded.
     * @param llmInferenceGraphOptions The settings for the LLM Inference
     *    Engine.
     */
    async createLlmInferenceEngineConverted(
      modelStream: ReadableStreamDefaultReader,
      llmInferenceGraphOptions: LlmInferenceGraphOptions,
    ) {
      this._startLlmEngineProcessing();
      try {
        await this.uploadToWasmFileSystem(modelStream, 'llm.task');
        // TODO: b/398858545 - Pass llmInferenceGraphOptions to the C function.
        await (this.wasmModule as unknown as WasmLlmInferenceModule).ccall(
          'CreateLlmInferenceEngineConverted',
          'void',
          ['number', 'number', 'boolean'],
          [
            llmInferenceGraphOptions.getMaxTokens() ?? DEFAULT_MAX_TOKENS,
            llmInferenceGraphOptions.getSamplerParams()?.getK() ??
              DEFAULT_TOP_K,
            llmInferenceGraphOptions.getForceF32() ?? DEFAULT_FORCE_F32,
          ],
          {async: true},
        );
      } finally {
        this._endLlmEngineProcessing();
      }
    }

    /**
     * Delete LLM Inference Engine.
     */
    deleteLlmInferenceEngine() {
      this._startLlmEngineProcessing();
      try {
        const llmWasm = this.wasmModule as unknown as WasmLlmInferenceModule;
        llmWasm.ccall('DeleteLlmInferenceEngine', 'void', [], [], {
          async: false,
        });
        if (this._audioKeepaliveSession) {
          llmWasm._FreeSession(this._audioKeepaliveSession);
          // Don't double-free
          if (this._visionKeepaliveSession === this._audioKeepaliveSession) {
            this._visionKeepaliveSession = 0;
          }
          this._audioKeepaliveSession = 0;
        }
        if (this._visionKeepaliveSession) {
          llmWasm._FreeSession(this._visionKeepaliveSession);
          this._visionKeepaliveSession = 0;
        }
      } finally {
        this._endLlmEngineProcessing();
      }
    }

    /**
     * Process a query using LLM Inference Engine for text generation.
     *
     * @param query The prompt to process.
     * @param samplerParameters The settings for sampler, used to configure the
     *     LLM Inference sessions.
     * @return The generated text result.
     */
    async generateResponse(
      query: PromptPart[],
      samplerParameters: SamplerParameters,
      // TODO: b/398949555 - Support multi-response generation for converted LLM
      // models (.task format).
      userProgressListener?: ProgressListener,
    ): Promise<string> {
      this._startLlmEngineProcessing();
      try {
        const result: string[] = [];
        // This additional wrapper on top of userProgressListener is to collect
        // all partial results and then to return them together as the full
        // result.
        const progressListener = (partialResult: string, done: boolean) => {
          if (partialResult) {
            // TODO: b/398904237 - Support streaming generation: use the done flag
            // to indicate the end of the generation.
            result.push(partialResult);
          }
          if (userProgressListener) {
            userProgressListener(partialResult, done);
          }
        };
        const llmWasm = this.wasmModule as unknown as WasmLlmInferenceModule;
        llmWasm._userProgressListener = progressListener;
        // Sampler params
        // OSS build does not support SamplerParameters.serializeBinary(...).
        // tslint:disable-next-line:deprecation
        const samplerParamsBin = samplerParameters.serializeBinary();
        const samplerParamsSize = samplerParamsBin.length;
        const samplerParamsPtr = this.wasmModule._malloc(samplerParamsSize);
        this.wasmModule.HEAPU8.set(samplerParamsBin, samplerParamsPtr);

        // For running a query: we first create our session.
        const useAudio = query.some(instanceOfAudio);
        const useVision = query.some(instanceOfImage);
        llmWasm.ccallNum = llmWasm.ccall as unknown as (
          name: string,
          type: string,
          inParams: unknown,
          outParams: unknown,
          options: unknown,
        ) => Promise<number>;
        const session = await llmWasm.ccallNum(
          'MakeSessionForPredict',
          'number',
          ['number', 'number', 'boolean', 'boolean'],
          [samplerParamsPtr, samplerParamsSize, useVision, useAudio],
          {
            async: true,
          },
        );

        const mediaToFree: number[] = [];

        // Then we add all the query chunks in order
        for (const chunk of query) {
          if (typeof chunk === 'string') {
            // Add text chunk to query
            this.wrapStringPtr(chunk, (textPtr: number) => {
              llmWasm._AddTextQueryChunk(session, textPtr);
            });
          } else if (instanceOfImage(chunk)) {
            // We load image data from whatever source type is given.
            const {image, width, height} = await this.getImageFromSource(
              chunk.imageSource,
            );

            // Now we extract the bytes. TODO: b/424221732 - This should also be
            // made more efficient in the future, ideally by keeping on the GPU.
            const canvas =
              typeof OffscreenCanvas !== 'undefined'
                ? new OffscreenCanvas(width, height)
                : document.createElement('canvas');
            canvas.width = width;
            canvas.height = height;
            const ctx = canvas.getContext('2d') as
              | CanvasRenderingContext2D
              | OffscreenCanvasRenderingContext2D;
            ctx.drawImage(image, 0, 0);
            const imageData = ctx.getImageData(0, 0, width, height);

            // Next we copy the bytes into C++ heap.
            const imageDataPtr = this.wasmModule._malloc(
              imageData.width * imageData.height * 4,
            );
            this.wasmModule.HEAPU8.set(imageData.data, imageDataPtr);

            // Add image chunk to query
            llmWasm._AddImageQueryChunk(
              session,
              imageDataPtr,
              imageData.width,
              imageData.height,
            );
            mediaToFree.push(imageDataPtr);
          } else if (instanceOfAudio(chunk)) {
            const audioSource = await this.getAudioFromSource(
              chunk.audioSource,
            );
            // We use C++ API for passing a raw audio chunk. Can be used for
            // generic audio streams, like microphone data. But it will error
            // out currently if the segment is too small.
            const audioDataPtr = this.wasmModule._malloc(
              audioSource.audioSamples.byteLength,
            );
            this.wasmModule.HEAPF32.set(
              audioSource.audioSamples,
              audioDataPtr / 4,
            );
            llmWasm._AddAudioQueryChunk(
              session,
              audioSource.audioSampleRateHz,
              audioDataPtr,
              audioSource.audioSamples.length,
            );
            mediaToFree.push(audioDataPtr);
          } else {
            throw new Error('Unsupported PromptPart type in query.');
          }
        }

        // And finally we asynchronously run the request processing
        await llmWasm.ccall('PredictSession', 'void', ['number'], [session], {
          async: true,
        });

        // Vision and audio runners are lazily loaded during the first query
        // that uses them, and are freed automatically afterwards. We cannot
        // afford to reload them though, so we intentionally leave our first
        // vision and/or audio session(s) alive and unfreed.
        // TODO: b/431095215 - Fix this.
        let freeSession = true;
        if (useVision && this._visionKeepaliveSession === 0) {
          this._visionKeepaliveSession = session;
          freeSession = false;
        }
        if (useAudio && this._audioKeepaliveSession === 0) {
          this._audioKeepaliveSession = session;
          freeSession = false;
        }
        if (freeSession) {
          llmWasm._FreeSession(session);
        }

        // After our query has finished, we free all image memory we were
        // holding.
        for (const mediaDataPtr of mediaToFree) {
          this.wasmModule._free(mediaDataPtr);
        }
        mediaToFree.length = 0;

        // TODO: b/399215600 - Remove the following trigger of the user progress
        // listener when the underlying LLM Inference Engine is fixed to trigger
        // it at the end of the generation.
        if (userProgressListener) {
          userProgressListener(/* partialResult= */ '', /* done= */ true);
        }
        this.wasmModule._free(samplerParamsPtr);
        llmWasm._userProgressListener = undefined;
        // TODO: b/398880215 - return the generated string from the C function.
        return result.join('');
      } finally {
        this._endLlmEngineProcessing();
      }
    }

    /**
     * Runs an invocation of *only* the tokenization for the LLM, and returns
     * the size (in tokens) of the result. Runs synchronously.
     *
     * @param query The prompt to tokenize.
     * @return The number of tokens in the resulting tokenization of the text.
     */
    sizeInTokens(query: PromptPart[]): number {
      this._startLlmEngineProcessing();
      // First we chop out all image pieces. Since there's only one supported
      // vision model, which uses a fixed number of tokens per image, we simply
      // add that to our count manually. But once the engine supports
      // GetSizeInTokens for non-text modalities, this should be fixed.
      // TODO: b/426691212 - Remove this workaround once that functionality
      // exists.
      let tokensFromImages = 0;
      let promptWithoutImages = '';
      for (const chunk of query) {
        // For now, just text and images to handle.
        if (typeof chunk === 'string') {
          promptWithoutImages += chunk;
        } else if (instanceOfImage(chunk)) {
          tokensFromImages += TOKENS_PER_IMAGE;
        } else if (instanceOfAudio(chunk)) {
          // TODO: Add tokens from audio to count when possible.
          console.warn(
            'sizeInTokens is not yet implemented for audio; audio tokens will not be counted',
          );
        }
      }
      try {
        let result: number;
        this.wrapStringPtr(promptWithoutImages, (textPtr: number) => {
          result = (
            this.wasmModule as unknown as WasmLlmInferenceModule
          )._GetSizeInTokens(textPtr);
        });
        return tokensFromImages + result!;
      } finally {
        this._endLlmEngineProcessing();
      }
    }

    /**
     * Upload the LLM asset to the wasm file system.
     *
     * @param modelStream The stream object for the model to be uploaded.
     * @param filename The name in the file system where the model will be put.
     */
    async uploadToWasmFileSystem(
      modelStream: ReadableStreamDefaultReader,
      filename: string,
    ): Promise<void> {
      const fileContent = await streamToUint8Array(modelStream);
      try {
        // Try to delete file as we cannot overwrite an existing file
        // using our current API.
        this.wasmModule.FS_unlink(filename);
      } catch {}
      this.wasmModule.FS_createDataFile(
        '/',
        filename,
        fileContent,
        /* canRead= */ true,
        /* canWrite= */ false,
        /* canOwn= */ false,
      );
    }

    /**
     * Parse an image source into an Image, along with its width and height, for
     * extracting the raw bytes.
     *
     * @param image The image source.
     * @return The image, ready to be drawn, along with its width and height.
     */
    async getImageFromSource(image: ImageSource): Promise<ImageByteSource> {
      if (typeof image === 'string') {
        // Load image from the given url. Note that we currently must wait for
        // our image to load so we can pass its bytes into the session API in
        // order to preserve chunk ordering. This could be made more efficient
        // in the future.
        const imageElement = new Image();
        imageElement.src = image;
        imageElement.crossOrigin = 'Anonymous';

        // TODO: b/424014728 - Wrap below in try/catch block so we can make
        // failures more user-friendly.
        try {
          await imageElement.decode();
        } catch {
          throw new Error(`Image from URL ${image} failed to load`);
        }
        return {
          image: imageElement,
          width: imageElement.naturalWidth,
          height: imageElement.naturalHeight,
        };
      } else if (image instanceof HTMLImageElement) {
        try {
          await image.decode();
        } catch {
          throw new Error(`Image from HTMLImageElement failed to load`);
        }
        return {image, width: image.naturalWidth, height: image.naturalHeight};
      } else if (image instanceof HTMLVideoElement) {
        return {image, width: image.videoWidth, height: image.videoHeight};
      } else if (image instanceof VideoFrame) {
        return {image, width: image.displayWidth, height: image.displayHeight};
      } else {
        return {image, width: image.width, height: image.height};
      }
    }

    /**
     * Parse an audio source into an Audio, along with its sample rate, for
     * extracting the raw bytes.
     *
     * @param audio The audio source.
     * @return The audio, ready to be ingested by the LLM, along with its sample
     *   rate.
     */
    async getAudioFromSource(audio: AudioSource): Promise<AudioChunk> {
      if (typeof audio === 'string') {
        const response = await fetch(audio);
        if (!response.ok) {
          throw new Error(
            `Audio fetch for ${audio} had error: ${response.status}`,
          );
        }
        const arrayBuffer = await response.arrayBuffer();
        const audioContext = new AudioContext({sampleRate: 16000});
        const audioBuffer = await audioContext.decodeAudioData(arrayBuffer);
        return {
          audioSamples: audioBuffer.getChannelData(0),
          audioSampleRateHz: audioBuffer.sampleRate,
        };
      } else if (instanceOfAudioChunk(audio)) {
        return audio;
      } else {
        return {
          audioSamples: audio.getChannelData(0),
          audioSampleRateHz: audio.sampleRate,
        };
      }
    }
  };
}
