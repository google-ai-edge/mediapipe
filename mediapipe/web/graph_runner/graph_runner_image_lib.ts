import {GraphRunner, ImageSource, SimpleListener} from './graph_runner';

/**
 * We extend from a GraphRunner constructor. This ensures our mixin has
 * access to the wasmModule, among other things. The `any` type is required for
 * mixin constructors.
 */
// tslint:disable-next-line:no-any
type LibConstructor = new (...args: any[]) => GraphRunner;

/** An image returned from a MediaPipe graph. */
export interface WasmImage {
  data: Uint8Array|Float32Array|WebGLTexture;
  width: number;
  height: number;
}

/**
 * Declarations for Emscripten's WebAssembly Module behavior, so TS compiler
 * doesn't break our JS/C++ bridge.
 */
export declare interface WasmImageModule {
  _addBoundTextureAsImageToStream:
      (streamNamePtr: number, width: number, height: number,
       timestamp: number) => void;
  _attachImageListener: (streamNamePtr: number) => void;
  _attachImageVectorListener: (streamNamePtr: number) => void;
}

/**
 * An implementation of GraphRunner that supports binding GPU image data as
 * `mediapipe::Image` instances. We implement as a proper TS mixin, to allow
 * for effective multiple inheritance. Example usage: `const GraphRunnerImageLib
 * = SupportImage(GraphRunner);`
 */
// tslint:disable-next-line:enforce-name-casing
export function SupportImage<TBase extends LibConstructor>(Base: TBase) {
  return class extends Base {
    get wasmImageModule(): WasmImageModule {
      return this.wasmModule as unknown as WasmImageModule;
    }

    /**
     * Takes the relevant information from the HTML video or image element,
     * and passes it into the WebGL-based graph for processing on the given
     * stream at the given timestamp as a MediaPipe image. Processing will not
     * occur until a blocking call (like processVideoGl or finishProcessing)
     * is made.
     * @param imageSource Reference to the video frame we wish to add into our
     *     graph.
     * @param streamName The name of the MediaPipe graph stream to add the
     *     frame to.
     * @param timestamp The timestamp of the input frame, in ms.
     */
    addGpuBufferAsImageToStream(
        imageSource: ImageSource, streamName: string, timestamp: number): void {
      this.wrapStringPtr(streamName, (streamNamePtr: number) => {
        const [width, height] =
            this.bindTextureToStream(imageSource, streamNamePtr);
        this.wasmImageModule._addBoundTextureAsImageToStream(
            streamNamePtr, width, height, timestamp);
      });
    }

    /**
     * Attaches a mediapipe:Image packet listener to the specified output
     * stream.
     * @param outputStreamName The name of the graph output stream to grab
     *     mediapipe::Image data from.
     * @param callbackFcn The function that will be called back with the data,
     *     as it is received.  Note that the data is only guaranteed to exist
     *     for the duration of the callback, and the callback will be called
     *     inline, so it should not perform overly complicated (or any async)
     *     behavior.
     */
    attachImageListener(
        outputStreamName: string,
        callbackFcn: SimpleListener<WasmImage>): void {
      // Set up our TS listener to receive any packets for this stream.
      this.setListener(outputStreamName, callbackFcn);

      // Tell our graph to listen for mediapipe::Image packets on this stream.
      this.wrapStringPtr(outputStreamName, (outputStreamNamePtr: number) => {
        this.wasmImageModule._attachImageListener(outputStreamNamePtr);
      });
    }

    /**
     * Attaches a mediapipe:Image[] packet listener to the specified
     * output_stream.
     * @param outputStreamName The name of the graph output stream to grab
     *     std::vector<mediapipe::Image> data from.
     * @param callbackFcn The function that will be called back with the data,
     *     as it is received.  Note that the data is only guaranteed to exist
     *     for the duration of the callback, and the callback will be called
     *     inline, so it should not perform overly complicated (or any async)
     *     behavior.
     */
    attachImageVectorListener(
        outputStreamName: string,
        callbackFcn: SimpleListener<WasmImage[]>): void {
      // Set up our TS listener to receive any packets for this stream.
      this.setVectorListener(outputStreamName, callbackFcn);

      // Tell our graph to listen for std::vector<mediapipe::Image> packets on
      // this stream.
      this.wrapStringPtr(outputStreamName, (outputStreamNamePtr: number) => {
        this.wasmImageModule._attachImageVectorListener(outputStreamNamePtr);
      });
    }
  };
}
