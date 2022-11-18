import {ImageSource, GraphRunner} from './graph_runner';

/**
 * We extend from a GraphRunner constructor. This ensures our mixin has
 * access to the wasmModule, among other things. The `any` type is required for
 * mixin constructors.
 */
// tslint:disable-next-line:no-any
type LibConstructor = new (...args: any[]) => GraphRunner;

/**
 * Declarations for Emscripten's WebAssembly Module behavior, so TS compiler
 * doesn't break our JS/C++ bridge.
 */
export declare interface WasmImageModule {
  _addBoundTextureAsImageToStream:
      (streamNamePtr: number, width: number, height: number,
       timestamp: number) => void;
}

/**
 * An implementation of GraphRunner that supports binding GPU image data as
 * `mediapipe::Image` instances. We implement as a proper TS mixin, to allow for
 * effective multiple inheritance. Example usage:
 * `const WasmMediaPipeImageLib = SupportImage(GraphRunner);`
 */
// tslint:disable-next-line:enforce-name-casing
export function SupportImage<TBase extends LibConstructor>(Base: TBase) {
  return class extends Base {
    /**
     * Takes the relevant information from the HTML video or image element, and
     * passes it into the WebGL-based graph for processing on the given stream
     * at the given timestamp as a MediaPipe image. Processing will not occur
     * until a blocking call (like processVideoGl or finishProcessing) is made.
     * @param imageSource Reference to the video frame we wish to add into our
     *     graph.
     * @param streamName The name of the MediaPipe graph stream to add the frame
     *     to.
     * @param timestamp The timestamp of the input frame, in ms.
     */
    addGpuBufferAsImageToStream(
        imageSource: ImageSource, streamName: string, timestamp: number): void {
      this.wrapStringPtr(streamName, (streamNamePtr: number) => {
        const [width, height] =
            this.bindTextureToStream(imageSource, streamNamePtr);
        (this.wasmModule as unknown as WasmImageModule)
            ._addBoundTextureAsImageToStream(
                streamNamePtr, width, height, timestamp);
      });
    }
  };
}
