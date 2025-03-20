import {EmptyPacketListener, ErrorListener, SimpleListener, VectorListener} from './listener_types';

/**
 * This file can serve as a common interface for most MediaPipe-based TypeScript
 * libraries. Additionally, it can hook automatically into `wasm_mediapipe_demo`
 * to allow for easy migrations from old pure JS demos. See `mediapipe_ts_demo`
 * and `wasm_mediapipe_files` BUILD rules. Public API is described below in more
 * detail, public API for GraphRunner factory functions is described in
 * "graph_runner_factory_api.d.ts", and the same for listener callbacks
 * can be found in "listenertypes.d.ts". An actual implementation is coded in
 * `graph_runner.ts`, and that can be built against directly. The purpose of
 * this file is primarily to enforce a common interface and provide better
 * documentation on the design of the API as a whole.
 */

// We re-export all of our imported public listener types, so that users can
// import everything they need for the public interfaces directly from here.
export {
  EmptyPacketListener,
  ErrorListener,
  SimpleListener,
  VectorListener,
};

/**
 * Valid types of image sources which we can run our GraphRunner over.
 *
 * @deprecated Use TexImageSource instead.
 */
export type ImageSource = TexImageSource;

/**
 * Simple interface for a class to run an arbitrary MediaPipe graph on web, and
 * either render results into canvas, or else stream output into attached
 * listeners. Takes a WebAssembly Module and an optional canvas for rendering.
 * Standard implementation is `GraphRunner`.
 * There are three categories of functions:
 *   - Those which add inputs into the graph (named `add*ToStream` or
 *     `add*ToInputSidePacket`).
 *   - Those which listen for outputs from the graph (named `attach*Listener`).
 *   - Those concerned with the graph running itself (runner settings, loading
 *     the graph, destruction, and `finishProcessing`).
 * The expected ordering of one-time initialization calls should be:
 *   1. Construction
 *   2. Adding any input side packets and attaching any output listeners
 *   3. Initializing/setting the graph
 * After this, for the main loop, the user would add packets to streams for a
 * given frame and then call `finishProcessing()`.
 * Example usage pattern would be:
 *   ```
 *   // We assume we have already constructed a GraphRunner called graphRunner.
 *   // Generally we would do this via a helpful factory method like
 *   // `createGraphRunner` or some variant thereof.
 *   const graphRunner: GraphRunner;
 *
 *   // Initialization code:
 *   graphRunner.addBoolToInputSidePacket(true, 'some_input_bool_side_packet');
 *   graphRunner.attachStringVectorListener('some_output_string_vec_stream',
 *       (data: string[], timestamp: number) => { console.log(data); });
 *   await graphRunner.initializeGraph('path/to/graph_file.pbtxt');
 *
 *   // Main loop code (run every frame):
 *   const frameTimestamp = performance.now();
 *   graphRunner.addStringToStream(
 *       'Hello World', 'some_input_string_stream', frameTimestamp);
 *   graphRunner.addFloatToStream(
 *       3.7, 'some_input_float_stream', frameTimestamp);
 *   graphRunner.finishProcessing();
 *
 *   // When done (no need to call if the user is just expected to close the
 *   // browser tab):
 *   graphRunner.closeGraph();
 *   ```
 */
export interface GraphRunnerApi {
  /**
   * Fetches a MediaPipe graph from a URL string pointing to the graph file.
   * Will then set the graph using the results, replacing the previously running
   * MediaPipe graph, if there is one. If the graph file's name ends with
   * ".pbtxt" with ".textproto", it will assume text-formatted, and otherwise
   * will assume a binary format for the proto.
   * @param graphFile The url of the MediaPipe graph file to load.
   */
  initializeGraph(graphFile: string): Promise<void>;

  /**
   * Convenience helper for loading a MediaPipe graph from a string representing
   * a text proto config. Useful for graph files which are expected to be edited
   * locally, web-side. Will replace the previously running MediaPipe graph,
   * if there is one.
   * @param graphConfig The text proto graph config, expected to be a string in
   * default JavaScript UTF-16 format.
   */
  setGraphFromString(graphConfig: string): void;

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
  setGraph(graphData: Uint8Array, isBinary: boolean): void;

  /**
   * Configures the current graph to handle audio processing in a certain way
   * for all its audio input streams. Additionally can configure audio headers
   * (both input side packets as well as input stream headers), but these
   * configurations only take effect if called before the graph is set/started.
   * @param numChannels The number of channels of audio input. Only 1
   *     is supported for now.
   * @param numSamples The number of samples that are taken in each
   *     audio capture. Setting this to null will allow for variable-length
   *     audio input.
   * @param sampleRate The rate, in Hz, of the sampling.
   * @param streamName The optional name of the input stream to additionally
   *     configure with audio information. This configuration only occurs before
   *     the graph is set/started. If unset, a default stream name will be used.
   * @param headerName The optional name of the header input side packet to
   *     additionally configure with audio information. This configuration only
   *     occurs before the graph is set/started. If unset, a default header name
   *     will be used.
   */
  configureAudio(
      numChannels: number, numSamples: number, sampleRate: number,
      streamName?: string, headerName?: string): void;

  /**
   * Allows disabling automatic canvas resizing, in case clients want to control
   * control this. By default, the canvas will be resized to the size of the
   * last GPU image input (see `addGpuBufferToStream`).
   * @param resize True will re-enable automatic canvas resizing, while false
   *     will disable the feature.
   */
  setAutoResizeCanvas(resize: boolean): void;

  /**
   * Allows disabling the automatic render-to-screen code, in case clients don't
   * need/want this. In particular, this removes the requirement for pipelines
   * to have access to GPU resources, as well as the requirement for graphs to
   * have an "output_frames_gpu" stream defined, so pure CPU pipelines and
   * non-video pipelines can be created.
   * NOTE: This only affects future graph initializations (via `setGraph`` or
   *     `initializeGraph``), and does NOT affect the currently running graph,
   *     so calls to this should be made *before* `setGraph`/`initializeGraph`
   *     for the graph file being targeted.
   * @param enabled True will re-enable automatic render-to-screen code and
   *     cause GPU resources to once again be requested, while false will
   *     disable the feature.
   */
  setAutoRenderToScreen(enabled: boolean): void;

  /**
   * Overrides the vertical orientation for input GpuBuffers and the automatic
   * render-to-screen code.  The default for our OpenGL code on other platforms
   * (Android, Linux) is to use a bottom-left origin.  But the default for WebGL
   * is to use a top-left origin. We use WebGL default normally, and many
   * calculators and graphs have platform-specific code to handle the resulting
   * orientation flip. However, in order to be able to use a single graph on all
   * platforms without alterations, it may be useful to send images into a web
   * graph using the OpenGL orientation. Users can call this function with
   * `bottomLeftIsOrigin = true` in order to enforce an orientation for all
   * GpuBuffer inputs which is consistent with OpenGL on other platforms.
   * This call will also vertically flip the automatic render-to-screen code as
   * well, so that webcam input (for example) will render properly when passed
   * through the graph still.
   * NOTE: This will immediately affect GpuBuffer inputs, but must be called
   * *before* graph start in order to affect the automatic render-to-screen
   * code!
   * @param bottomLeftIsOrigin True will flip our input GpuBuffers and auto
   * render-to-screen to match the classic OpenGL orientation, while false will
   * disable this feature to match the default WebGL orientation.
   */
  setGpuBufferVerticalFlip(bottomLeftIsOrigin: boolean): void;

  /**
   * Attaches a listener that will be invoked when the MediaPipe web framework
   * returns an error.
   */
  attachErrorListener(callbackFcn: ErrorListener): void;

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
      outputStreamName: string, callbackFcn: EmptyPacketListener): void;

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
      audioData: Float32Array, streamName: string, timestamp: number): void;

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
      streamName: string, timestamp: number): void;

  /**
   * Takes the relevant information from the HTML video or image element, and
   * passes it into the WebGL-based graph for processing on the given stream at
   * the given timestamp. Can be used for additional auxiliary GpuBuffer input
   * streams. Like all `add*ToStream` calls, processing will not occur until a
   * blocking call like `finishProcessing` or the deprecated `processGl` is
   * made. For use with 'gl_graph_runner_internal_multi_input'.
   * @param imageSource Reference to the video frame we wish to add into our
   *     graph.
   * @param streamName The name of the MediaPipe graph stream to add the frame
   *     to.
   * @param timestamp The timestamp of the input frame, in ms.
   */
  addGpuBufferToStream(
      imageSource: TexImageSource, streamName: string, timestamp: number): void;

  /**
   * Sends a boolean packet into the specified stream at the given timestamp.
   * @param data The boolean data to send.
   * @param streamName The name of the graph input stream to send data into.
   * @param timestamp The timestamp of the input data, in ms.
   */
  addBoolToStream(data: boolean, streamName: string, timestamp: number): void;

  /**
   * Sends a double packet into the specified stream at the given timestamp.
   * @param data The double data to send.
   * @param streamName The name of the graph input stream to send data into.
   * @param timestamp The timestamp of the input data, in ms.
   */
  addDoubleToStream(data: number, streamName: string, timestamp: number): void;

  /**
   * Sends a float packet into the specified stream at the given timestamp.
   * @param data The float data to send.
   * @param streamName The name of the graph input stream to send data into.
   * @param timestamp The timestamp of the input data, in ms.
   */
  addFloatToStream(data: number, streamName: string, timestamp: number): void;

  /**
   * Sends an integer packet into the specified stream at the given timestamp.
   * @param data The integer data to send.
   * @param streamName The name of the graph input stream to send data into.
   * @param timestamp The timestamp of the input data, in ms.
   */
  addIntToStream(data: number, streamName: string, timestamp: number): void;

  /**
   * Sends an unsigned integer packet into the specified stream at the given
   * timestamp.
   * @param data The unsigned integer data to send.
   * @param streamName The name of the graph input stream to send data into.
   * @param timestamp The timestamp of the input data, in ms.
   */
  addUintToStream(data: number, streamName: string, timestamp: number): void;

  /**
   * Sends a string packet into the specified stream at the given timestamp.
   * @param data The string data to send.
   * @param streamName The name of the graph input stream to send data into.
   * @param timestamp The timestamp of the input data, in ms.
   */
  addStringToStream(data: string, streamName: string, timestamp: number): void;

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
      timestamp: number): void;

  /**
   * Sends a serialized protobuffer packet into the specified stream at the
   *     given timestamp, to be parsed into the specified protobuffer type.
   * @param data The binary (serialized) raw protobuffer data.
   * @param protoType The C++ namespaced type this protobuffer data corresponds
   *     to (e.g. "foo.Bar"). It will be converted to this type when output as a
   *     packet into the graph.
   * @param streamName The name of the graph input stream to send data into.
   * @param timestamp The timestamp of the input data, in ms.
   */
  addProtoToStream(
      data: Uint8Array, protoType: string, streamName: string,
      timestamp: number): void;

  /**
   * Sends an empty packet into the specified stream at the given timestamp,
   *     effectively advancing that input stream's timestamp bounds without
   *     sending additional data packets.
   * @param streamName The name of the graph input stream to send the empty
   *     packet into.
   * @param timestamp The timestamp of the empty packet, in ms.
   */
  addEmptyPacketToStream(streamName: string, timestamp: number): void;

  /**
   * Sends a vector<bool> packet into the specified stream at the given
   * timestamp.
   * @param data The ordered array of boolean data to send as a vector.
   * @param streamName The name of the graph input stream to send data into.
   * @param timestamp The timestamp of the input data, in ms.
   */
  addBoolVectorToStream(data: boolean[], streamName: string, timestamp: number):
      void;

  /**
   * Sends a vector<double> packet into the specified stream at the given
   * timestamp.
   * @param data The ordered array of double-precision float data to send as a
   *     vector.
   * @param streamName The name of the graph input stream to send data into.
   * @param timestamp The timestamp of the input data, in ms.
   */
  addDoubleVectorToStream(
      data: number[], streamName: string, timestamp: number): void;

  /**
   * Sends a vector<float> packet into the specified stream at the given
   * timestamp.
   * @param data The ordered array of float data to send as a vector.
   * @param streamName The name of the graph input stream to send data into.
   * @param timestamp The timestamp of the input data, in ms.
   */
  addFloatVectorToStream(data: number[], streamName: string, timestamp: number):
      void;

  /**
   * Sends a vector<int> packet into the specified stream at the given
   * timestamp.
   * @param data The ordered array of integer data to send as a vector.
   * @param streamName The name of the graph input stream to send data into.
   * @param timestamp The timestamp of the input data, in ms.
   */
  addIntVectorToStream(data: number[], streamName: string, timestamp: number):
      void;

  /**
   * Sends a vector<uint32_t> packet into the specified stream at the given
   * timestamp.
   * @param data The ordered array of unsigned integer data to send as a vector.
   * @param streamName The name of the graph input stream to send data into.
   * @param timestamp The timestamp of the input data, in ms.
   */
  addUintVectorToStream(data: number[], streamName: string, timestamp: number):
      void;

  /**
   * Sends a vector<string> packet into the specified stream at the given
   * timestamp.
   * @param data The ordered array of string data to send as a vector.
   * @param streamName The name of the graph input stream to send data into.
   * @param timestamp The timestamp of the input data, in ms.
   */
  addStringVectorToStream(
      data: string[], streamName: string, timestamp: number): void;

  /**
   * Attaches a boolean packet to the specified input_side_packet.
   * @param data The boolean data to send.
   * @param sidePacketName The name of the graph input side packet to send data
   *     into.
   */
  addBoolToInputSidePacket(data: boolean, sidePacketName: string): void;

  /**
   * Attaches a double packet to the specified input_side_packet.
   * @param data The double data to send.
   * @param sidePacketName The name of the graph input side packet to send data
   *     into.
   */
  addDoubleToInputSidePacket(data: number, sidePacketName: string): void;

  /**
   * Attaches a float packet to the specified input_side_packet.
   * @param data The float data to send.
   * @param sidePacketName The name of the graph input side packet to send data
   *     into.
   */
  addFloatToInputSidePacket(data: number, sidePacketName: string): void;

  /**
   * Attaches a integer packet to the specified input_side_packet.
   * @param data The integer data to send.
   * @param sidePacketName The name of the graph input side packet to send data
   *     into.
   */
  addIntToInputSidePacket(data: number, sidePacketName: string): void;

  /**
   * Attaches a unsigned integer packet to the specified input_side_packet.
   * @param data The unsigned integer data to send.
   * @param sidePacketName The name of the graph input side packet to send data
   *     into.
   */
  addUintToInputSidePacket(data: number, sidePacketName: string): void;

  /**
   * Attaches a string packet to the specified input_side_packet.
   * @param data The string data to send.
   * @param sidePacketName The name of the graph input side packet to send data
   *     into.
   */
  addStringToInputSidePacket(data: string, sidePacketName: string): void;

  /**
   * Attaches a serialized proto packet to the specified input_side_packet.
   * @param data The binary (serialized) raw protobuffer data.
   * @param protoType The C++ namespaced type this protobuffer data corresponds
   *     to. It will be converted to this type for use in the graph.
   * @param sidePacketName The name of the graph input side packet to send data
   *     into.
   */
  addProtoToInputSidePacket(
      data: Uint8Array, protoType: string, sidePacketName: string): void;

  /**
   * Attaches a vector<bool> packet to the specified input_side_packet.
   * @param data The ordered array of boolean data to send as a vector.
   * @param sidePacketName The name of the graph input side packet to send data
   *     into.
   */
  addBoolVectorToInputSidePacket(data: boolean[], sidePacketName: string): void;

  /**
   * Attaches a vector<double> packet to the specified input_side_packet.
   * @param data The ordered array of double-precision float data to send as a
   *     vector.
   * @param sidePacketName The name of the graph input side packet to send data
   *     into.
   */
  addDoubleVectorToInputSidePacket(data: number[], sidePacketName: string):
      void;

  /**
   * Attaches a vector<float> packet to the specified input_side_packet.
   * @param data The ordered array of float data to send as a vector.
   * @param sidePacketName The name of the graph input side packet to send data
   *     into.
   */
  addFloatVectorToInputSidePacket(data: number[], sidePacketName: string): void;

  /**
   * Attaches a vector<int> packet to the specified input_side_packet.
   * @param data The ordered array of integer data to send as a vector.
   * @param sidePacketName The name of the graph input side packet to send data
   *     into.
   */
  addIntVectorToInputSidePacket(data: number[], sidePacketName: string): void;

  /**
   * Attaches a vector<uint32_t> packet to the specified input_side_packet.
   * @param data The ordered array of unsigned integer data to send as a vector.
   * @param sidePacketName The name of the graph input side packet to send data
   *     into.
   */
  addUintVectorToInputSidePacket(data: number[], sidePacketName: string): void;

  /**
   * Attaches a vector<string> packet to the specified input_side_packet.
   * @param data The ordered array of string data to send as a vector.
   * @param sidePacketName The name of the graph input side packet to send data
   *     into.
   */
  addStringVectorToInputSidePacket(data: string[], sidePacketName: string):
      void;

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
      outputStreamName: string, callbackFcn: SimpleListener<boolean>): void;

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
      outputStreamName: string, callbackFcn: SimpleListener<boolean[]>): void;

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
      outputStreamName: string, callbackFcn: SimpleListener<number>): void;

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
      outputStreamName: string, callbackFcn: SimpleListener<number[]>): void;

  /**
   * Attaches an unsigned int packet listener to the specified output_stream.
   * @param outputStreamName The name of the graph output stream to grab
   *     unsigned int data from.
   * @param callbackFcn The function that will be called back with the data, as
   *     it is received.  Note that the data is only guaranteed to exist for the
   *     duration of the callback, and the callback will be called inline, so it
   *     should not perform overly complicated (or any async) behavior.
   */
  attachUintListener(
      outputStreamName: string, callbackFcn: SimpleListener<number>): void;

  /**
   * Attaches an uint[] packet listener to the specified output_stream.
   * @param outputStreamName The name of the graph output stream to grab
   *     std::vector<uint32_t> data from.
   * @param callbackFcn The function that will be called back with the data, as
   *     it is received.  Note that the data is only guaranteed to exist for the
   *     duration of the callback, and the callback will be called inline, so it
   *     should not perform overly complicated (or any async) behavior.
   */
  attachUintVectorListener(
      outputStreamName: string, callbackFcn: SimpleListener<number[]>): void;

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
      outputStreamName: string, callbackFcn: SimpleListener<number>): void;

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
      outputStreamName: string, callbackFcn: SimpleListener<number[]>): void;

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
      outputStreamName: string, callbackFcn: SimpleListener<number>): void;

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
      outputStreamName: string, callbackFcn: SimpleListener<number[]>): void;

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
      outputStreamName: string, callbackFcn: SimpleListener<string>): void;

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
      outputStreamName: string, callbackFcn: SimpleListener<string[]>): void;

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
      makeDeepCopy?: boolean): void;

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
      makeDeepCopy?: boolean): void;

  /**
   * Attaches an audio packet listener to the specified output_stream, to be
   * given a Float32Array as output. Requires wasm build dependency
   * "gl_graph_runner_audio_out".
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
      makeDeepCopy?: boolean): void;

  /**
   * Forces all queued-up packets to be pushed through the MediaPipe graph as
   * far as possible, performing all processing until no more processing can be
   * done. This is fully synchronous and by default single-threaded, so the
   * calling thread will be blocked until processing completes. This must be
   * called once for every frame, as `add*ToStream` calls merely queue up
   * packets to the appropriate streams for processing, but processing does not
   * occur until this function is called. Any listeners receiving output will
   * be called back before this operation completes.
   */
  finishProcessing(): void;

  /**
   * Closes the input streams and all calculators for this graph and frees up
   * any C++ resources. The graph will not be usable once closed!
   */
  closeGraph(): void;
}
