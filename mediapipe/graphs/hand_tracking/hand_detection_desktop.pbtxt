# MediaPipe graph that performs hand detection on desktop with TensorFlow Lite
# on CPU.
# Used in the example in
# mediapipie/examples/desktop/hand_tracking:hand_detection_tflite.

# max_queue_size limits the number of packets enqueued on any input stream
# by throttling inputs to the graph. This makes the graph only process one
# frame per time.
max_queue_size: 1

# Decodes an input video file into images and a video header.
node {
  calculator: "OpenCvVideoDecoderCalculator"
  input_side_packet: "INPUT_FILE_PATH:input_video_path"
  output_stream: "VIDEO:input_video"
  output_stream: "VIDEO_PRESTREAM:input_video_header"
}

# Detects palms.
node {
  calculator: "PalmDetectionCpu"
  input_stream: "IMAGE:input_video"
  output_stream: "DETECTIONS:output_detections"
}

# Converts the detections to drawing primitives for annotation overlay.
node {
  calculator: "DetectionsToRenderDataCalculator"
  input_stream: "DETECTIONS:output_detections"
  output_stream: "RENDER_DATA:render_data"
  node_options: {
    [type.googleapis.com/mediapipe.DetectionsToRenderDataCalculatorOptions] {
      thickness: 4.0
      color { r: 0 g: 255 b: 0 }
    }
  }
}

# Draws annotations and overlays them on top of the original image coming into
# the graph.
node {
  calculator: "AnnotationOverlayCalculator"
  input_stream: "IMAGE:input_video"
  input_stream: "render_data"
  output_stream: "IMAGE:output_video"
}

# Encodes the annotated images into a video file, adopting properties specified
# in the input video header, e.g., video framerate.
node {
  calculator: "OpenCvVideoEncoderCalculator"
  input_stream: "VIDEO:output_video"
  input_stream: "VIDEO_PRESTREAM:input_video_header"
  input_side_packet: "OUTPUT_FILE_PATH:output_video_path"
  node_options: {
    [type.googleapis.com/mediapipe.OpenCvVideoEncoderCalculatorOptions]: {
      codec: "avc1"
      video_format: "mp4"
    }
  }
}
