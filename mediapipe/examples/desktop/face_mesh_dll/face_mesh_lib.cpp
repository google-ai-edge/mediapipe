#include "face_mesh_lib.h"

MPFaceMeshDetector::MPFaceMeshDetector(int numFaces,
                                       const char *face_detection_model_path,
                                       const char *face_landmark_model_path) {
  const auto status = InitFaceMeshDetector(numFaces, face_detection_model_path,
                                           face_landmark_model_path);
  if (!status.ok()) {
    LOG(INFO) << "Failed constructing FaceMeshDetector.";
    LOG(INFO) << status.message();
  }
}

absl::Status
MPFaceMeshDetector::InitFaceMeshDetector(int numFaces,
                                         const char *face_detection_model_path,
                                         const char *face_landmark_model_path) {
  numFaces = std::max(numFaces, 1);

  if (face_detection_model_path == nullptr) {
    face_detection_model_path =
        "mediapipe/modules/face_detection/face_detection_short_range.tflite";
  }

  if (face_landmark_model_path == nullptr) {
    face_landmark_model_path =
        "mediapipe/modules/face_landmark/face_landmark.tflite";
  }

  // Prepare graph config.
  auto preparedGraphConfig = absl::StrReplaceAll(
      graphConfig, {{"$numFaces", std::to_string(numFaces)}});
  preparedGraphConfig = absl::StrReplaceAll(
      preparedGraphConfig,
      {{"$faceDetectionModelPath", face_detection_model_path}});
  preparedGraphConfig = absl::StrReplaceAll(
      preparedGraphConfig,
      {{"$faceLandmarkModelPath", face_landmark_model_path}});

  LOG(INFO) << "Get calculator graph config contents: " << preparedGraphConfig;

  mediapipe::CalculatorGraphConfig config =
      mediapipe::ParseTextProtoOrDie<mediapipe::CalculatorGraphConfig>(
          preparedGraphConfig);
  LOG(INFO) << "Initialize the calculator graph.";

  MP_RETURN_IF_ERROR(graph.Initialize(config));

  LOG(INFO) << "Start running the calculator graph.";

  ASSIGN_OR_RETURN(mediapipe::OutputStreamPoller landmarks_poller,
                   graph.AddOutputStreamPoller(kOutputStream_landmarks));
  ASSIGN_OR_RETURN(mediapipe::OutputStreamPoller face_count_poller,
                   graph.AddOutputStreamPoller(kOutputStream_faceCount));

  landmarks_poller_ptr = std::make_unique<mediapipe::OutputStreamPoller>(
      std::move(landmarks_poller));
  face_count_poller_ptr = std::make_unique<mediapipe::OutputStreamPoller>(
      std::move(face_count_poller));

  MP_RETURN_IF_ERROR(graph.StartRun({}));

  LOG(INFO) << "MPFaceMeshDetector constructed successfully.";

  return absl::OkStatus();
}

absl::Status MPFaceMeshDetector::ProcessFrame2DWithStatus(
    const cv::Mat &camera_frame, int *numFaces,
    cv::Point2f **multi_face_landmarks) {
  *numFaces = 0;

  // Wrap Mat into an ImageFrame.
  auto input_frame = absl::make_unique<mediapipe::ImageFrame>(
      mediapipe::ImageFormat::SRGB, camera_frame.cols, camera_frame.rows,
      mediapipe::ImageFrame::kDefaultAlignmentBoundary);
  cv::Mat input_frame_mat = mediapipe::formats::MatView(input_frame.get());
  camera_frame.copyTo(input_frame_mat);

  // Send image packet into the graph.
  size_t frame_timestamp_us = static_cast<double>(cv::getTickCount()) /
                              static_cast<double>(cv::getTickFrequency()) * 1e6;
  MP_RETURN_IF_ERROR(graph.AddPacketToInputStream(
      kInputStream, mediapipe::Adopt(input_frame.release())
                        .At(mediapipe::Timestamp(frame_timestamp_us))));

  // Get face count.
  mediapipe::Packet face_count_packet;
  if (!face_count_poller_ptr ||
      !face_count_poller_ptr->Next(&face_count_packet)) {
    return absl::CancelledError(
        "Failed during getting next face_count_packet.");
  }

  auto &face_count = face_count_packet.Get<int>();

  if (face_count <= 0) {
    return absl::OkStatus();
  }

  // Get face landmarks.
  mediapipe::Packet face_landmarks_packet;
  if (!landmarks_poller_ptr ||
      !landmarks_poller_ptr->Next(&face_landmarks_packet)) {
    return absl::CancelledError("Failed during getting next landmarks_packet.");
  }

  auto &output_landmarks_vector =
      face_landmarks_packet
          .Get<::std::vector<::mediapipe::NormalizedLandmarkList>>();

  // Convert landmarks to cv::Point2f**.
  for (int i = 0; i < face_count; ++i) {
    const auto &normalizedLandmarkList = output_landmarks_vector[i];
    const auto landmarks_num = normalizedLandmarkList.landmark_size();

    if (landmarks_num != kLandmarksNum) {
      return absl::CancelledError("Detected unexpected landmarks number.");
    }

    auto &face_landmarks = multi_face_landmarks[i];

    for (int j = 0; j < landmarks_num; ++j) {
      const auto &landmark = normalizedLandmarkList.landmark(j);
      face_landmarks[j].x = landmark.x();
      face_landmarks[j].y = landmark.y();
    }
  }

  *numFaces = face_count;

  return absl::OkStatus();
}

void MPFaceMeshDetector::ProcessFrame2D(const cv::Mat &camera_frame,
                                        int *numFaces,
                                        cv::Point2f **multi_face_landmarks) {
  const auto status =
      ProcessFrame2DWithStatus(camera_frame, numFaces, multi_face_landmarks);
  if (!status.ok()) {
    LOG(INFO) << "Failed ProcessFrame2D.";
    LOG(INFO) << status.message();
  }
}

extern "C" {
DLLEXPORT MPFaceMeshDetector *
MPFaceMeshDetectorConstruct(int numFaces, const char *face_detection_model_path,
                            const char *face_landmark_model_path) {
  return new MPFaceMeshDetector(numFaces, face_detection_model_path,
                                face_landmark_model_path);
}

DLLEXPORT void MPFaceMeshDetectorDestruct(MPFaceMeshDetector *detector) {
  delete detector;
}

DLLEXPORT void
MPFaceMeshDetectorProcessFrame2D(MPFaceMeshDetector *detector,
                                 const cv::Mat &camera_frame, int *numFaces,
                                 cv::Point2f **multi_face_landmarks) {
  detector->ProcessFrame2D(camera_frame, numFaces, multi_face_landmarks);
}

DLLEXPORT const int MPFaceMeshDetectorLandmarksNum = MPFaceMeshDetector::kLandmarksNum;
}

const std::string MPFaceMeshDetector::graphConfig = R"pb(
# MediaPipe graph that performs face mesh with TensorFlow Lite on CPU.

# Input image. (ImageFrame)
input_stream: "input_video"

# Collection of detected/processed faces, each represented as a list of
# landmarks. (std::vector<NormalizedLandmarkList>)
output_stream: "multi_face_landmarks"

# Detected faces count. (int)
output_stream: "face_count"

node {
  calculator: "FlowLimiterCalculator"
  input_stream: "input_video"
  input_stream: "FINISHED:face_count"
  input_stream_info: {
    tag_index: "FINISHED"
    back_edge: true
  }
  output_stream: "throttled_input_video"
}

# Defines side packets for further use in the graph.
node {
  calculator: "ConstantSidePacketCalculator"
  output_side_packet: "PACKET:num_faces"
  node_options: {
    [type.googleapis.com/mediapipe.ConstantSidePacketCalculatorOptions]: {
      packet { int_value: $numFaces }
    }
  }
}

# Defines side packets for further use in the graph.
node {
    calculator: "ConstantSidePacketCalculator"
    output_side_packet: "PACKET:face_detection_model_path"
    options: {
        [mediapipe.ConstantSidePacketCalculatorOptions.ext]: {
            packet { string_value: "$faceDetectionModelPath" }
        }
    }
}

# Defines side packets for further use in the graph.
node {
    calculator: "ConstantSidePacketCalculator"
    output_side_packet: "PACKET:face_landmark_model_path"
    node_options: {
        [type.googleapis.com/mediapipe.ConstantSidePacketCalculatorOptions]: {
            packet { string_value: "$faceLandmarkModelPath" }
    }
  }
}

node {
    calculator: "LocalFileContentsCalculator"
    input_side_packet: "FILE_PATH:0:face_detection_model_path"
    input_side_packet: "FILE_PATH:1:face_landmark_model_path"
    output_side_packet: "CONTENTS:0:face_detection_model_blob"
    output_side_packet: "CONTENTS:1:face_landmark_model_blob"
}

node {
    calculator: "TfLiteModelCalculator"
    input_side_packet: "MODEL_BLOB:face_detection_model_blob"
    output_side_packet: "MODEL:face_detection_model"
}
node {
    calculator: "TfLiteModelCalculator"
    input_side_packet: "MODEL_BLOB:face_landmark_model_blob"
    output_side_packet: "MODEL:face_landmark_model"
}


# Subgraph that detects faces and corresponding landmarks.
node {
  calculator: "FaceLandmarkFrontSideModelCpuWithFaceCounter"
  input_stream: "IMAGE:throttled_input_video"
  input_side_packet: "NUM_FACES:num_faces"
  input_side_packet: "MODEL:0:face_detection_model"
  input_side_packet: "MODEL:1:face_landmark_model"
  output_stream: "LANDMARKS:multi_face_landmarks"
  output_stream: "ROIS_FROM_LANDMARKS:face_rects_from_landmarks"
  output_stream: "DETECTIONS:face_detections"
  output_stream: "ROIS_FROM_DETECTIONS:face_rects_from_detections"
  output_stream: "FACE_COUNT_FROM_LANDMARKS:face_count"
}

)pb";
