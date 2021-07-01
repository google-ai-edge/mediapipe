#include <windows.h>

#include "face_mesh_lib.h"

#define DEBUG

FaceMeshDetector::FaceMeshDetector() {
  const auto status = InitFaceMeshDetector();
  if (!status.ok()) {
    LOG(INFO) << "Failed constructing FaceMeshDetector.";
  }
}

absl::Status FaceMeshDetector::InitFaceMeshDetector() {
  LOG(INFO) << "Get calculator graph config contents: " << graphConfig;

  mediapipe::CalculatorGraphConfig config =
      mediapipe::ParseTextProtoOrDie<mediapipe::CalculatorGraphConfig>(
          graphConfig);

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

  return absl::Status();
}

absl::Status FaceMeshDetector::ProcessFrameWithStatus(cv::Mat &camera_frame) {
  // Wrap Mat into an ImageFrame.
  auto input_frame = absl::make_unique<mediapipe::ImageFrame>(
      mediapipe::ImageFormat::SRGB, camera_frame.cols, camera_frame.rows,
      mediapipe::ImageFrame::kDefaultAlignmentBoundary);
  cv::Mat input_frame_mat = mediapipe::formats::MatView(input_frame.get());
  camera_frame.copyTo(input_frame_mat);

  // Send image packet into the graph.

  size_t frame_timestamp_us =
      (double)cv::getTickCount() / (double)cv::getTickFrequency() * 1e6;
  MP_RETURN_IF_ERROR(graph.AddPacketToInputStream(
      kInputStream, mediapipe::Adopt(input_frame.release())
                        .At(mediapipe::Timestamp(frame_timestamp_us))));
  LOG(INFO) << "Pushed new frame.";

#ifdef DEBUG
  LOG(INFO) << "Pushed new frame.";
#endif
  mediapipe::Packet face_count_packet;
  if (!face_count_poller_ptr ||
      !face_count_poller_ptr->Next(&face_count_packet)) {
    LOG(INFO) << "Failed during getting next face_count_packet.";

    return absl::Status();
  }
  auto &face_count = face_count_packet.Get<int>();

#ifdef DEBUG
  LOG(INFO) << "Got face_count: " << face_count;
#endif

  if (!face_count) {
    return absl::Status();
  }

  mediapipe::Packet face_landmarks_packet;
  if (!landmarks_poller_ptr ||
      !landmarks_poller_ptr->Next(&face_landmarks_packet)) {
    LOG(INFO) << "Failed during getting next landmarks_packet.";

    return absl::Status();
  }

  auto &output_landmarks_vector =
      face_landmarks_packet
          .Get<::std::vector<::mediapipe::NormalizedLandmarkList>>();

  auto &output_landmarks = output_landmarks_vector[0];

#ifdef DEBUG
  LOG(INFO) << "Got landmarks_packet: " << output_landmarks.landmark_size();
#endif

  auto &landmark = output_landmarks.landmark(0);
#ifdef DEBUG
  LOG(INFO) << "First landmark: x - " << landmark.x() << ", y - "
            << landmark.y() << ", z - " << landmark.z();
#endif

  return absl::Status();
}

std::vector<cv::Point2f> *
FaceMeshDetector::ProcessFrame(cv::Mat &camera_frame) {
  ProcessFrameWithStatus(camera_frame);

  return new std::vector<cv::Point2f>();
}

extern "C" {
DLLEXPORT FaceMeshDetector *FaceMeshDetector_Construct() {
  return new FaceMeshDetector();
}

DLLEXPORT void FaceMeshDetector_Destruct(FaceMeshDetector *detector) {
  delete detector;
}

DLLEXPORT void *FaceMeshDetector_ProcessFrame(FaceMeshDetector *detector,
                                              cv::Mat &camera_frame) {
  return reinterpret_cast<void *>(detector->ProcessFrame(camera_frame));
}
}

const char FaceMeshDetector::kInputStream[] = "input_video";
const char FaceMeshDetector::kOutputStream_landmarks[] = "multi_face_landmarks";
const char FaceMeshDetector::kOutputStream_faceCount[] = "face_count";

const std::string FaceMeshDetector::graphConfig = R"pb(
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
  input_stream: "FINISHED:multi_face_landmarks"
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
      packet { int_value: 1 }
    }
  }
}

# Subgraph that detects faces and corresponding landmarks.
node {
  calculator: "FaceLandmarkFrontCpuWithFaceCounter"
  input_stream: "IMAGE:throttled_input_video"
  input_side_packet: "NUM_FACES:num_faces"
  output_stream: "LANDMARKS:multi_face_landmarks"
  output_stream: "ROIS_FROM_LANDMARKS:face_rects_from_landmarks"
  output_stream: "DETECTIONS:face_detections"
  output_stream: "ROIS_FROM_DETECTIONS:face_rects_from_detections"
  output_stream: "FACE_COUNT_FROM_LANDMARKS:face_count"
}

)pb";
