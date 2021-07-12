#ifndef FACE_MESH_LIBRARY_H
#define FACE_MESH_LIBRARY_H

#ifdef COMPILING_DLL
#define DLLEXPORT __declspec(dllexport)
#else
#define DLLEXPORT __declspec(dllimport)
#endif

#include <cstdlib>
#include <memory>
#include <string>

#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "absl/strings/str_replace.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/calculator_graph.h"
#include "mediapipe/framework/formats/image_frame.h"
#include "mediapipe/framework/formats/image_frame_opencv.h"
#include "mediapipe/framework/formats/landmark.pb.h"
#include "mediapipe/framework/output_stream_poller.h"
#include "mediapipe/framework/port/file_helpers.h"
#include "mediapipe/framework/port/opencv_highgui_inc.h"
#include "mediapipe/framework/port/opencv_imgproc_inc.h"
#include "mediapipe/framework/port/opencv_video_inc.h"
#include "mediapipe/framework/port/parse_text_proto.h"
#include "mediapipe/framework/port/status.h"

class MPFaceMeshDetector {
public:
  MPFaceMeshDetector(int numFaces, const char *face_detection_model_path,
                     const char *face_landmark_model_path);
  int GetFaceCount(const cv::Mat &camera_frame);
  void GetFaceLandmarks(cv::Point2f **multi_face_landmarks);

private:
  absl::Status InitFaceMeshDetector(int numFaces,
                                    const char *face_detection_model_path,
                                    const char *face_landmark_model_path);
  absl::Status ProcessFrameWithStatus(
      const cv::Mat &camera_frame,
      std::vector<std::vector<cv::Point2f>> &multi_face_landmarks);
  absl::Status GetFaceCountWithStatus(const cv::Mat &camera_frame);
  absl::Status GetFaceLandmarksWithStatus(cv::Point2f **multi_face_landmarks);

  static const char kInputStream[];
  static const char kOutputStream_landmarks[];
  static const char kOutputStream_faceCount[];

  static const std::string graphConfig;

  mediapipe::CalculatorGraph graph;

  std::unique_ptr<mediapipe::OutputStreamPoller> landmarks_poller_ptr;
  std::unique_ptr<mediapipe::OutputStreamPoller> face_count_poller_ptr;

  int faceCount = -1;
};

#ifdef __cplusplus
extern "C" {
#endif

DLLEXPORT MPFaceMeshDetector *FaceMeshDetector_Construct(
    int numFaces = 1,
    const char *face_detection_model_path =
        "mediapipe/modules/face_detection/face_detection_short_range.tflite",
    const char *face_landmark_model_path =
        "mediapipe/modules/face_landmark/face_landmark.tflite");


DLLEXPORT void FaceMeshDetector_Destruct(MPFaceMeshDetector *detector);

DLLEXPORT int FaceMeshDetector_GetFaceCount(MPFaceMeshDetector *detector,
                                            const cv::Mat &camera_frame);
DLLEXPORT void
FaceMeshDetector_GetFaceLandmarks(MPFaceMeshDetector *detector,
                                  cv::Point2f **multi_face_landmarks);

#ifdef __cplusplus
};
#endif
#endif