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
#include <windows.h>

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
  static constexpr auto kLandmarksNum = 468;

  MPFaceMeshDetector(int numFaces, const char *face_detection_model_path,
                     const char *face_landmark_model_path);

  void ProcessFrame2D(const cv::Mat &camera_frame, int *numFaces,
                      cv::Point2f **multi_face_landmarks);

private:
  absl::Status InitFaceMeshDetector(int numFaces,
                                    const char *face_detection_model_path,
                                    const char *face_landmark_model_path);
  absl::Status ProcessFrame2DWithStatus(const cv::Mat &camera_frame,
                                        int *numFaces,
                                        cv::Point2f **multi_face_landmarks);

  static constexpr auto kInputStream = "input_video";
  static constexpr auto kOutputStream_landmarks = "multi_face_landmarks";
  static constexpr auto kOutputStream_faceCount = "face_count";

  static const std::string graphConfig;

  mediapipe::CalculatorGraph graph;

  std::unique_ptr<mediapipe::OutputStreamPoller> landmarks_poller_ptr;
  std::unique_ptr<mediapipe::OutputStreamPoller> face_count_poller_ptr;
};

#ifdef __cplusplus
extern "C" {
#endif

DLLEXPORT MPFaceMeshDetector *
MPFaceMeshDetectorConstruct(int numFaces, const char *face_detection_model_path,
                            const char *face_landmark_model_path);

DLLEXPORT void MPFaceMeshDetectorDestruct(MPFaceMeshDetector *detector);

DLLEXPORT void
MPFaceMeshDetectorProcessFrame2D(MPFaceMeshDetector *detector,
                                 const cv::Mat &camera_frame, int *numFaces,
                                 cv::Point2f **multi_face_landmarks);

DLLEXPORT extern const int MPFaceMeshDetectorLandmarksNum;

#ifdef __cplusplus
};
#endif
#endif