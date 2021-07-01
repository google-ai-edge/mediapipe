#include "face_mesh_lib.h"

int main(int argc, char **argv) {
  google::InitGoogleLogging(argv[0]);
  absl::ParseCommandLine(argc, argv);


  cv::VideoCapture capture;
  capture.open(0);
  if (!capture.isOpened()) {
    return -1;
  }

  constexpr char kWindowName[] = "MediaPipe";

  cv::namedWindow(kWindowName, /*flags=WINDOW_AUTOSIZE*/ 1);
#if (CV_MAJOR_VERSION >= 3) && (CV_MINOR_VERSION >= 2)
  capture.set(cv::CAP_PROP_FRAME_WIDTH, 640);
  capture.set(cv::CAP_PROP_FRAME_HEIGHT, 480);
  capture.set(cv::CAP_PROP_FPS, 30);
#endif

  LOG(INFO) << "VideoCapture initialized.";

  FaceMeshDetector *faceMeshDetector = FaceMeshDetector_Construct();

  LOG(INFO) << "FaceMeshDetector constructed.";

  LOG(INFO) << "Start grabbing and processing frames.";
  bool grab_frames = true;

  while (grab_frames) {
    // Capture opencv camera.
    cv::Mat camera_frame_raw;
    capture >> camera_frame_raw;
    if (camera_frame_raw.empty()) {
      LOG(INFO) << "Ignore empty frames from camera.";
      continue;
    }
    cv::Mat camera_frame;
    cv::cvtColor(camera_frame_raw, camera_frame, cv::COLOR_BGR2RGB);
    cv::flip(camera_frame, camera_frame, /*flipcode=HORIZONTAL*/ 1);

    FaceMeshDetector_ProcessFrame(faceMeshDetector, camera_frame);

    const int pressed_key = cv::waitKey(5);
    if (pressed_key >= 0 && pressed_key != 255)
      grab_frames = false;

    cv::imshow(kWindowName, camera_frame_raw);
  }

  LOG(INFO) << "Shutting down.";

  FaceMeshDetector_Destruct(faceMeshDetector);
}