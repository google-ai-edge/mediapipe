/**
   Copyright 2022, Nimagna AG

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
 */
 
#include "pose_tracking.h"

#include <cstdlib>
#include <string>

#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/formats/image_frame.h"
#include "mediapipe/framework/formats/image_frame_opencv.h"
#include "mediapipe/framework/formats/landmark.pb.h"
#include "mediapipe/framework/port/file_helpers.h"
#include "mediapipe/framework/port/opencv_highgui_inc.h"
#include "mediapipe/framework/port/opencv_imgproc_inc.h"
#include "mediapipe/framework/port/opencv_video_inc.h"
#include "mediapipe/framework/port/parse_text_proto.h"
#include "mediapipe/framework/port/status.h"

class PoseTrackingImpl {
 public:
  PoseTrackingImpl(const std::string& calculatorGraphConfigFile) {
    auto status = initialize(calculatorGraphConfigFile);
	LOG(WARNING) << "Initialized PoseTracking with status: " << status;
  }

  absl::Status initialize(const std::string& calculatorGraphConfigFile) {
    std::string graphContents;
    MP_RETURN_IF_ERROR(mediapipe::file::GetContents(calculatorGraphConfigFile, &graphContents));

    mediapipe::CalculatorGraphConfig config =
        mediapipe::ParseTextProtoOrDie<mediapipe::CalculatorGraphConfig>(graphContents);

    MP_RETURN_IF_ERROR(graph.Initialize(config));
    ASSIGN_OR_RETURN(mediapipe::OutputStreamPoller poller,
                     graph.AddOutputStreamPoller(kOutputSegmentationStream, true));

    ASSIGN_OR_RETURN(mediapipe::OutputStreamPoller landmarksPoller,
                     graph.AddOutputStreamPoller(kOutpuLandmarksStream, true));


    maskPollerPtr = std::make_unique<mediapipe::OutputStreamPoller>(std::move(poller));

    landmarksPollerPtr =
        std::make_unique<mediapipe::OutputStreamPoller>(std::move(landmarksPoller));


    MP_RETURN_IF_ERROR(graph.StartRun({}));
  }

  bool processFrame(const cv::Mat& inputRGB8Bit) {
    // Wrap Mat into an ImageFrame.
    auto inputFrame = absl::make_unique<mediapipe::ImageFrame>(
        mediapipe::ImageFormat::SRGB, inputRGB8Bit.cols, inputRGB8Bit.rows,
        mediapipe::ImageFrame::kDefaultAlignmentBoundary);
    cv::Mat inputFrameMat = mediapipe::formats::MatView(inputFrame.get());
    inputRGB8Bit.copyTo(inputFrameMat);

    // Send image packet into the graph.
    size_t frameTimestampUs =
        static_cast<double>(cv::getTickCount()) / static_cast<double>(cv::getTickFrequency()) * 1e6;
    auto status = graph.AddPacketToInputStream(
        kInputStream,
        mediapipe::Adopt(inputFrame.release()).At(mediapipe::Timestamp(frameTimestampUs)));

    if (!status.ok()) {
      LOG(WARNING) << "Graph execution failed: " << status;
      return false;
    }

    // Get the graph result packet, or stop if that fails.
    mediapipe::Packet maskPacket;
    if (!maskPollerPtr || !maskPollerPtr->Next(&maskPacket) || maskPacket.IsEmpty()) return false;
    auto& outputFrame = maskPacket.Get<mediapipe::ImageFrame>();

    // Get pose landmarks.
    if (!landmarksPollerPtr || !landmarksPollerPtr->Next(&poseLandmarksPacket)) {
      return false;
    }

    // Convert back to opencv for display or saving.
    auto mask = mediapipe::formats::MatView(&outputFrame);
    segmentedMask = mask.clone();

    absl::Status landmarksStatus = detectLandmarksWithStatus(poseLandmarks);

    return landmarksStatus.ok();
  }

  absl::Status detectLandmarksWithStatus(nimagna::cv_wrapper::Point3f* poseLandmarks) {
    if (poseLandmarksPacket.IsEmpty()) {
      return absl::CancelledError("Pose landmarks packet is empty.");
    }

    auto retrievedLandmarks = poseLandmarksPacket.Get<::mediapipe::NormalizedLandmarkList>();

    // Convert landmarks to cv::Point3f**.
    const auto landmarksCount = retrievedLandmarks.landmark_size();

    for (int j = 0; j < landmarksCount; ++j) {
      const auto& landmark = retrievedLandmarks.landmark(j);
      poseLandmarks[j].x = landmark.x();
      poseLandmarks[j].y = landmark.y();
      poseLandmarks[j].z = landmark.z();
      visibility[j] = landmark.visibility();
    }

    return absl::OkStatus();
  }

  nimagna::cv_wrapper::Point3f* lastDetectedLandmarks() { return poseLandmarks; }

  cv::Mat lastSegmentedFrame() { return segmentedMask; }
  float* landmarksVisibility() { return visibility; }

  static constexpr size_t kLandmarksCount = 33u;

 private:
  mediapipe::Packet poseLandmarksPacket;
  cv::Mat segmentedMask;
  nimagna::cv_wrapper::Point3f poseLandmarks[kLandmarksCount];
  float visibility[kLandmarksCount] = {0};
  std::unique_ptr<mediapipe::OutputStreamPoller> maskPollerPtr;
  std::unique_ptr<mediapipe::OutputStreamPoller> landmarksPollerPtr;
  mediapipe::CalculatorGraph graph;
  const char* kInputStream = "input_video";
  const char* kOutputSegmentationStream = "segmentation_mask";
  const char* kOutpuLandmarksStream = "pose_landmarks";
};

namespace nimagna {
PoseTracking::PoseTracking(const char* calculatorGraphConfigFile) {
  mImplementation = new PoseTrackingImpl(calculatorGraphConfigFile);
}

bool PoseTracking::processFrame(const cv_wrapper::Mat& inputRGB8Bit) {
  const auto frame = cv::Mat(inputRGB8Bit.rows, inputRGB8Bit.cols, CV_8UC3, inputRGB8Bit.data);
  return mImplementation->processFrame(frame);
}

PoseTracking::PoseLandmarks PoseTracking::lastDetectedLandmarks() {
  return {mImplementation->lastDetectedLandmarks(), mImplementation->landmarksVisibility()};
}

cv_wrapper::Mat PoseTracking::lastSegmentedFrame() {
  const cv::Mat result = mImplementation->lastSegmentedFrame();

  return cv_wrapper::Mat(result.rows, result.cols, result.data);
}

PoseTracking::~PoseTracking()
{
	delete mImplementation;
}
}  // namespace nimagna
