#include <cstdlib>
#include <string>

#include "pose_tracking.h"

#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "mediapipe/framework/formats/landmark.pb.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/formats/image_frame.h"
#include "mediapipe/framework/formats/image_frame_opencv.h"
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
		if (!status.ok()) {
			LOG(WARNING) << "Warning: " << status;
		}
	}

	absl::Status initialize(const std::string& calculatorGraphConfigFile) {
		std::string graphContents;
		MP_RETURN_IF_ERROR(mediapipe::file::GetContents(
			calculatorGraphConfigFile,
			&graphContents));

		mediapipe::CalculatorGraphConfig config =
			mediapipe::ParseTextProtoOrDie<mediapipe::CalculatorGraphConfig>(
				graphContents);
		
		MP_RETURN_IF_ERROR(graph.Initialize(config));
		ASSIGN_OR_RETURN(mediapipe::OutputStreamPoller poller,
			graph.AddOutputStreamPoller(kOutputSegmentationStream));

		ASSIGN_OR_RETURN(mediapipe::OutputStreamPoller landmarksPoller,
			graph.AddOutputStreamPoller(kOutpuLandmarksStream));

		ASSIGN_OR_RETURN(mediapipe::OutputStreamPoller posePresencePoller,
			graph.AddOutputStreamPoller(kOutpuPosePresenceStream));


		maskPollerPtr = std::make_unique<mediapipe::OutputStreamPoller>(std::move(poller));

		landmarksPollerPtr = std::make_unique<mediapipe::OutputStreamPoller>(
			std::move(landmarksPoller));

		posePresencePollerPtr = std::make_unique<mediapipe::OutputStreamPoller>(
			std::move(posePresencePoller));

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
			kInputStream, mediapipe::Adopt(inputFrame.release())
			.At(mediapipe::Timestamp(frameTimestampUs)));

		if (!status.ok()) {
			LOG(WARNING) << "Graph execution failed: " << status;
			return false;
		}

		mediapipe::Packet posePresencePacket;
		if (!posePresencePollerPtr || !posePresencePollerPtr->Next(&posePresencePacket)) return false;
		auto landmarksDetected = posePresencePacket.Get<bool>();

		if (!landmarksDetected) {
			return false;
		}

		// Get the graph result packet, or stop if that fails.
		mediapipe::Packet maskPacket;
		if (!maskPollerPtr || !maskPollerPtr->Next(&maskPacket)) return false;
		auto& outputFrame = maskPacket.Get<mediapipe::ImageFrame>();

		// Get pose landmarks.
		if (!landmarksPollerPtr ||
			!landmarksPollerPtr->Next(&poseLandmarksPacket)) {
			return false;
		}

		// Convert back to opencv for display or saving.
		auto mask = mediapipe::formats::MatView(&outputFrame);
		segmentedMask = mask.clone();

		absl::Status landmarksStatus = detectLandmarksWithStatus(poseLandmarks);

		return landmarksStatus.ok();
	}

	absl::Status detectLandmarksWithStatus(
		nimagna::cv_wrapper::Point3f* poseLandmarks) {

		if (poseLandmarksPacket.IsEmpty()) {
			return absl::CancelledError("Pose landmarks packet is empty.");
		}

		auto retrievedLandmarks =
			poseLandmarksPacket
			.Get<::mediapipe::NormalizedLandmarkList>();

		// Convert landmarks to cv::Point3f**.
		const auto landmarksCount = retrievedLandmarks.landmark_size();

		for (int j = 0; j < landmarksCount; ++j) {
			const auto& landmark = retrievedLandmarks.landmark(j);
			poseLandmarks[j].x = landmark.x();
			poseLandmarks[j].y = landmark.y();
			poseLandmarks[j].z = landmark.z();
		}

		return absl::OkStatus();
	}

	nimagna::cv_wrapper::Point3f* lastDetectedLandmarks() {
		return poseLandmarks;
	}

	cv::Mat lastSegmentedFrame() {
		return segmentedMask;
	}

	static constexpr size_t kLandmarksCount = 33u;

private:
	mediapipe::Packet poseLandmarksPacket;
	cv::Mat segmentedMask;
	nimagna::cv_wrapper::Point3f poseLandmarks[kLandmarksCount];
	std::unique_ptr<mediapipe::OutputStreamPoller> posePresencePollerPtr;
	std::unique_ptr<mediapipe::OutputStreamPoller> maskPollerPtr;
	std::unique_ptr<mediapipe::OutputStreamPoller> landmarksPollerPtr;
	mediapipe::CalculatorGraph graph;
	const char* kInputStream = "input_video";
	const char* kOutputSegmentationStream = "segmentation_mask";
	const char* kOutpuLandmarksStream = "pose_landmarks";
	const char* kOutpuPosePresenceStream = "pose_presence";
};

namespace nimagna {
	PoseTracking::PoseTracking(const char* calculatorGraphConfigFile) {
		myInstance = new PoseTrackingImpl(calculatorGraphConfigFile);
	}

	bool PoseTracking::processFrame(const cv_wrapper::Mat& inputRGB8Bit) {
		auto* instance = static_cast<PoseTrackingImpl*>(myInstance);
		const auto frame = cv::Mat(inputRGB8Bit.rows, inputRGB8Bit.cols, CV_8UC3, inputRGB8Bit.data);
		return instance->processFrame(frame);
	}

	cv_wrapper::Point3f* PoseTracking::lastDetectedLandmarks() {
		auto* instance = static_cast<PoseTrackingImpl*>(myInstance);
		return instance->lastDetectedLandmarks();
	}

	cv_wrapper::Mat PoseTracking::lastSegmentedFrame() {
		auto* instance = static_cast<PoseTrackingImpl*>(myInstance);
		const cv::Mat result = instance->lastSegmentedFrame();

		return cv_wrapper::Mat(result.rows, result.cols, result.data);
	}

}
