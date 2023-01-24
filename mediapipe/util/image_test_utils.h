#ifndef MEDIAPIPE_UTIL_IMAGE_TEST_UTILS_H_
#define MEDIAPIPE_UTIL_IMAGE_TEST_UTILS_H_

#include <string>

#include "mediapipe/framework/formats/image.h"
#include "mediapipe/framework/packet.h"
#include "mediapipe/framework/port/opencv_core_inc.h"

namespace mediapipe {

// Reads the image file into cv::Mat with RGB channels.
cv::Mat GetRgb(const std::string& path);

// Reads the image file into cv::Mat with RGBA channels.
cv::Mat GetRgba(const std::string& path);

// Reads the image file into cv::Mat with Gray channel.
cv::Mat GetGray(const std::string& path);

// Converts the image channels into corresponding ImageFormat.
mediapipe::ImageFormat::Format GetImageFormat(int image_channels);

// Converts the cv::Mat into ImageFrame packet.
Packet MakeImageFramePacket(cv::Mat input, int timestamp = 0);

// Converts the cv::Mat into Image packet.
Packet MakeImagePacket(cv::Mat input, int timestamp = 0);

}  // namespace mediapipe

#endif  // MEDIAPIPE_UTIL_IMAGE_TEST_UTILS_H_
