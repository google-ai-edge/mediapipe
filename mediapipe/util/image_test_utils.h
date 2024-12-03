#ifndef MEDIAPIPE_UTIL_IMAGE_TEST_UTILS_H_
#define MEDIAPIPE_UTIL_IMAGE_TEST_UTILS_H_

#include <string>

#include "mediapipe/framework/formats/image_format.pb.h"
#include "mediapipe/framework/formats/image_frame.h"
#include "mediapipe/framework/packet.h"
#include "mediapipe/framework/port/opencv_core_inc.h"
#include "mediapipe/gpu/gpu_buffer.h"

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

// Converts RGBA Mat to BGR.
cv::Mat RgbaToBgr(cv::Mat rgba);

// Generates single-channel float32 ImageFrame with increasing [0,1] values.
ImageFrame CreateTestFloat32ImageFrame(int width, int height);

// Generates single-channel uint8 ImageFrame with increasing [0,255] values.
ImageFrame CreateTestGrey8ImageFrame(int width, int height);

// Generates 4 channel uint8 RGBA ImageFrame with increasing [0,255] values.
ImageFrame CreateTestRgba8ImageFrame(int width, int height);

// Generates single-channel float32 GpuBuffer with increasing [0,1] values.
GpuBuffer CreateTestFloat32GpuBuffer(int width, int height);

// Generates single-channel uint8 GpuBuffer with increasing [0,255] values.
GpuBuffer CreateTestGrey8GpuBuffer(int width, int height);

// Generates 4 channel uint8 RGBA GpuBuffer with increasing [0,255] values.
GpuBuffer CreateTestRgba8GpuBuffer(int width, int height);

}  // namespace mediapipe

#endif  // MEDIAPIPE_UTIL_IMAGE_TEST_UTILS_H_
