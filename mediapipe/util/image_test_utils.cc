#include "mediapipe/util/image_test_utils.h"

#include <cstdint>
#include <memory>
#include <utility>

#include "absl/log/absl_log.h"
#include "mediapipe/framework/formats/image.h"
#include "mediapipe/framework/formats/image_format.pb.h"
#include "mediapipe/framework/formats/image_frame.h"
#include "mediapipe/framework/formats/image_frame_opencv.h"
#include "mediapipe/framework/packet.h"
#include "mediapipe/framework/port/opencv_core_inc.h"
#include "mediapipe/framework/port/opencv_imgcodecs_inc.h"
#include "mediapipe/framework/port/opencv_imgproc_inc.h"
#include "mediapipe/framework/timestamp.h"

namespace mediapipe {

cv::Mat GetRgb(const std::string& path) {
  cv::Mat bgr = cv::imread(path);
  cv::Mat rgb;
  cv::cvtColor(bgr, rgb, cv::COLOR_BGR2RGB);
  return rgb;
}

cv::Mat GetRgba(const std::string& path) {
  cv::Mat bgr = cv::imread(path);
  cv::Mat rgb;
  cv::cvtColor(bgr, rgb, cv::COLOR_BGR2RGBA);
  return rgb;
}

cv::Mat GetGray(const std::string& path) {
  cv::Mat bgr = cv::imread(path);
  cv::Mat gray;
  cv::cvtColor(bgr, gray, cv::COLOR_BGR2GRAY);
  return gray;
}

mediapipe::ImageFormat::Format GetImageFormat(int image_channels) {
  if (image_channels == 4) {
    return ImageFormat::SRGBA;
  } else if (image_channels == 3) {
    return ImageFormat::SRGB;
  } else if (image_channels == 1) {
    return ImageFormat::GRAY8;
  }
  ABSL_LOG(FATAL) << "Unsupported input image channles: " << image_channels;
}

Packet MakeImageFramePacket(cv::Mat input, int timestamp) {
  ImageFrame input_image(GetImageFormat(input.channels()), input.cols,
                         input.rows, input.step, input.data,
                         [input](uint8_t*) mutable { input.release(); });
  return MakePacket<ImageFrame>(std::move(input_image))
      .At(Timestamp(timestamp));
}

Packet MakeImagePacket(cv::Mat input, int timestamp) {
  mediapipe::Image input_image(std::make_shared<mediapipe::ImageFrame>(
      GetImageFormat(input.channels()), input.cols, input.rows, input.step,
      input.data, [input](uint8_t*) mutable { input.release(); }));
  return MakePacket<mediapipe::Image>(std::move(input_image))
      .At(Timestamp(timestamp));
}

cv::Mat RgbaToBgr(cv::Mat rgba) {
  cv::Mat bgra;
  cv::cvtColor(rgba, bgra, cv::COLOR_RGBA2BGR);
  return bgra;
}

}  // namespace mediapipe
