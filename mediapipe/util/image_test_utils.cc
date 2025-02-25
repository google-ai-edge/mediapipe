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
#include "mediapipe/gpu/gpu_buffer.h"
#include "mediapipe/gpu/gpu_buffer_format.h"

namespace mediapipe {

namespace {

template <ImageFormat::Format Format, typename DataType>
ImageFrame CreateTestImageFrame(int width, int height, DataType max_value) {
  ImageFrame image_frame(Format, width, height,
                         /*alignment_boundary=*/1);
  const int num_channels = image_frame.NumberOfChannels();
  const float num_values = width * height * num_channels;
  uint8_t* const data_ptr =
      reinterpret_cast<uint8_t*>(image_frame.MutablePixelData());
  for (int y = 0; y < height; ++y) {
    uint8_t* const row = data_ptr + image_frame.WidthStep() * y;
    for (int x = 0; x < width; ++x) {
      DataType* pixel = reinterpret_cast<DataType*>(row) + x * num_channels;
      for (int c = 0; c < num_channels; ++c) {
        // Fill pixel channel with a value in [0:max_value] range.
        pixel[c] =
            static_cast<DataType>(static_cast<float>(y * width * num_channels +
                                                     x * num_channels + c) /
                                  num_values * max_value);
      }
    }
  }
  return image_frame;
}

}  // namespace

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

ImageFrame CreateTestFloat32ImageFrame(int width, int height) {
  return CreateTestImageFrame<ImageFormat::VEC32F1, float>(width, height,
                                                           /*max_value=*/1.0f);
}

ImageFrame CreateTestGrey8ImageFrame(int width, int height) {
  return CreateTestImageFrame<ImageFormat::GRAY8, uint8_t>(width, height,
                                                           /*max_value=*/255);
}

ImageFrame CreateTestRgba8ImageFrame(int width, int height) {
  return CreateTestImageFrame<ImageFormat::SRGBA, uint8_t>(
      width, height, /*max_value=*/255.0f);
}

ImageFrame CreateTestRgb8ImageFrame(int width, int height) {
  return CreateTestImageFrame<ImageFormat::SRGB, uint8_t>(width, height,
                                                          /*max_value=*/255.0f);
}

GpuBuffer CreateTestFloat32GpuBuffer(int width, int height) {
  GpuBuffer buffer(width, height, GpuBufferFormat::kGrayFloat32);
  std::shared_ptr<ImageFrame> view = buffer.GetWriteView<ImageFrame>();
  *view = CreateTestFloat32ImageFrame(width, height);
  return buffer;
}

GpuBuffer CreateTestGrey8GpuBuffer(int width, int height) {
  GpuBuffer buffer(width, height, GpuBufferFormat::kOneComponent8);
  std::shared_ptr<ImageFrame> view = buffer.GetWriteView<ImageFrame>();
  *view = CreateTestGrey8ImageFrame(width, height);
  return buffer;
}

GpuBuffer CreateTestRgba8GpuBuffer(int width, int height) {
  GpuBuffer buffer(width, height, GpuBufferFormat::kRGBA32);
  std::shared_ptr<ImageFrame> view = buffer.GetWriteView<ImageFrame>();
  *view = CreateTestRgba8ImageFrame(width, height);
  return buffer;
}

GpuBuffer CreateTestRgb8GpuBuffer(int width, int height) {
  GpuBuffer buffer(width, height, GpuBufferFormat::kRGB24);
  std::shared_ptr<ImageFrame> view = buffer.GetWriteView<ImageFrame>();
  *view = CreateTestRgb8ImageFrame(width, height);
  return buffer;
}

}  // namespace mediapipe
