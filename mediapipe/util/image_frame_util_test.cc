#include "mediapipe/util/image_frame_util.h"

#include <cstdint>

#include "absl/types/span.h"
#include "libyuv/video_common.h"
#include "mediapipe/framework/formats/image_format.pb.h"
#include "mediapipe/framework/formats/image_frame.h"
#include "mediapipe/framework/formats/image_frame_opencv.h"
#include "mediapipe/framework/formats/yuv_image.h"
#include "mediapipe/framework/port/benchmark.h"
#include "mediapipe/framework/port/gtest.h"
#include "mediapipe/framework/port/opencv_core_inc.h"

namespace mediapipe {
namespace image_frame_util {
namespace {

TEST(LinearRgb16ToSrgbTest, Black1x1) {
  cv::Mat source(1, 1, CV_16UC3, cv::Scalar(0, 0, 0));
  cv::Mat destination;
  LinearRgb16ToSrgb(source, &destination);
  EXPECT_EQ(destination.type(), CV_8UC3);
  EXPECT_EQ(destination.at<cv::Vec3b>(0, 0), cv::Vec3b(0, 0, 0));
}

TEST(LinearRgb16ToSrgbTest, White1x1) {
  cv::Mat source(1, 1, CV_16UC3, cv::Scalar(65535, 65535, 65535));
  cv::Mat destination;
  LinearRgb16ToSrgb(source, &destination);
  EXPECT_EQ(destination.at<cv::Vec3b>(0, 0), cv::Vec3b(255, 255, 255));
}

TEST(LinearRgb16ToSrgbTest, MidValue1x1) {
  cv::Mat source(1, 1, CV_16UC3, cv::Scalar(32768, 32768, 32768));
  cv::Mat destination;
  LinearRgb16ToSrgb(source, &destination);
  // 32768/65535 = 0.5. sRGB(0.5) is approx 188.
  EXPECT_EQ(destination.at<cv::Vec3b>(0, 0), cv::Vec3b(188, 188, 188));
}

TEST(LinearRgb16ToSrgbTest, MixedValues2x2) {
  // 2x2 test with mixed values
  cv::Mat source(2, 2, CV_16UC3);
  source.at<cv::Vec3w>(0, 0) = cv::Vec3w(0, 0, 0);
  source.at<cv::Vec3w>(0, 1) = cv::Vec3w(65535, 65535, 65535);
  source.at<cv::Vec3w>(1, 0) = cv::Vec3w(32768, 16384, 8192);
  source.at<cv::Vec3w>(1, 1) = cv::Vec3w(200, 400, 600);

  cv::Mat destination;
  LinearRgb16ToSrgb(source, &destination);

  EXPECT_EQ(destination.at<cv::Vec3b>(0, 0), cv::Vec3b(0, 0, 0));
  EXPECT_EQ(destination.at<cv::Vec3b>(0, 1), cv::Vec3b(255, 255, 255));

  // 32768 -> 188
  // 16384 -> 137
  // 8192 -> 99
  EXPECT_EQ(destination.at<cv::Vec3b>(1, 0), cv::Vec3b(188, 137, 99));

  // 200 -> 10
  // 400 -> 18
  // 600 -> 24
  EXPECT_EQ(destination.at<cv::Vec3b>(1, 1), cv::Vec3b(10, 18, 24));
}

// YUV tests use a solid 2x2 RGB (200, 100, 50) image, which is (123, 91, 175)
// in limited-range BT.601 YCbCr. Tolerance covers 8-bit rounding and libyuv's
// fixed-point math.
constexpr int kTolerance = 5;

TEST(ImageFrameUtilTest, ImageFrameToYUVImage) {
  for (ImageFormat::Format format : {ImageFormat::SRGB, ImageFormat::SRGBA}) {
    ImageFrame rgb(format, 2, 2);
    formats::MatView(&rgb).setTo(cv::Scalar(200, 100, 50, 255));
    YUVImage yuv;
    ImageFrameToYUVImage(rgb, &yuv);
    EXPECT_EQ(yuv.fourcc(), libyuv::FOURCC_I420);
    EXPECT_LE(
        cv::norm(cv::Mat_<uint8_t>(2, 2, yuv.mutable_data(0), yuv.stride(0)),
                 cv::Mat_<uint8_t>(2, 2, 123), cv::NORM_INF),
        kTolerance);
    EXPECT_LE(cv::norm(cv::Mat_<uint8_t>(1, 1, yuv.mutable_data(1)),
                       cv::Mat_<uint8_t>(1, 1, 91), cv::NORM_INF),
              kTolerance);
    EXPECT_LE(cv::norm(cv::Mat_<uint8_t>(1, 1, yuv.mutable_data(2)),
                       cv::Mat_<uint8_t>(1, 1, 175), cv::NORM_INF),
              kTolerance);
  }
}

TEST(ImageFrameUtilTest, ImageFrameToYUVNV12Image) {
  ImageFrame rgb(ImageFormat::SRGB, 2, 2);
  formats::MatView(&rgb).setTo(cv::Scalar(200, 100, 50));
  YUVImage yuv;
  ImageFrameToYUVNV12Image(rgb, &yuv);
  EXPECT_EQ(yuv.fourcc(), libyuv::FOURCC_NV12);
  EXPECT_LE(
      cv::norm(cv::Mat_<uint8_t>(2, 2, yuv.mutable_data(0), yuv.stride(0)),
               cv::Mat_<uint8_t>(2, 2, 123), cv::NORM_INF),
      kTolerance);
  EXPECT_LE(cv::norm(cv::Mat_<uint8_t>(1, 2, yuv.mutable_data(1)),
                     cv::Mat_<uint8_t>({1, 2}, {91, 175}), cv::NORM_INF),
            kTolerance);
}

TEST(ImageFrameUtilTest, YUVImageToImageFrameBt601) {
  uint8_t y[] = {123, 123, 123, 123}, u[] = {91}, v[] = {175};
  YUVImage yuv;
  yuv.Initialize(libyuv::FOURCC_I420, nullptr, absl::MakeSpan(y), 2,
                 absl::MakeSpan(u), 1, absl::MakeSpan(v), 1, 2, 2);
  ImageFrame frame;
  YUVImageToImageFrame(yuv, &frame, /*use_bt709=*/false);
  EXPECT_EQ(frame.Format(), ImageFormat::SRGB);
  EXPECT_LE(cv::norm(formats::MatView(&frame),
                     cv::Mat_<cv::Vec3b>(2, 2, cv::Vec3b(200, 100, 50)),
                     cv::NORM_INF),
            kTolerance);
}

TEST(ImageFrameUtilTest, YUVImageToImageFrameBt709) {
  uint8_t y[] = {123, 123, 123, 123}, u[] = {91}, v[] = {175};
  YUVImage yuv;
  yuv.Initialize(libyuv::FOURCC_I420, nullptr, absl::MakeSpan(y), 2,
                 absl::MakeSpan(u), 1, absl::MakeSpan(v), 1, 2, 2);
  ImageFrame frame;
  YUVImageToImageFrame(yuv, &frame, /*use_bt709=*/true);
  // The samples are BT.601-encoded, so decoding them with the BT.709 matrix
  // shifts the color by ~10 per channel; 30 leaves headroom.
  EXPECT_LE(cv::norm(formats::MatView(&frame),
                     cv::Mat_<cv::Vec3b>(2, 2, cv::Vec3b(200, 100, 50)),
                     cv::NORM_INF),
            30);
}

TEST(ImageFrameUtilTest, I420ToImageFrame) {
  uint8_t y[] = {123, 123, 123, 123}, u[] = {91}, v[] = {175};
  YUVImage yuv;
  yuv.Initialize(libyuv::FOURCC_I420, nullptr, absl::MakeSpan(y), 2,
                 absl::MakeSpan(u), 1, absl::MakeSpan(v), 1, 2, 2);
  ImageFrame frame;
  YUVImageToImageFrameFromFormat(yuv, &frame);
  EXPECT_LE(cv::norm(formats::MatView(&frame),
                     cv::Mat_<cv::Vec3b>(2, 2, cv::Vec3b(200, 100, 50)),
                     cv::NORM_INF),
            kTolerance);
}

TEST(ImageFrameUtilTest, Yv12ToImageFrame) {
  uint8_t y[] = {123, 123, 123, 123}, u[] = {91}, v[] = {175};
  YUVImage yuv;  // YV12: V plane before U plane.
  yuv.Initialize(libyuv::FOURCC_YV12, nullptr, absl::MakeSpan(y), 2,
                 absl::MakeSpan(v), 1, absl::MakeSpan(u), 1, 2, 2);
  ImageFrame frame;
  YUVImageToImageFrameFromFormat(yuv, &frame);
  EXPECT_LE(cv::norm(formats::MatView(&frame),
                     cv::Mat_<cv::Vec3b>(2, 2, cv::Vec3b(200, 100, 50)),
                     cv::NORM_INF),
            kTolerance);
}

TEST(ImageFrameUtilTest, Nv12ToImageFrame) {
  uint8_t y[] = {123, 123, 123, 123}, uv[] = {91, 175};
  YUVImage yuv;
  yuv.Initialize(libyuv::FOURCC_NV12, nullptr, absl::MakeSpan(y), 2,
                 absl::MakeSpan(uv), 2, {}, 0, 2, 2);
  ImageFrame frame;
  YUVImageToImageFrameFromFormat(yuv, &frame);
  EXPECT_LE(cv::norm(formats::MatView(&frame),
                     cv::Mat_<cv::Vec3b>(2, 2, cv::Vec3b(200, 100, 50)),
                     cv::NORM_INF),
            kTolerance);
}

TEST(ImageFrameUtilTest, Nv21ToImageFrame) {
  uint8_t y[] = {123, 123, 123, 123}, vu[] = {175, 91};
  YUVImage yuv;
  yuv.Initialize(libyuv::FOURCC_NV21, nullptr, absl::MakeSpan(y), 2,
                 absl::MakeSpan(vu), 2, {}, 0, 2, 2);
  ImageFrame frame;
  YUVImageToImageFrameFromFormat(yuv, &frame);
  EXPECT_LE(cv::norm(formats::MatView(&frame),
                     cv::Mat_<cv::Vec3b>(2, 2, cv::Vec3b(200, 100, 50)),
                     cv::NORM_INF),
            kTolerance);
}

cv::Mat MakeRGBTestImage(int rows, int cols) {
  cv::Mat m(rows, cols, CV_16UC3);
  for (int r = 0; r < rows; r++) {
    for (int c = 0; c < cols; c++) {
      m.at<cv::Vec3w>(r, c) = cv::Vec3w(r % 256, c % 256, (r + c) % 256);
    }
  }
  return m;
}

void BM_LinearRgb16ToSrgb(benchmark::State& state) {
  const int rows = state.range(0);
  const int cols = state.range(0);
  cv::Mat source = MakeRGBTestImage(rows, cols);
  for (auto s : state) {
    benchmark::DoNotOptimize(source);
    cv::Mat destination(rows, cols, CV_8UC3);
    LinearRgb16ToSrgb(source, &destination);
    benchmark::DoNotOptimize(destination);
  }
}
BENCHMARK(BM_LinearRgb16ToSrgb)->Range(32, 1024);

}  // namespace
}  // namespace image_frame_util
}  // namespace mediapipe
