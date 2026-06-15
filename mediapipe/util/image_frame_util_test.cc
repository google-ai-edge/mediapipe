#include "mediapipe/util/image_frame_util.h"

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
