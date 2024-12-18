#include "mediapipe/framework/formats/yuv_image.h"

#include <cstdint>
#include <utility>

#include "libyuv/video_common.h"
#include "mediapipe/framework/port/gtest.h"

namespace mediapipe {
namespace {

// See:
// https://clang.llvm.org/extra/clang-tidy/checks/bugprone/use-after-move.html
template <class T>
void SILENCE_USE_AFTER_MOVE(T&) {}

TEST(YUVImageTest, TestInitializeAndDestruct) {
  uint8_t data0 = 0, data1 = 1, data2 = 2;
  const libyuv::FourCC fourcc = libyuv::FOURCC_I420;
  const int stride0 = 100, stride1 = 50, stride2 = 50;
  const int width = 100, height = 60;
  const int bit_depth = 4;
  int deallocation_counter = 0;
  auto deallocation_function = [&deallocation_counter] {
    ++deallocation_counter;
  };
  {
    YUVImage yuv_image;
    yuv_image.Initialize(fourcc, deallocation_function,  //
                         &data0, stride0,                //
                         &data1, stride1,                //
                         &data2, stride2,                //
                         width, height, bit_depth);

    EXPECT_EQ(yuv_image.fourcc(), fourcc);
    EXPECT_EQ(yuv_image.data(0), &data0);
    EXPECT_EQ(yuv_image.data(1), &data1);
    EXPECT_EQ(yuv_image.data(2), &data2);
    EXPECT_EQ(yuv_image.stride(0), stride0);
    EXPECT_EQ(yuv_image.stride(1), stride1);
    EXPECT_EQ(yuv_image.stride(2), stride2);
    EXPECT_EQ(yuv_image.width(), width);
    EXPECT_EQ(yuv_image.height(), height);
    EXPECT_EQ(yuv_image.bit_depth(), bit_depth);
  }
  EXPECT_EQ(deallocation_counter, 1);
}

TEST(YUVImageTest, TestMoveConstructor) {
  uint8_t data0 = 0, data1 = 1, data2 = 2;
  const libyuv::FourCC fourcc = libyuv::FOURCC_I420;
  const int stride0 = 100, stride1 = 50, stride2 = 50;
  const int width = 100, height = 60;
  const int bit_depth = 4;
  int deallocation_counter = 0;
  auto deallocation_function = [&deallocation_counter] {
    ++deallocation_counter;
  };
  {
    YUVImage yuv_image;
    yuv_image.Initialize(fourcc, deallocation_function,  //
                         &data0, stride0,                //
                         &data1, stride1,                //
                         &data2, stride2,                //
                         width, height, bit_depth);

    EXPECT_EQ(yuv_image.fourcc(), fourcc);
    EXPECT_EQ(yuv_image.data(0), &data0);
    EXPECT_EQ(yuv_image.data(1), &data1);
    EXPECT_EQ(yuv_image.data(2), &data2);
    EXPECT_EQ(yuv_image.stride(0), stride0);
    EXPECT_EQ(yuv_image.stride(1), stride1);
    EXPECT_EQ(yuv_image.stride(2), stride2);
    EXPECT_EQ(yuv_image.width(), width);
    EXPECT_EQ(yuv_image.height(), height);
    EXPECT_EQ(yuv_image.bit_depth(), bit_depth);

    YUVImage yuv_image2(std::move(yuv_image));

    // ClangTidy will complain about accessing yuv_image after it has been moved
    // from. The C++ standard says that "moved-from objects shall be placed in a
    // valid but unspecified state". These tests are here to ensure that.
    SILENCE_USE_AFTER_MOVE(yuv_image);
    EXPECT_EQ(yuv_image.fourcc(), libyuv::FOURCC_ANY);
    EXPECT_EQ(yuv_image.data(0), nullptr);
    EXPECT_EQ(yuv_image.data(1), nullptr);
    EXPECT_EQ(yuv_image.data(2), nullptr);
    EXPECT_EQ(yuv_image.stride(0), 0);
    EXPECT_EQ(yuv_image.stride(1), 0);
    EXPECT_EQ(yuv_image.stride(2), 0);
    EXPECT_EQ(yuv_image.width(), 0);
    EXPECT_EQ(yuv_image.height(), 0);
    EXPECT_EQ(yuv_image.bit_depth(), 0);

    EXPECT_EQ(yuv_image2.fourcc(), fourcc);
    EXPECT_EQ(yuv_image2.data(0), &data0);
    EXPECT_EQ(yuv_image2.data(1), &data1);
    EXPECT_EQ(yuv_image2.data(2), &data2);
    EXPECT_EQ(yuv_image2.stride(0), stride0);
    EXPECT_EQ(yuv_image2.stride(1), stride1);
    EXPECT_EQ(yuv_image2.stride(2), stride2);
    EXPECT_EQ(yuv_image2.width(), width);
    EXPECT_EQ(yuv_image2.height(), height);
    EXPECT_EQ(yuv_image2.bit_depth(), bit_depth);
  }
  EXPECT_EQ(deallocation_counter, 1);
}

TEST(YUVImageTest, TestMoveAssignment) {
  uint8_t data0 = 0, data1 = 1, data2 = 2;
  const libyuv::FourCC fourcc = libyuv::FOURCC_I420;
  const int stride0 = 100, stride1 = 50, stride2 = 50;
  const int width = 100, height = 60;
  const int bit_depth = 4;
  int deallocation_counter = 0;
  auto deallocation_function = [&deallocation_counter] {
    ++deallocation_counter;
  };
  {
    YUVImage yuv_image;
    yuv_image.Initialize(fourcc, deallocation_function,  //
                         &data0, stride0,                //
                         &data1, stride1,                //
                         &data2, stride2,                //
                         width, height, bit_depth);

    EXPECT_EQ(yuv_image.fourcc(), fourcc);
    EXPECT_EQ(yuv_image.data(0), &data0);
    EXPECT_EQ(yuv_image.data(1), &data1);
    EXPECT_EQ(yuv_image.data(2), &data2);
    EXPECT_EQ(yuv_image.stride(0), stride0);
    EXPECT_EQ(yuv_image.stride(1), stride1);
    EXPECT_EQ(yuv_image.stride(2), stride2);
    EXPECT_EQ(yuv_image.width(), width);
    EXPECT_EQ(yuv_image.height(), height);
    EXPECT_EQ(yuv_image.bit_depth(), bit_depth);

    YUVImage yuv_image2;
    yuv_image2 = std::move(yuv_image);

    // ClangTidy will complain about accessing yuv_image after it has been moved
    // from. The C++ standard says that "moved-from objects shall be placed in a
    // valid but unspecified state". These tests are here to ensure that.
    SILENCE_USE_AFTER_MOVE(yuv_image);
    EXPECT_EQ(yuv_image.fourcc(), libyuv::FOURCC_ANY);
    EXPECT_EQ(yuv_image.data(0), nullptr);
    EXPECT_EQ(yuv_image.data(1), nullptr);
    EXPECT_EQ(yuv_image.data(2), nullptr);
    EXPECT_EQ(yuv_image.stride(0), 0);
    EXPECT_EQ(yuv_image.stride(1), 0);
    EXPECT_EQ(yuv_image.stride(2), 0);
    EXPECT_EQ(yuv_image.width(), 0);
    EXPECT_EQ(yuv_image.height(), 0);
    EXPECT_EQ(yuv_image.bit_depth(), 0);

    EXPECT_EQ(yuv_image2.fourcc(), fourcc);
    EXPECT_EQ(yuv_image2.data(0), &data0);
    EXPECT_EQ(yuv_image2.data(1), &data1);
    EXPECT_EQ(yuv_image2.data(2), &data2);
    EXPECT_EQ(yuv_image2.stride(0), stride0);
    EXPECT_EQ(yuv_image2.stride(1), stride1);
    EXPECT_EQ(yuv_image2.stride(2), stride2);
    EXPECT_EQ(yuv_image2.width(), width);
    EXPECT_EQ(yuv_image2.height(), height);
    EXPECT_EQ(yuv_image2.bit_depth(), bit_depth);
  }
  EXPECT_EQ(deallocation_counter, 1);
}

}  // namespace
}  // namespace mediapipe
