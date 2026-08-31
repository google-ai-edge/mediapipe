#include "mediapipe/framework/formats/yuv_image.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

#include "absl/base/attributes.h"
#include "absl/types/span.h"
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
    ABSL_INTERNAL_DISABLE_DEPRECATED_DECLARATION_WARNING
    yuv_image.Initialize(fourcc, deallocation_function,  //
                         &data0, stride0,                //
                         &data1, stride1,                //
                         &data2, stride2,                //
                         width, height, bit_depth);
    ABSL_INTERNAL_RESTORE_DEPRECATED_DECLARATION_WARNING

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
    ABSL_INTERNAL_DISABLE_DEPRECATED_DECLARATION_WARNING
    yuv_image.Initialize(fourcc, deallocation_function,  //
                         &data0, stride0,                //
                         &data1, stride1,                //
                         &data2, stride2,                //
                         width, height, bit_depth);
    ABSL_INTERNAL_RESTORE_DEPRECATED_DECLARATION_WARNING

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
    ABSL_INTERNAL_DISABLE_DEPRECATED_DECLARATION_WARNING
    yuv_image.Initialize(fourcc, deallocation_function,  //
                         &data0, stride0,                //
                         &data1, stride1,                //
                         &data2, stride2,                //
                         width, height, bit_depth);
    ABSL_INTERNAL_RESTORE_DEPRECATED_DECLARATION_WARNING

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

// ---------------------------------------------------------------------------
// Span-based API tests.
// ---------------------------------------------------------------------------

// Verifies the non-deprecated, span-based Initialize() overload.
TEST(YUVImageTest, TestSpanBasedInitialize) {
  const libyuv::FourCC fourcc = libyuv::FOURCC_I420;
  const int stride0 = 100, stride1 = 50, stride2 = 50;
  const int width = 100, height = 60;
  const int bit_depth = 8;
  const size_t size0 = static_cast<size_t>(stride0) * height;
  const size_t size1 = static_cast<size_t>(stride1) * ((height + 1) / 2);
  const size_t size2 = static_cast<size_t>(stride2) * ((height + 1) / 2);

  std::vector<uint8_t> buf0(size0);
  std::vector<uint8_t> buf1(size1);
  std::vector<uint8_t> buf2(size2);
  absl::Span<uint8_t> span0 = absl::MakeSpan(buf0);
  absl::Span<uint8_t> span1 = absl::MakeSpan(buf1);
  absl::Span<uint8_t> span2 = absl::MakeSpan(buf2);
  uint8_t* ptr0 = span0.data();
  uint8_t* ptr1 = span1.data();
  uint8_t* ptr2 = span2.data();

  int deallocation_counter = 0;
  {
    YUVImage yuv_image;
    yuv_image.Initialize(
        fourcc, [&deallocation_counter] { ++deallocation_counter; },  //
        span0, stride0,                                               //
        span1, stride1,                                               //
        span2, stride2,                                               //
        width, height, bit_depth);

    EXPECT_EQ(yuv_image.fourcc(), fourcc);
    EXPECT_EQ(yuv_image.data(0), ptr0);
    EXPECT_EQ(yuv_image.data(1), ptr1);
    EXPECT_EQ(yuv_image.data(2), ptr2);
    EXPECT_EQ(yuv_image.data_span(0).size(), size0);
    EXPECT_EQ(yuv_image.data_span(1).size(), size1);
    EXPECT_EQ(yuv_image.data_span(2).size(), size2);
    EXPECT_EQ(yuv_image.stride(0), stride0);
    EXPECT_EQ(yuv_image.stride(1), stride1);
    EXPECT_EQ(yuv_image.stride(2), stride2);
    EXPECT_EQ(yuv_image.width(), width);
    EXPECT_EQ(yuv_image.height(), height);
    EXPECT_EQ(yuv_image.bit_depth(), bit_depth);
  }
  EXPECT_EQ(deallocation_counter, 1);
}

// ---------------------------------------------------------------------------
// Deprecated (raw/unique_ptr pointer-based) API tests. These exercise the
// legacy constructors that are retained for backwards compatibility.
// ---------------------------------------------------------------------------

// Verifies the deprecated constructor that takes a single owning unique_ptr
// buffer plus raw pointers into it for each plane.
TEST(YUVImageTest, TestDeprecatedConstructorSingleBuffer) {
  const libyuv::FourCC fourcc = libyuv::FOURCC_I420;
  const int stride0 = 100, stride1 = 50, stride2 = 50;
  const int width = 100, height = 60;
  const int bit_depth = 8;
  const size_t size0 = static_cast<size_t>(stride0) * height;
  const size_t size1 = static_cast<size_t>(stride1) * ((height + 1) / 2);
  const size_t size2 = static_cast<size_t>(stride2) * ((height + 1) / 2);

  auto data = std::make_unique<uint8_t[]>(size0 + size1 + size2);
  uint8_t* ptr0 = data.get();
  uint8_t* ptr1 = data.get() + size0;
  uint8_t* ptr2 = data.get() + size0 + size1;

  ABSL_INTERNAL_DISABLE_DEPRECATED_DECLARATION_WARNING
  YUVImage yuv_image(fourcc, std::move(data),  //
                     ptr0, stride0,            //
                     ptr1, stride1,            //
                     ptr2, stride2,            //
                     width, height, bit_depth);
  ABSL_INTERNAL_RESTORE_DEPRECATED_DECLARATION_WARNING

  EXPECT_EQ(yuv_image.fourcc(), fourcc);
  EXPECT_EQ(yuv_image.data(0), ptr0);
  EXPECT_EQ(yuv_image.data(1), ptr1);
  EXPECT_EQ(yuv_image.data(2), ptr2);
  EXPECT_EQ(yuv_image.stride(0), stride0);
  EXPECT_EQ(yuv_image.stride(1), stride1);
  EXPECT_EQ(yuv_image.stride(2), stride2);
  EXPECT_EQ(yuv_image.width(), width);
  EXPECT_EQ(yuv_image.height(), height);
  EXPECT_EQ(yuv_image.bit_depth(), bit_depth);
}

// Verifies the deprecated constructor that takes three separate owning
// unique_ptr buffers, one per plane.
TEST(YUVImageTest, TestDeprecatedConstructorThreeBuffers) {
  const libyuv::FourCC fourcc = libyuv::FOURCC_I420;
  const int stride0 = 100, stride1 = 50, stride2 = 50;
  const int width = 100, height = 60;
  const int bit_depth = 8;
  const size_t size0 = static_cast<size_t>(stride0) * height;
  const size_t size1 = static_cast<size_t>(stride1) * ((height + 1) / 2);
  const size_t size2 = static_cast<size_t>(stride2) * ((height + 1) / 2);

  auto data0 = std::make_unique<uint8_t[]>(size0);
  auto data1 = std::make_unique<uint8_t[]>(size1);
  auto data2 = std::make_unique<uint8_t[]>(size2);
  uint8_t* ptr0 = data0.get();
  uint8_t* ptr1 = data1.get();
  uint8_t* ptr2 = data2.get();

  ABSL_INTERNAL_DISABLE_DEPRECATED_DECLARATION_WARNING
  YUVImage yuv_image(fourcc,                     //
                     std::move(data0), stride0,  //
                     std::move(data1), stride1,  //
                     std::move(data2), stride2,  //
                     width, height, bit_depth);
  ABSL_INTERNAL_RESTORE_DEPRECATED_DECLARATION_WARNING

  EXPECT_EQ(yuv_image.fourcc(), fourcc);
  EXPECT_EQ(yuv_image.data(0), ptr0);
  EXPECT_EQ(yuv_image.data(1), ptr1);
  EXPECT_EQ(yuv_image.data(2), ptr2);
  EXPECT_EQ(yuv_image.stride(0), stride0);
  EXPECT_EQ(yuv_image.stride(1), stride1);
  EXPECT_EQ(yuv_image.stride(2), stride2);
  EXPECT_EQ(yuv_image.width(), width);
  EXPECT_EQ(yuv_image.height(), height);
  EXPECT_EQ(yuv_image.bit_depth(), bit_depth);
}

// Verifies the deprecated constructor that disambiguates all-nullptr planes.
TEST(YUVImageTest, TestDeprecatedConstructorNullptr) {
  const libyuv::FourCC fourcc = libyuv::FOURCC_I420;
  const int stride0 = 100, stride1 = 50, stride2 = 50;
  const int width = 100, height = 60;
  const int bit_depth = 8;

  ABSL_INTERNAL_DISABLE_DEPRECATED_DECLARATION_WARNING
  YUVImage yuv_image(fourcc,            //
                     nullptr, stride0,  //
                     nullptr, stride1,  //
                     nullptr, stride2,  //
                     width, height, bit_depth);
  ABSL_INTERNAL_RESTORE_DEPRECATED_DECLARATION_WARNING

  EXPECT_EQ(yuv_image.fourcc(), fourcc);
  EXPECT_EQ(yuv_image.data(0), nullptr);
  EXPECT_EQ(yuv_image.data(1), nullptr);
  EXPECT_EQ(yuv_image.data(2), nullptr);
  EXPECT_EQ(yuv_image.stride(0), stride0);
  EXPECT_EQ(yuv_image.stride(1), stride1);
  EXPECT_EQ(yuv_image.stride(2), stride2);
  EXPECT_EQ(yuv_image.width(), width);
  EXPECT_EQ(yuv_image.height(), height);
  EXPECT_EQ(yuv_image.bit_depth(), bit_depth);
}

}  // namespace
}  // namespace mediapipe
