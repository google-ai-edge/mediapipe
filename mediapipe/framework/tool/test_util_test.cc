#include "mediapipe/framework/tool/test_util.h"

#include <cstdint>
#include <memory>

#include "absl/status/status.h"
#include "mediapipe/framework/formats/image_format.pb.h"
#include "mediapipe/framework/formats/image_frame.h"
#include "mediapipe/framework/port/gmock.h"
#include "mediapipe/framework/port/gtest.h"
#include "mediapipe/framework/port/status_matchers.h"

using ::testing::HasSubstr;

namespace mediapipe {
namespace {

TEST(TestUtilTest, CompareImageToExpectedWithIdenticalImages) {
  auto image = std::make_unique<ImageFrame>(ImageFormat::SRGB, 10, 10);
  image->SetToZero();
  MP_EXPECT_OK(CompareAndSaveImageOutputDynamic(*image, *image));
}

TEST(TestUtilTest, CompareImageToExpectedWithDifferentImages) {
  auto image1 = std::make_unique<ImageFrame>(ImageFormat::SRGB, 10, 10);
  image1->SetToZero();
  auto image2 = std::make_unique<ImageFrame>(ImageFormat::SRGB, 10, 10);
  image2->SetToZero();
  uint8_t* pixel = image2->MutablePixelData();
  pixel[0] = 255;  // Introduce a difference

  EXPECT_THAT(
      CompareAndSaveImageOutputDynamic(*image1, *image2),
      StatusIs(absl::StatusCode::kInternal, HasSubstr("images differ")));
}

TEST(TestUtilTest, CompareImageToExpectedWithDifferentSizes) {
  auto image1 = std::make_unique<ImageFrame>(ImageFormat::SRGB, 10, 10);
  image1->SetToZero();
  auto image2 = std::make_unique<ImageFrame>(ImageFormat::SRGB, 20, 20);
  image2->SetToZero();

  EXPECT_THAT(
      CompareAndSaveImageOutputDynamic(*image1, *image2),
      StatusIs(absl::StatusCode::kInternal, HasSubstr("image size mismatch")));
}

}  // namespace
}  // namespace mediapipe
