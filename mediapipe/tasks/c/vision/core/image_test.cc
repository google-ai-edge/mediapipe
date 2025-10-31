#include "mediapipe/tasks/c/vision/core/image.h"

#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/flags/flag.h"
#include "absl/strings/string_view.h"
#include "mediapipe/framework/deps/file_path.h"
#include "mediapipe/framework/formats/image.h"
#include "mediapipe/framework/port/gtest.h"
#include "mediapipe/tasks/c/core/mp_status.h"
#include "mediapipe/tasks/c/vision/core/image_frame_util.h"
#include "mediapipe/tasks/c/vision/core/image_test_util.h"

namespace {

using ::mediapipe::Image;
using ::mediapipe::tasks::vision::core::ScopedMpImage;

constexpr char kTestDataDirectory[] = "/mediapipe/tasks/testdata/vision/";
constexpr char kImageFile[] = "portrait.jpg";

std::string GetFullPath(absl::string_view file_name) {
  return mediapipe::file::JoinPath("./", kTestDataDirectory, file_name);
}

TEST(ImageTest, CreateFromUint8Data) {
  const int width = 10;
  const int height = 20;
  const int channels = 3;
  const std::vector<uint8_t> pixel_data(width * height * channels, 128);

  MpImagePtr image_ptr = nullptr;
  MpStatus status = MpImageCreateFromUint8Data(kMpImageFormatSrgb, width,
                                               height, pixel_data.data(),
                                               pixel_data.size(), &image_ptr);

  ASSERT_EQ(status, kMpOk);
  ASSERT_NE(image_ptr, nullptr);
  auto image = ScopedMpImage{image_ptr};

  EXPECT_EQ(MpImageGetWidth(image.get()), width);
  EXPECT_EQ(MpImageGetHeight(image.get()), height);
  EXPECT_EQ(MpImageGetChannels(image.get()), channels);
  EXPECT_EQ(MpImageGetFormat(image.get()), kMpImageFormatSrgb);
  EXPECT_EQ(MpImageGetByteDepth(image.get()), 1);
  EXPECT_FALSE(MpImageIsContiguous(image.get()));
  EXPECT_FALSE(MpImageIsEmpty(image.get()));

  const uint8_t* data = nullptr;
  status = MpImageDataUint8(image.get(), &data);
  ASSERT_EQ(status, kMpOk);
  ASSERT_NE(data, nullptr);
  EXPECT_EQ(data[0], 128);
}

TEST(ImageTest, CreateFromFile) {
  const std::string image_path = GetFullPath(kImageFile);
  MpImagePtr image_ptr = nullptr;
  MpStatus status = MpImageCreateFromFile(image_path.c_str(), &image_ptr);

  ASSERT_EQ(status, kMpOk);
  ASSERT_NE(image_ptr, nullptr);
  auto image = ScopedMpImage{image_ptr};

  // portrait.jpg is 820x1024, 3 channels (SRGB)
  EXPECT_EQ(MpImageGetWidth(image.get()), 820);
  EXPECT_EQ(MpImageGetHeight(image.get()), 1024);
  EXPECT_EQ(MpImageGetChannels(image.get()), 3);
  EXPECT_EQ(MpImageGetFormat(image.get()), kMpImageFormatSrgb);
}

TEST(ImageTest, CreateFromImageFrame) {
  const std::string image_path = GetFullPath(kImageFile);
  MpImagePtr original_image_ptr = nullptr;
  MpStatus status =
      MpImageCreateFromFile(image_path.c_str(), &original_image_ptr);
  ASSERT_EQ(status, kMpOk);
  auto original_image = ScopedMpImage{original_image_ptr};

  MpImagePtr copied_image_ptr = nullptr;
  status = MpImageCreateFromImageFrame(original_image.get(), &copied_image_ptr);
  ASSERT_EQ(status, kMpOk);
  ASSERT_NE(copied_image_ptr, nullptr);

  // portrait.jpg is 820x1024, 3 channels (SRGB)
  auto copied_image = ScopedMpImage{copied_image_ptr};
  EXPECT_EQ(MpImageGetWidth(copied_image.get()), 820);
  EXPECT_EQ(MpImageGetHeight(copied_image.get()), 1024);
  EXPECT_EQ(MpImageGetChannels(copied_image.get()), 3);
  EXPECT_EQ(MpImageGetFormat(copied_image.get()), kMpImageFormatSrgb);
}

TEST(ImageTest, GetValueUint8) {
  const int width = 2;
  const int height = 2;
  const std::vector<uint8_t> pixel_data = {1, 2, 3, 4,  5,  6,
                                           7, 8, 9, 10, 11, 12};
  MpImagePtr image_ptr = nullptr;
  MpStatus status = MpImageCreateFromUint8Data(kMpImageFormatSrgb, width,
                                               height, pixel_data.data(),
                                               pixel_data.size(), &image_ptr);

  ASSERT_EQ(status, kMpOk);
  ASSERT_NE(image_ptr, nullptr);
  auto image = ScopedMpImage{image_ptr};

  int pos1[] = {0, 1, 1};  // row 0, col 1, channel 1
  uint8_t value1 = 0;
  status = MpImageGetValueUint8(image.get(), pos1, 3, &value1);
  EXPECT_EQ(status, kMpOk);
  EXPECT_EQ(value1, 5);

  int pos2[] = {1, 0, 2};  // row 1, col 0, channel 2
  uint8_t value2 = 0;
  status = MpImageGetValueUint8(image.get(), pos2, 3, &value2);
  EXPECT_EQ(status, kMpOk);
  EXPECT_EQ(value2, 9);
}

TEST(ImageTest, GetValueUint8Grayscale) {
  const int width = 2;
  const int height = 2;
  const std::vector<uint8_t> pixel_data = {1, 2, 3, 4};
  MpImagePtr image_ptr = nullptr;
  MpStatus status = MpImageCreateFromUint8Data(kMpImageFormatGray8, width,
                                               height, pixel_data.data(),
                                               pixel_data.size(), &image_ptr);

  ASSERT_EQ(status, kMpOk);
  ASSERT_NE(image_ptr, nullptr);
  auto image = ScopedMpImage{image_ptr};

  int pos1[] = {0, 1};  // row 0, col 1
  uint8_t value1 = 0;
  status = MpImageGetValueUint8(image.get(), pos1, 2, &value1);
  EXPECT_EQ(status, kMpOk);
  EXPECT_EQ(value1, 2);

  int pos2[] = {1, 0};  // row 1, col 0
  uint8_t value2 = 0;
  status = MpImageGetValueUint8(image.get(), pos2, 2, &value2);
  EXPECT_EQ(status, kMpOk);
  EXPECT_EQ(value2, 3);
}

TEST(ImageTest, GetValueUint16) {
  const int width = 2;
  const int height = 2;
  const std::vector<uint16_t> pixel_data = {100, 200, 300, 400,  500,  600,
                                            700, 800, 900, 1000, 1100, 1200};
  MpImagePtr image_ptr = nullptr;
  MpStatus status = MpImageCreateFromUint16Data(kMpImageFormatSrgb48, width,
                                                height, pixel_data.data(),
                                                pixel_data.size(), &image_ptr);

  ASSERT_EQ(status, kMpOk);
  ASSERT_NE(image_ptr, nullptr);
  auto image = ScopedMpImage{image_ptr};

  int pos1[] = {0, 1, 1};  // row 0, col 1, channel 1
  uint16_t value1 = 0;
  status = MpImageGetValueUint16(image.get(), pos1, 3, &value1);
  EXPECT_EQ(status, kMpOk);
  EXPECT_EQ(value1, 500);

  int pos2[] = {1, 0, 0};  // row 1, col 0, channel 0
  uint16_t value2 = 0;
  status = MpImageGetValueUint16(image.get(), pos2, 3, &value2);
  EXPECT_EQ(status, kMpOk);
  EXPECT_EQ(value2, 700);
}

TEST(ImageTest, GetValueFloat32) {
  const int width = 2;
  const int height = 2;
  const std::vector<float> pixel_data = {1.0f, 2.0f, 3.0f, 4.0f,
                                         5.0f, 6.0f, 7.0f, 8.0f};
  MpImagePtr image_ptr = nullptr;
  MpStatus status = MpImageCreateFromFloatData(kMpImageFormatVec32F2, width,
                                               height, pixel_data.data(),
                                               pixel_data.size(), &image_ptr);

  ASSERT_EQ(status, kMpOk);
  ASSERT_NE(image_ptr, nullptr);
  auto image = ScopedMpImage{image_ptr};

  int pos1[] = {0, 1, 1};  // row 0, col 1, channel 1
  float value1 = 0.0f;
  status = MpImageGetValueFloat32(image.get(), pos1, 3, &value1);
  EXPECT_EQ(status, kMpOk);
  EXPECT_FLOAT_EQ(value1, 4.0f);

  int pos2[] = {1, 0, 0};  // row 1, col 0, channel 0
  float value2 = 0.0f;
  status = MpImageGetValueFloat32(image.get(), pos2, 3, &value2);
  EXPECT_EQ(status, kMpOk);
  EXPECT_FLOAT_EQ(value2, 5.0f);
}

TEST(ImageTest, CreateFromUint8DataError) {
  const int width = 10;
  const int height = 20;
  const int channels = 3;

  // one byte too small
  const std::vector<uint8_t> pixel_data(width * height * channels - 1, 128);
  MpImagePtr image_ptr = nullptr;
  MpStatus status = MpImageCreateFromUint8Data(kMpImageFormatSrgb, width,
                                               height, pixel_data.data(),
                                               pixel_data.size(), &image_ptr);

  EXPECT_EQ(status, kMpInvalidArgument);
  EXPECT_EQ(image_ptr, nullptr);
}

TEST(ImageTest, GetDataFromContiguousImageFrame) {
  const std::string filename = mediapipe::file::JoinPath(
      "./", "mediapipe/tasks/testdata/vision/burger.jpg");
  MpImagePtr image_ptr = nullptr;
  MpStatus status = MpImageCreateFromFile(filename.c_str(), &image_ptr);
  ASSERT_NE(image_ptr, nullptr);

  auto image = ScopedMpImage{image_ptr};
  EXPECT_TRUE(MpImageIsContiguous(image.get()));

  const uint8_t* result_data = nullptr;
  status = MpImageDataUint8(image.get(), &result_data);

  ASSERT_EQ(status, kMpOk);
  ASSERT_NE(result_data, nullptr);
}

TEST(ImageTest, GetDataFromNonContiguousImageFrame) {
  const int width = 10;
  const int height = 20;
  const int channels = 3;
  const int pixel_data_size = width * height * channels;
  const std::vector<uint8_t> pixel_data(pixel_data_size, 128);
  MpImagePtr image_ptr = nullptr;
  MpStatus status = MpImageCreateFromUint8Data(kMpImageFormatSrgb, width,
                                               height, pixel_data.data(),
                                               pixel_data.size(), &image_ptr);
  ASSERT_EQ(status, kMpOk);

  auto image = ScopedMpImage{image_ptr};
  EXPECT_FALSE(MpImageIsContiguous(image.get()));

  const uint8_t* result_data = nullptr;
  status = MpImageDataUint8(image.get(), &result_data);
  ASSERT_EQ(status, kMpOk);
  ASSERT_EQ(std::vector<uint8_t>(result_data, result_data + pixel_data_size),
            pixel_data);
}

struct ImageFormatTestData {
  MpImageFormat format;
  int channels;
  int byte_depth;
};

TEST(ImageTest, ImageFormatRoundtrip) {
  const int width = 1;
  const int height = 1;

  const std::vector<ImageFormatTestData> test_data = {
      {kMpImageFormatSrgb, 3, 1},    {kMpImageFormatSrgba, 4, 1},
      {kMpImageFormatGray8, 1, 1},   {kMpImageFormatGray16, 1, 2},
      {kMpImageFormatSrgb48, 3, 2},  {kMpImageFormatSrgba64, 4, 2},
      {kMpImageFormatVec32F1, 1, 4}, {kMpImageFormatVec32F2, 2, 4},
      {kMpImageFormatVec32F4, 4, 4},
  };

  for (const auto& data : test_data) {
    SCOPED_TRACE(data.format);
    MpImagePtr image_ptr = nullptr;
    MpStatus status;
    int pixel_data_size = width * height * data.channels * data.byte_depth;

    if (data.byte_depth == 1) {
      std::vector<uint8_t> pixel_data(width * height * data.channels, 0);
      status = MpImageCreateFromUint8Data(data.format, width, height,
                                          pixel_data.data(), pixel_data_size,
                                          &image_ptr);
    } else if (data.byte_depth == 2) {
      std::vector<uint16_t> pixel_data(width * height * data.channels, 0);
      status = MpImageCreateFromUint16Data(data.format, width, height,
                                           pixel_data.data(), pixel_data_size,
                                           &image_ptr);
    } else if (data.byte_depth == 4) {
      std::vector<float> pixel_data(width * height * data.channels, 0.0f);
      status = MpImageCreateFromFloatData(data.format, width, height,
                                          pixel_data.data(), pixel_data_size,
                                          &image_ptr);
    } else {
      FAIL() << "Unsupported byte depth: " << data.byte_depth;
    }

    ASSERT_EQ(status, kMpOk);
    auto image = ScopedMpImage{image_ptr};
    EXPECT_EQ(MpImageGetFormat(image.get()), data.format);
    EXPECT_EQ(MpImageGetByteDepth(image.get()), data.byte_depth);
    EXPECT_EQ(MpImageGetChannels(image.get()), data.channels);
  }
}

}  // namespace
