#include "mediapipe/framework/formats/tensor_opencv.h"

#include <cstdint>
#include <vector>

#include "absl/status/status.h"
#include "mediapipe/framework/formats/tensor.h"
#include "mediapipe/framework/port/gmock.h"
#include "mediapipe/framework/port/gtest.h"
#include "mediapipe/framework/port/status_matchers.h"

namespace mediapipe::formats {
namespace {

using ::testing::HasSubstr;

template <typename T>
Tensor::ElementType GetElementType() {
  if constexpr (std::is_same_v<T, float>) {
    return Tensor::ElementType::kFloat32;
  } else if constexpr (std::is_same_v<T, uint8_t>) {
    return Tensor::ElementType::kUInt8;
  } else if constexpr (std::is_same_v<T, int32_t>) {
    return Tensor::ElementType::kInt32;
  } else {
    static_assert(std::is_void_v<T>, "Unsupported type");
  }
}

// Creates a tensor with values [0, 1, 2, ..., num_elements - 1].
template <typename T>
Tensor MakeTensor(std::vector<int> dims) {
  Tensor tensor(GetElementType<T>(), Tensor::Shape(dims));
  auto view = tensor.GetCpuWriteView();
  int num_elements = tensor.shape().num_elements();
  for (int n = 0; n < num_elements; ++n) {
    view.buffer<T>()[n] = n;
  }
  return tensor;
}

TEST(TensorOpenCVTest, ViewsFullTensor) {
  Tensor tensor = MakeTensor<float>({2, 3, 4});
  auto view = tensor.GetCpuReadView();
  MP_ASSERT_OK_AND_ASSIGN(cv::Mat mat, MatView(tensor, view));
  EXPECT_EQ(mat.dims, 2);
  ASSERT_EQ(mat.rows, 2);
  ASSERT_EQ(mat.cols, 3);
  ASSERT_EQ(mat.channels(), 4);
  for (int i = 0; i < mat.rows; ++i) {
    for (int j = 0; j < mat.cols; ++j) {
      for (int k = 0; k < mat.channels(); ++k) {
        EXPECT_EQ(mat.ptr<float>(i, j)[k], i * 12 + j * 4 + k);
      }
    }
  }
}

TEST(TensorOpenCVTest, ViewsTensorSlicedToTwoDimensions) {
  Tensor tensor = MakeTensor<float>({2, 3, 4});
  auto view = tensor.GetCpuReadView();
  MP_ASSERT_OK_AND_ASSIGN(cv::Mat mat, MatView(tensor, view, {1, -1, -1}));
  ASSERT_EQ(mat.dims, 2);
  ASSERT_EQ(mat.rows, 3);
  ASSERT_EQ(mat.cols, 1);
  ASSERT_EQ(mat.channels(), 4);
  for (int n = 0; n < mat.rows; ++n) {
    for (int c = 0; c < mat.channels(); ++c) {
      EXPECT_EQ(mat.ptr<float>(n, 0)[c], 12 + n * 4 + c);
    }
  }
}

TEST(TensorOpenCVTest, ViewsTensorSlicedToOneDimension) {
  Tensor tensor = MakeTensor<float>({2, 3, 4});
  auto view = tensor.GetCpuReadView();
  MP_ASSERT_OK_AND_ASSIGN(cv::Mat mat, MatView(tensor, view, {1, 1, -1}));
  ASSERT_EQ(mat.dims, 0);
  ASSERT_EQ(mat.rows, 0);
  ASSERT_EQ(mat.cols, 0);
  ASSERT_EQ(mat.channels(), 4);
  for (int c = 0; c < mat.channels(); ++c) {
    EXPECT_EQ(mat.ptr<float>(0)[c], 16 + c);
  }
}

TEST(TensorOpenCVTest, IntTensor) {
  Tensor tensor = MakeTensor<int32_t>({2, 3, 4});
  auto view = tensor.GetCpuReadView();
  MP_ASSERT_OK_AND_ASSIGN(cv::Mat mat, MatView(tensor, view));
  EXPECT_EQ(mat.dims, 2);
  ASSERT_EQ(mat.rows, 2);
  ASSERT_EQ(mat.cols, 3);
  ASSERT_EQ(mat.channels(), 4);
  for (int i = 0; i < mat.rows; ++i) {
    for (int j = 0; j < mat.cols; ++j) {
      for (int k = 0; k < mat.channels(); ++k) {
        EXPECT_EQ(mat.ptr<int32_t>(i, j)[k], i * 12 + j * 4 + k);
      }
    }
  }
}

TEST(TensorOpenCVTest, Uint8Tensor) {
  Tensor tensor = MakeTensor<uint8_t>({2, 3, 4});
  auto view = tensor.GetCpuReadView();
  MP_ASSERT_OK_AND_ASSIGN(cv::Mat mat, MatView(tensor, view));
  EXPECT_EQ(mat.dims, 2);
  ASSERT_EQ(mat.rows, 2);
  ASSERT_EQ(mat.cols, 3);
  ASSERT_EQ(mat.channels(), 4);
  for (int i = 0; i < mat.rows; ++i) {
    for (int j = 0; j < mat.cols; ++j) {
      for (int k = 0; k < mat.channels(); ++k) {
        EXPECT_EQ(mat.ptr<uint8_t>(i, j)[k], i * 12 + j * 4 + k);
      }
    }
  }
}

TEST(TensorOpenCVTest, SlicingLastDimensionFails) {
  // cv::Mat's last dimension must be consecutive, unfortunately.
  Tensor tensor = MakeTensor<float>({2, 3, 4});
  auto view = tensor.GetCpuReadView();
  EXPECT_THAT(MatView(tensor, view, {-1, -1, 2}),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("slicing the last dimension")));
}

TEST(TensorOpenCVTest, SliceDimensionOutOfBoundsFails) {
  Tensor tensor = MakeTensor<float>({2, 3, 4});
  auto view = tensor.GetCpuReadView();
  EXPECT_THAT(
      MatView(tensor, view, {-1, 3, -1}),
      StatusIs(absl::StatusCode::kInvalidArgument, HasSubstr("out of bounds")));
  EXPECT_THAT(
      MatView(tensor, view, {-1, -2, -1}),
      StatusIs(absl::StatusCode::kInvalidArgument, HasSubstr("out of bounds")));
}

TEST(TensorOpenCVTest, BadSliceSizeFails) {
  Tensor tensor = MakeTensor<float>({2, 3, 4});
  auto view = tensor.GetCpuReadView();

  EXPECT_THAT(MatView(tensor, view, {-1, -1}),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("number of elements")));
}

}  // namespace

}  // namespace mediapipe::formats
