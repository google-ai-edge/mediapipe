#include "mediapipe/framework/formats/tensor.h"

#include "mediapipe/framework/port/gmock.h"
#include "mediapipe/framework/port/gtest.h"
#if !MEDIAPIPE_DISABLE_GPU
#include "mediapipe/gpu/gl_calculator_helper.h"
#include "mediapipe/gpu/gpu_buffer_format.h"
#endif

namespace mediapipe {

TEST(General, TestDimensions) {
  Tensor t1(Tensor::ElementType::kFloat32, Tensor::Shape{1, 2, 3, 4});
  EXPECT_EQ(t1.shape().num_elements(), 1 * 2 * 3 * 4);

  Tensor t2(Tensor::ElementType::kFloat16, Tensor::Shape{4, 3, 2, 3});
  EXPECT_EQ(t2.shape().num_elements(), 4 * 3 * 2 * 3);
}

TEST(General, TestDataTypes) {
  Tensor t1(Tensor::ElementType::kFloat32, Tensor::Shape{1, 2, 3, 4});
  EXPECT_EQ(t1.bytes(), t1.shape().num_elements() * sizeof(float));

  Tensor t2(Tensor::ElementType::kFloat16, Tensor::Shape{4, 3, 2, 3});
  EXPECT_EQ(t2.bytes(), t2.shape().num_elements() * 2);
}

TEST(Cpu, TestMemoryAllocation) {
  Tensor t1(Tensor::ElementType::kFloat32, Tensor::Shape{4, 3, 2, 3});
  auto v1 = t1.GetCpuWriteView();
  float* f1 = v1.buffer<float>();
  EXPECT_NE(f1, nullptr);
}

TEST(Cpu, TestTensorMove) {
  Tensor t1(Tensor::ElementType::kFloat32, Tensor::Shape{4, 3, 2, 3});
  void* p1 = t1.GetCpuWriteView().buffer<float>();
  EXPECT_NE(p1, nullptr);
  Tensor t2(std::move(t1));
  EXPECT_NE(t2.bytes(), 0);
  EXPECT_EQ(t1.bytes(), 0);  // NOLINT
  void* p2 = t2.GetCpuWriteView().buffer<float>();
  EXPECT_EQ(p1, p2);
}

TEST(Cpu, TestViewMove) {
  Tensor t(Tensor::ElementType::kFloat32, Tensor::Shape{4, 3, 2, 3});
  auto v1 = t.GetCpuWriteView();
  auto p1 = v1.buffer<float>();
  EXPECT_NE(p1, nullptr);
  Tensor::CpuWriteView v2(std::move(v1));
  auto p2 = v2.buffer<float>();
  EXPECT_EQ(p1, p2);
  EXPECT_EQ(v1.buffer<float>(), nullptr);  // NOLINT
}

}  // namespace mediapipe

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
