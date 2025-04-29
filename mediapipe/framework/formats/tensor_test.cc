#include "mediapipe/framework/formats/tensor.h"

#include <cstdint>
#include <cstring>
#include <utility>
#include <vector>

#include "mediapipe/framework/port.h"  // IWYU pragma: keep
#include "mediapipe/framework/port/gmock.h"
#include "mediapipe/framework/port/gtest.h"
#if !MEDIAPIPE_DISABLE_GPU
#include "mediapipe/gpu/gl_calculator_helper.h"
#include "mediapipe/gpu/gl_context.h"  // IWYU pragma: keep
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

  Tensor t_char(Tensor::ElementType::kChar, Tensor::Shape{4});
  EXPECT_EQ(t_char.bytes(), t_char.shape().num_elements() * sizeof(char));

  Tensor t_bool(Tensor::ElementType::kBool, Tensor::Shape{2, 3});
  EXPECT_EQ(t_bool.bytes(), t_bool.shape().num_elements() * sizeof(bool));

  Tensor t_int64(Tensor::ElementType::kInt64, Tensor::Shape{2, 3});
  EXPECT_EQ(t_int64.bytes(), t_int64.shape().num_elements() * sizeof(int64_t));
}

TEST(General, TestDynamic) {
  Tensor t1(Tensor::ElementType::kFloat32, Tensor::Shape({1, 2, 3, 4}, true));
  EXPECT_EQ(t1.shape().num_elements(), 1 * 2 * 3 * 4);
  EXPECT_TRUE(t1.shape().is_dynamic);

  std::vector<int> t2_dims = {4, 3, 2, 3};
  Tensor t2(Tensor::ElementType::kFloat16, Tensor::Shape(t2_dims, true));
  EXPECT_EQ(t2.shape().num_elements(), 4 * 3 * 2 * 3);
  EXPECT_TRUE(t2.shape().is_dynamic);
}

TEST(Cpu, TestMemoryAllocation) {
  Tensor t1(Tensor::ElementType::kFloat32, Tensor::Shape{4, 3, 2, 3});
  auto v1 = t1.GetCpuWriteView();
  float* f1 = v1.buffer<float>();
  EXPECT_NE(f1, nullptr);
}

TEST(Cpu, TestAlignedMemoryAllocation) {
  for (int i = 0; i < 8; ++i) {
    const int alignment_bytes = sizeof(void*) << i;
    Tensor t1(Tensor::ElementType::kFloat32, Tensor::Shape{4, 3, 2, 3},
              /*memory_manager=*/nullptr, alignment_bytes);
    auto v1 = t1.GetCpuWriteView();
    void* data_ptr = v1.buffer<void>();
    EXPECT_EQ(reinterpret_cast<uintptr_t>(data_ptr) % alignment_bytes, 0);
    memset(data_ptr, 0, t1.bytes());
  }
}

TEST(Cpu, TestTensorMove) {
  Tensor t1(Tensor::ElementType::kFloat32, Tensor::Shape{4, 3, 2, 3},
            Tensor::QuantizationParameters(0.5, 127));
  void* p1 = t1.GetCpuWriteView().buffer<float>();
  EXPECT_NE(p1, nullptr);
  Tensor t2(std::move(t1));
  EXPECT_NE(t2.bytes(), 0);
  EXPECT_EQ(t1.bytes(), 0);  // NOLINT
  void* p2 = t2.GetCpuWriteView().buffer<float>();
  EXPECT_EQ(p1, p2);
  EXPECT_EQ(t1.quantization_parameters().scale,
            t2.quantization_parameters().scale);
  EXPECT_EQ(t1.quantization_parameters().zero_point,
            t2.quantization_parameters().zero_point);
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
