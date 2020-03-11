#include "mediapipe/util/tflite/tensor_buffer.h"

#include "mediapipe/framework/port/gmock.h"
#include "mediapipe/framework/port/gtest.h"

namespace mediapipe {

TEST(Cpu, BasicTest) {
  TensorBuffer tb;
  TfLiteTensor tfl_tb;
  tb = TensorBuffer(tfl_tb);
  EXPECT_FALSE(tb.UsesGpu());
}

#if !defined(MEDIAPIPE_DISABLE_GPU)
TEST(Gpu, BasicTest) {
  TensorBuffer tb;
  std::shared_ptr<tflite::gpu::gl::GlBuffer> tfg_tb =
      TensorBuffer::CreateGlBuffer(nullptr);
  tb = TensorBuffer(tfg_tb);
  EXPECT_TRUE(tb.UsesGpu());
}
#endif  // !MEDIAPIPE_DISABLE_GPU

}  // namespace mediapipe

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
