
#if !defined(MEDIAPIPE_NO_JNI) && \
    (__ANDROID_API__ >= 26 ||     \
     defined(__ANDROID_UNAVAILABLE_SYMBOLS_ARE_WEAK__))
#include <android/hardware_buffer.h>

#include <cstdint>

#include "absl/algorithm/container.h"
#include "mediapipe/framework/formats/tensor.h"
#include "mediapipe/framework/formats/tensor/views/data_types.h"
#include "mediapipe/gpu/gpu_test_base.h"
#include "mediapipe/gpu/shader_util.h"
#include "tensorflow/lite/delegates/gpu/gl/gl_call.h"
#include "testing/base/public/gunit.h"

// The test creates OpenGL ES buffer, fills the buffer with incrementing values
// 0.0, 0.1, 0.2 etc. with the compute shader on GPU.
// Then the test requests the CPU view and compares the values.
// Float32 and Float16 tests are there.

namespace mediapipe {

using mediapipe::Float16;
using mediapipe::Tensor;

MATCHER_P(NearWithPrecision, precision, "") {
  return std::abs(std::get<0>(arg) - std::get<1>(arg)) < precision;
}

template <typename F = float>
std::vector<F> CreateReferenceData(int num_elements) {
  std::vector<F> reference;
  reference.resize(num_elements);
  for (int i = 0; i < num_elements; i++) {
    reference[i] = static_cast<float>(i) / 10.0f;
  }
  return reference;
}

#if MEDIAPIPE_OPENGL_ES_VERSION >= MEDIAPIPE_OPENGL_ES_31

// Utility function to fill the GPU buffer.
void FillGpuBuffer(GLuint name, std::size_t size,
                   const Tensor::ElementType fmt) {
  std::string shader_source;
  if (fmt == Tensor::ElementType::kFloat32) {
    shader_source = R"( #version 310 es
    precision highp float;
    layout(local_size_x = 1, local_size_y = 1) in;
    layout(std430, binding = 0) buffer Output {float elements[];} output_data;
    void main() {
      uint v = gl_GlobalInvocationID.x * 2u;
      output_data.elements[v] = float(v) / 10.0;
      output_data.elements[v + 1u] = float(v + 1u) / 10.0;
    })";
  } else {
    shader_source = R"( #version 310 es
      precision highp float;
      layout(local_size_x = 1, local_size_y = 1) in;
      layout(std430, binding = 0) buffer Output {float elements[];} output_data;
      void main() {
        uint v = gl_GlobalInvocationID.x;
        uint tmp = packHalf2x16(vec2((float(v)* 2.0 + 0.0) / 10.0,
                                     (float(v) * 2.0 + 1.0) / 10.0));
        output_data.elements[v] = uintBitsToFloat(tmp);
      })";
  }
  GLuint shader;
  MP_ASSERT_OK(TFLITE_GPU_CALL_GL(glCreateShader, &shader, GL_COMPUTE_SHADER));
  const GLchar* sources[] = {shader_source.c_str()};
  MP_ASSERT_OK(TFLITE_GPU_CALL_GL(glShaderSource, shader, 1, sources, nullptr));
  MP_ASSERT_OK(TFLITE_GPU_CALL_GL(glCompileShader, shader));
  GLint is_compiled = 0;
  MP_ASSERT_OK(TFLITE_GPU_CALL_GL(glGetShaderiv, shader, GL_COMPILE_STATUS,
                                  &is_compiled));
  if (is_compiled == GL_FALSE) {
    GLint max_length = 0;
    MP_ASSERT_OK(TFLITE_GPU_CALL_GL(glGetShaderiv, shader, GL_INFO_LOG_LENGTH,
                                    &max_length));
    std::vector<GLchar> error_log(max_length);
    MP_ASSERT_OK(TFLITE_GPU_CALL_GL(glGetShaderInfoLog, shader, max_length,
                                    &max_length, error_log.data()));
    MP_EXPECT_OK(TFLITE_GPU_CALL_GL(glDeleteShader, shader));
    FAIL() << error_log.data();
    return;
  }
  GLuint to_buffer_program;
  MP_ASSERT_OK(TFLITE_GPU_CALL_GL(glCreateProgram, &to_buffer_program));
  MP_ASSERT_OK(TFLITE_GPU_CALL_GL(glAttachShader, to_buffer_program, shader));
  MP_ASSERT_OK(TFLITE_GPU_CALL_GL(glDeleteShader, shader));
  MP_ASSERT_OK(TFLITE_GPU_CALL_GL(glLinkProgram, to_buffer_program));

  MP_ASSERT_OK(
      TFLITE_GPU_CALL_GL(glBindBufferBase, GL_SHADER_STORAGE_BUFFER, 0, name));
  MP_ASSERT_OK(TFLITE_GPU_CALL_GL(glUseProgram, to_buffer_program));
  MP_ASSERT_OK(TFLITE_GPU_CALL_GL(glDispatchCompute, size / 2, 1, 1));
  MP_EXPECT_OK(TFLITE_GPU_CALL_GL(glBindBuffer, GL_SHADER_STORAGE_BUFFER, 0));
  MP_EXPECT_OK(TFLITE_GPU_CALL_GL(glDeleteProgram, to_buffer_program));
}

class TensorAhwbGpuTest : public mediapipe::GpuTestBase {
 public:
};

TEST_F(TensorAhwbGpuTest, TestGpuToCpuFloat32) {
  constexpr size_t num_elements = 20;
  Tensor tensor{Tensor::ElementType::kFloat32, Tensor::Shape({num_elements})};
  {
    // Request Ahwb first to get Ahwb storage allocated internally.
    auto view = tensor.GetAHardwareBufferWriteView();
    ASSERT_NE(view.handle(), nullptr);
    view.SetWritingFinishedFD(-1, [](bool) { return true; });
  }
  RunInGlContext([&tensor] {
    auto ssbo_view = tensor.GetOpenGlBufferWriteView();
    auto ssbo_name = ssbo_view.name();
    ASSERT_GT(ssbo_name, 0);
    FillGpuBuffer(ssbo_name, tensor.shape().num_elements(),
                  tensor.element_type());
  });
  auto ptr = tensor.GetCpuReadView().buffer<float>();
  ASSERT_NE(ptr, nullptr);
  std::vector<float> reference = CreateReferenceData(num_elements);
  EXPECT_THAT(absl::Span<const float>(ptr, num_elements),
              testing::Pointwise(testing::FloatEq(), reference));
}

TEST_F(TensorAhwbGpuTest, TestGpuToCpuFloat16) {
  constexpr size_t num_elements = 20;
  Tensor tensor{Tensor::ElementType::kFloat16, Tensor::Shape({num_elements})};
  {
    // Request Ahwb first to get Ahwb storage allocated internally.
    auto view = tensor.GetAHardwareBufferWriteView();
    ASSERT_NE(view.handle(), nullptr);
    view.SetWritingFinishedFD(/*fd=*/-1, [](bool) { return true; });
  }
  RunInGlContext([&tensor] {
    auto ssbo_view = tensor.GetOpenGlBufferWriteView();
    auto ssbo_name = ssbo_view.name();
    ASSERT_GT(ssbo_name, 0);
    FillGpuBuffer(ssbo_name, tensor.shape().num_elements(),
                  tensor.element_type());
  });
  auto ptr = tensor.GetCpuReadView().buffer<Float16>();
  ASSERT_NE(ptr, nullptr);
  std::vector<Float16> reference = CreateReferenceData<Float16>(num_elements);
  // Precision is set to a reasonable value for Float16.
  EXPECT_THAT(absl::Span<const Float16>(ptr, num_elements),
              testing::Pointwise(NearWithPrecision(0.001), reference));
}

TEST_F(TensorAhwbGpuTest, TestReplacingCpuByAhwb) {
  // Request the CPU view to get the memory to be allocated.
  // Request Ahwb view then to transform the storage into Ahwb.
  constexpr size_t num_elements = 20;
  Tensor tensor{Tensor::ElementType::kFloat32, Tensor::Shape({num_elements})};
  {
    auto ptr = tensor.GetCpuWriteView().buffer<float>();
    ASSERT_NE(ptr, nullptr);
    for (int i = 0; i < num_elements; i++) {
      ptr[i] = static_cast<float>(i) / 10.0f;
    }
  }
  {
    auto view = tensor.GetAHardwareBufferReadView();
    ASSERT_NE(view.handle(), nullptr);
    view.SetReadingFinishedFunc([](bool) { return true; });
  }
  auto ptr = tensor.GetCpuReadView().buffer<float>();
  ASSERT_NE(ptr, nullptr);
  std::vector<float> reference = CreateReferenceData(num_elements);
  EXPECT_THAT(absl::Span<const float>(ptr, num_elements),
              testing::Pointwise(testing::FloatEq(), reference));
}

TEST_F(TensorAhwbGpuTest, TestReplacingGpuByAhwb) {
  // Request the GPU view to get the ssbo allocated internally.
  // Request Ahwb view then to transform the storage into Ahwb.
  constexpr size_t num_elements = 20;
  Tensor tensor{Tensor::ElementType::kFloat32, Tensor::Shape({num_elements})};
  RunInGlContext([&tensor] {
    auto ssbo_view = tensor.GetOpenGlBufferWriteView();
    auto ssbo_name = ssbo_view.name();
    ASSERT_GT(ssbo_name, 0);
    FillGpuBuffer(ssbo_name, tensor.shape().num_elements(),
                  tensor.element_type());
  });
  {
    auto view = tensor.GetAHardwareBufferReadView();
    ASSERT_NE(view.handle(), nullptr);
    view.SetReadingFinishedFunc([](bool) { return true; });
  }
  auto ptr = tensor.GetCpuReadView().buffer<float>();
  ASSERT_NE(ptr, nullptr);
  std::vector<float> reference = CreateReferenceData(num_elements);
  EXPECT_THAT(absl::Span<const float>(ptr, num_elements),
              testing::Pointwise(testing::FloatEq(), reference));
}

std::vector<float> ReadGlBufferView(const Tensor::OpenGlBufferView& view,
                                    int num_elements) {
  glBindBuffer(GL_SHADER_STORAGE_BUFFER, view.name());
  int bytes = num_elements * sizeof(float);
  void* ptr =
      glMapBufferRange(GL_SHADER_STORAGE_BUFFER, 0, bytes, GL_MAP_READ_BIT);
  ABSL_CHECK(ptr) << "glMapBufferRange failed: " << glGetError();

  std::vector<float> data;
  data.resize(num_elements);
  std::memcpy(data.data(), ptr, bytes);
  glUnmapBuffer(GL_SHADER_STORAGE_BUFFER);
  return data;
}

TEST_F(TensorAhwbGpuTest, TestGetOpenGlBufferReadViewNoAhwb) {
  constexpr size_t kNumElements = 20;
  std::vector<float> reference = CreateReferenceData(kNumElements);

  Tensor tensor(Tensor::ElementType::kFloat32, Tensor::Shape({kNumElements}));
  {
    // Populate tensor on CPU and make sure view is destroyed
    absl::c_copy(reference, tensor.GetCpuWriteView().buffer<float>());
  }

  RunInGlContext([&] {
    // Triggers conversion to GL buffer.
    auto ssbo_view = tensor.GetOpenGlBufferReadView();
    ASSERT_NE(ssbo_view.name(), 0);
    // ssbo_read_ must NOT be populated, as there's no AHWB associated with
    // GL buffer
    ASSERT_EQ(ssbo_view.ssbo_read_, nullptr);

    std::vector<float> output = ReadGlBufferView(ssbo_view, kNumElements);
    EXPECT_THAT(output, testing::Pointwise(testing::FloatEq(), reference));
  });
}

TEST_F(TensorAhwbGpuTest, TestGetOpenGlBufferReadViewAhwbFromCpu) {
  constexpr size_t kNumElements = 20;
  std::vector<float> reference = CreateReferenceData(kNumElements);

  Tensor tensor(Tensor::ElementType::kFloat32, Tensor::Shape({kNumElements}));
  {
    // Populate tensor on CPU and make sure view is destroyed
    absl::c_copy(reference, tensor.GetCpuWriteView().buffer<float>());
  }

  {
    // Make tensor to allocate ahwb and make sure view is destroyed.
    ASSERT_NE(tensor.GetAHardwareBufferReadView().handle(), nullptr);
  }

  RunInGlContext([&] {
    // Triggers conversion to GL buffer.
    auto ssbo_view = tensor.GetOpenGlBufferReadView();
    ASSERT_NE(ssbo_view.name(), 0);
    // ssbo_read_ must be populated, so during view destruction it's set
    // properly for further AHWB destruction
    ASSERT_NE(ssbo_view.ssbo_read_, nullptr);

    std::vector<float> output = ReadGlBufferView(ssbo_view, kNumElements);
    EXPECT_THAT(output, testing::Pointwise(testing::FloatEq(), reference));
  });
}

TEST_F(TensorAhwbGpuTest, TestGetOpenGlBufferReadViewAhwbFromGpu) {
  constexpr size_t kNumElements = 20;
  std::vector<float> reference = CreateReferenceData(kNumElements);

  Tensor tensor(Tensor::ElementType::kFloat32, Tensor::Shape({kNumElements}));
  {
    // Make tensor to allocate ahwb and make sure view is destroyed.
    ASSERT_NE(tensor.GetAHardwareBufferWriteView().handle(), nullptr);
  }

  RunInGlContext([&] {
    FillGpuBuffer(tensor.GetOpenGlBufferWriteView().name(),
                  tensor.shape().num_elements(), tensor.element_type());
  });

  RunInGlContext([&] {
    // Triggers conversion to GL buffer.
    auto ssbo_view = tensor.GetOpenGlBufferReadView();
    ASSERT_NE(ssbo_view.name(), 0);
    // ssbo_read_ must be populated, so during view destruction it's set
    // properly for further AHWB destruction
    ASSERT_NE(ssbo_view.ssbo_read_, nullptr);

    std::vector<float> output = ReadGlBufferView(ssbo_view, kNumElements);
    EXPECT_THAT(output, testing::Pointwise(testing::FloatEq(), reference));
  });
}

#endif  // MEDIAPIPE_OPENGL_ES_VERSION >= MEDIAPIPE_OPENGL_ES_31
}  // namespace mediapipe

#endif  // !defined(MEDIAPIPE_NO_JNI) && (__ANDROID_API__ >= 26 ||
        // defined(__ANDROID_UNAVAILABLE_SYMBOLS_ARE_WEAK__))
