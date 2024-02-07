#include "mediapipe/framework/formats/hardware_buffer_pool.h"

#include <cstdint>

#include "mediapipe/framework/formats/hardware_buffer.h"
#include "mediapipe/framework/port/status_matchers.h"
#include "mediapipe/gpu/multi_pool.h"
#include "testing/base/public/gunit.h"

namespace mediapipe {
namespace {

HardwareBufferSpec GetTestHardwareBufferSpec(uint32_t size_bytes) {
  HardwareBufferSpec hardware_buffer_spec;
  hardware_buffer_spec.width = size_bytes;
  hardware_buffer_spec.height = 1;
  hardware_buffer_spec.layers = 1;
  hardware_buffer_spec.format = HardwareBufferSpec::AHARDWAREBUFFER_FORMAT_BLOB;
  hardware_buffer_spec.usage =
      HardwareBufferSpec::AHARDWAREBUFFER_USAGE_CPU_WRITE_RARELY |
      HardwareBufferSpec::AHARDWAREBUFFER_USAGE_CPU_WRITE_OFTEN |
      HardwareBufferSpec::AHARDWAREBUFFER_USAGE_CPU_READ_OFTEN |
      HardwareBufferSpec::AHARDWAREBUFFER_USAGE_GPU_DATA_BUFFER;
  return hardware_buffer_spec;
}

MultiPoolOptions GetTestMultiPoolOptions() {
  MultiPoolOptions options;
  options.min_requests_before_pool = 0;
  return options;
}

TEST(HardwareBufferPoolTest, ShouldPoolHardwareBuffer) {
  HardwareBufferPool hardware_buffer_pool(GetTestMultiPoolOptions());

  const HardwareBufferSpec hardware_buffer_spec =
      GetTestHardwareBufferSpec(/*size_bytes=*/123);

  HardwareBuffer* hardware_buffer_ptr = nullptr;
  // First request instantiates new HardwareBuffer.
  {
    MP_ASSERT_OK_AND_ASSIGN(
        auto hardware_buffer,
        hardware_buffer_pool.GetBuffer(hardware_buffer_spec));
    hardware_buffer_ptr = hardware_buffer.get();
  }
  // Second request returns same HardwareBuffer.
  {
    MP_ASSERT_OK_AND_ASSIGN(
        auto hardware_buffer,
        hardware_buffer_pool.GetBuffer(hardware_buffer_spec));
    EXPECT_EQ(hardware_buffer.get(), hardware_buffer_ptr);
  }
}

TEST(HardwareBufferPoolTest, ShouldReturnNewHardwareBuffer) {
  HardwareBufferPool hardware_buffer_pool(GetTestMultiPoolOptions());

  HardwareBuffer* hardware_buffer_ptr = nullptr;
  // First request instantiates new HardwareBuffer.
  {
    MP_ASSERT_OK_AND_ASSIGN(auto hardware_buffer,
                            hardware_buffer_pool.GetBuffer(
                                GetTestHardwareBufferSpec(/*size_bytes=*/123)));
    hardware_buffer_ptr = hardware_buffer.get();
    EXPECT_NE(hardware_buffer_ptr, nullptr);
  }
  // Second request with different size returns new HardwareBuffer.
  {
    MP_ASSERT_OK_AND_ASSIGN(auto hardware_buffer,
                            hardware_buffer_pool.GetBuffer(
                                GetTestHardwareBufferSpec(/*size_bytes=*/567)));
    EXPECT_NE(hardware_buffer.get(), hardware_buffer_ptr);
  }
}

}  // namespace
}  // namespace mediapipe
