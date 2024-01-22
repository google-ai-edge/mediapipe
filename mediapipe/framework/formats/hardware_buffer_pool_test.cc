#include "mediapipe/framework/formats/hardware_buffer_pool.h"

#include <cstdint>

#include "mediapipe/framework/formats/hardware_buffer.h"
#include "testing/base/public/gunit.h"

namespace mediapipe {
namespace {

HardwareBufferSpec GetTestHardwareBufferSpec(uint32_t size_bytes) {
  return {.width = size_bytes,
          .height = 1,
          .layers = 1,
          .format = HardwareBufferSpec::AHARDWAREBUFFER_FORMAT_BLOB,
          .usage = HardwareBufferSpec::AHARDWAREBUFFER_USAGE_CPU_WRITE_RARELY |
                   HardwareBufferSpec::AHARDWAREBUFFER_USAGE_CPU_WRITE_OFTEN |
                   HardwareBufferSpec::AHARDWAREBUFFER_USAGE_CPU_READ_OFTEN |
                   HardwareBufferSpec::AHARDWAREBUFFER_USAGE_GPU_DATA_BUFFER};
}

TEST(HardwareBufferPoolTest, ShouldPoolHardwareBuffer) {
  HardwareBufferPool hardware_buffer_pool({.min_requests_before_pool = 0});

  const HardwareBufferSpec hardware_buffer_spec =
      GetTestHardwareBufferSpec(/*size_bytes=*/123);

  HardwareBuffer* hardware_buffer_ptr = nullptr;
  // First request instantiates new HardwareBuffer.
  {
    auto hardware_buffer = hardware_buffer_pool.GetBuffer(hardware_buffer_spec);
    hardware_buffer_ptr = hardware_buffer.get();
  }
  // Second request returns same HardwareBuffer.
  {
    auto hardware_buffer = hardware_buffer_pool.GetBuffer(hardware_buffer_spec);
    EXPECT_EQ(hardware_buffer.get(), hardware_buffer_ptr);
  }
}

TEST(HardwareBufferPoolTest, ShouldReturnNewHardwareBuffer) {
  HardwareBufferPool hardware_buffer_pool({.min_requests_before_pool = 0});

  HardwareBuffer* hardware_buffer_ptr = nullptr;
  // First request instantiates new HardwareBuffer.
  {
    auto hardware_buffer = hardware_buffer_pool.GetBuffer(
        GetTestHardwareBufferSpec(/*size_bytes=*/123));
    hardware_buffer_ptr = hardware_buffer.get();
    EXPECT_NE(hardware_buffer_ptr, nullptr);
  }
  // Second request with different size returns new HardwareBuffer.
  {
    auto hardware_buffer = hardware_buffer_pool.GetBuffer(
        GetTestHardwareBufferSpec(/*size_bytes=*/567));
    EXPECT_NE(hardware_buffer.get(), hardware_buffer_ptr);
  }
}

}  // namespace
}  // namespace mediapipe
