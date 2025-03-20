#include "mediapipe/framework/formats/hardware_buffer.h"

#include <android/hardware_buffer.h>

#include <memory>

#include "base/logging.h"
#include "mediapipe/framework/port/status_macros.h"
#include "testing/base/public/gmock.h"
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

TEST(HardwareBufferTest, ShouldConstructFromExistingHardwareBuffer) {
  AHardwareBuffer* a_hardware_buffer_test = nullptr;
  const HardwareBufferSpec spec = GetTestHardwareBufferSpec(/*size_bytes=*/123);
  if (__builtin_available(android 26, *)) {
    AHardwareBuffer_Desc desc = {
        .width = spec.width,
        .height = spec.height,
        .layers = spec.layers,
        .format = spec.format,
        .usage = spec.usage,
    };
    const int error = AHardwareBuffer_allocate(&desc, &a_hardware_buffer_test);
    ABSL_CHECK(!error) << "AHardwareBuffer_allocate failed: " << error;
  }

  MP_ASSERT_OK_AND_ASSIGN(
      HardwareBuffer hardware_buffer,
      HardwareBuffer::WrapAndAcquireAHardwareBuffer(a_hardware_buffer_test));
  EXPECT_TRUE(hardware_buffer.IsValid());
  EXPECT_FALSE(hardware_buffer.IsLocked());
  EXPECT_EQ(hardware_buffer.spec(), spec);
  EXPECT_EQ(hardware_buffer.GetAHardwareBuffer(), a_hardware_buffer_test);
}

TEST(HardwareBufferTest, ShouldConstructValidAHardwareBuffer) {
  MP_ASSERT_OK_AND_ASSIGN(
      HardwareBuffer hardware_buffer,
      HardwareBuffer::Create(GetTestHardwareBufferSpec(/*size_bytes=*/123)));
  EXPECT_NE(hardware_buffer.GetAHardwareBuffer(), nullptr);
  EXPECT_TRUE(hardware_buffer.IsValid());
}

TEST(HardwareBufferTest, ShouldResetValidAHardwareBuffer) {
  MP_ASSERT_OK_AND_ASSIGN(
      HardwareBuffer hardware_buffer,
      HardwareBuffer::Create(GetTestHardwareBufferSpec(/*size_bytes=*/123)));
  EXPECT_TRUE(hardware_buffer.IsValid());
  EXPECT_NE(*hardware_buffer.Lock(
                HardwareBufferSpec::AHARDWAREBUFFER_USAGE_CPU_WRITE_RARELY),
            nullptr);
  EXPECT_TRUE(hardware_buffer.IsLocked());

  hardware_buffer.Reset();

  EXPECT_FALSE(hardware_buffer.IsValid());
  EXPECT_FALSE(hardware_buffer.IsLocked());
}

TEST(HardwareBufferTest, ShouldAllocateRequestedBufferSize) {
  constexpr int kBufferSize = 123;
  const HardwareBufferSpec spec = GetTestHardwareBufferSpec(kBufferSize);
  MP_ASSERT_OK_AND_ASSIGN(HardwareBuffer hardware_buffer,
                          HardwareBuffer::Create(spec));

  EXPECT_TRUE(hardware_buffer.IsValid());
  if (__builtin_available(android 26, *)) {
    AHardwareBuffer_Desc desc;
    AHardwareBuffer_describe(hardware_buffer.GetAHardwareBuffer(), &desc);
    EXPECT_EQ(desc.width, spec.width);
    EXPECT_EQ(desc.height, spec.height);
    EXPECT_EQ(desc.layers, spec.layers);
    EXPECT_EQ(desc.format, spec.format);
    EXPECT_EQ(desc.usage, spec.usage);
  }
  EXPECT_EQ(hardware_buffer.spec().width, spec.width);
  EXPECT_EQ(hardware_buffer.spec().height, spec.height);
  EXPECT_EQ(hardware_buffer.spec().layers, spec.layers);
  EXPECT_EQ(hardware_buffer.spec().format, spec.format);
  EXPECT_EQ(hardware_buffer.spec().usage, spec.usage);
}

TEST(HardwareBufferTest, ShouldSupportMoveConstructor) {
  constexpr int kBufferSize = 123;
  const auto spec = GetTestHardwareBufferSpec(kBufferSize);
  MP_ASSERT_OK_AND_ASSIGN(HardwareBuffer hardware_buffer_a,
                          HardwareBuffer::Create(spec));
  EXPECT_TRUE(hardware_buffer_a.IsValid());
  void* const ahardware_buffer_ptr_a = hardware_buffer_a.GetAHardwareBuffer();
  EXPECT_NE(ahardware_buffer_ptr_a, nullptr);
  EXPECT_FALSE(hardware_buffer_a.IsLocked());
  MP_ASSERT_OK_AND_ASSIGN(
      void* const hardware_buffer_a_locked_ptr,
      hardware_buffer_a.Lock(
          HardwareBufferSpec::AHARDWAREBUFFER_USAGE_CPU_WRITE_RARELY));
  EXPECT_NE(hardware_buffer_a_locked_ptr, nullptr);
  EXPECT_TRUE(hardware_buffer_a.IsLocked());

  HardwareBuffer hardware_buffer_b(std::move(hardware_buffer_a));

  EXPECT_FALSE(hardware_buffer_a.IsValid());
  EXPECT_FALSE(hardware_buffer_a.IsLocked());
  void* const ahardware_buffer_ptr_b = hardware_buffer_b.GetAHardwareBuffer();
  EXPECT_EQ(ahardware_buffer_ptr_a, ahardware_buffer_ptr_b);
  EXPECT_TRUE(hardware_buffer_b.IsValid());
  EXPECT_TRUE(hardware_buffer_b.IsLocked());

  EXPECT_EQ(hardware_buffer_a.spec(), HardwareBufferSpec());
  EXPECT_EQ(hardware_buffer_b.spec(), spec);

  MP_ASSERT_OK(hardware_buffer_b.Unlock());
}

TEST(HardwareBufferTest, ShouldSupportReadWrite) {
  constexpr std::string_view kTestString = "TestString";
  constexpr int kBufferSize = kTestString.size();
  MP_ASSERT_OK_AND_ASSIGN(
      HardwareBuffer hardware_buffer,
      HardwareBuffer::Create(GetTestHardwareBufferSpec(kBufferSize)));

  // Write test string.
  MP_ASSERT_OK_AND_ASSIGN(
      void* const write_ptr,
      hardware_buffer.Lock(
          HardwareBufferSpec::AHARDWAREBUFFER_USAGE_CPU_WRITE_RARELY));
  memcpy(write_ptr, kTestString.data(), kBufferSize);
  MP_ASSERT_OK(hardware_buffer.Unlock());

  // Read test string.
  MP_ASSERT_OK_AND_ASSIGN(
      void* const read_ptr,
      hardware_buffer.Lock(
          HardwareBufferSpec::AHARDWAREBUFFER_USAGE_CPU_READ_RARELY));
  EXPECT_EQ(memcmp(read_ptr, kTestString.data(), kBufferSize), 0);
  MP_ASSERT_OK(hardware_buffer.Unlock());
}

}  // namespace

}  // namespace mediapipe
