#include "mediapipe/tasks/cc/core/base_options.h"

#include <memory>
#include <optional>
#include <string>
#include <variant>

#include "mediapipe/calculators/tensor/inference_calculator.pb.h"
#include "mediapipe/framework/port/gmock.h"
#include "mediapipe/framework/port/gtest.h"
#include "mediapipe/tasks/cc/core/proto/acceleration.pb.h"
#include "mediapipe/tasks/cc/core/proto/external_file.pb.h"
#include "mediapipe/tasks/cc/core/utils.h"

constexpr char kTestModelBundlePath[] =
    "mediapipe/tasks/testdata/core/dummy_gesture_recognizer.task";
constexpr char kCachedModelDir[] = "/data/local/tmp";
constexpr char kModelToken[] = "dummy_model_token";

namespace mediapipe {
namespace tasks {
namespace core {
namespace {

TEST(BaseOptionsTest, ConvertBaseOptionsToProtoWithFile) {
  BaseOptions base_options;
  base_options.model_asset_buffer =
      std::make_unique<std::string>(LoadBinaryContent(kTestModelBundlePath));
  proto::BaseOptions proto = ConvertBaseOptionsToProto(&base_options);
  EXPECT_TRUE(proto.has_model_asset());
  EXPECT_TRUE(proto.model_asset().has_file_content());
}

TEST(BaseOptionsTest, ConvertBaseOptionsToProtoWithAcceleration) {
  BaseOptions base_options;
  proto::BaseOptions proto = ConvertBaseOptionsToProto(&base_options);
  EXPECT_TRUE(proto.acceleration().has_tflite());

  base_options.delegate = BaseOptions::Delegate::GPU;
  proto = ConvertBaseOptionsToProto(&base_options);
  EXPECT_TRUE(proto.acceleration().has_gpu());

  base_options.delegate = BaseOptions::Delegate::EDGETPU_NNAPI;
  proto = ConvertBaseOptionsToProto(&base_options);
  EXPECT_EQ(proto.acceleration().nnapi().accelerator_name(), "google-edgetpu");
}

TEST(DelegateOptionsTest, SucceedCpuOptions) {
  BaseOptions base_options;
  base_options.delegate = BaseOptions::Delegate::CPU;
  BaseOptions::CpuOptions cpu_options;
  base_options.delegate_options = cpu_options;
  proto::BaseOptions proto = ConvertBaseOptionsToProto(&base_options);
  EXPECT_TRUE(proto.acceleration().has_tflite());
  ASSERT_FALSE(proto.acceleration().has_gpu());
}

TEST(DelegateOptionsTest, SucceedGpuOptions) {
  BaseOptions base_options;
  base_options.delegate = BaseOptions::Delegate::GPU;
  BaseOptions::GpuOptions gpu_options;
  gpu_options.serialized_model_dir = kCachedModelDir;
  gpu_options.model_token = kModelToken;
  base_options.delegate_options = gpu_options;
  proto::BaseOptions proto = ConvertBaseOptionsToProto(&base_options);
  ASSERT_TRUE(proto.acceleration().has_gpu());
  ASSERT_FALSE(proto.acceleration().has_tflite());
  EXPECT_TRUE(proto.acceleration().gpu().use_advanced_gpu_api());
  EXPECT_FALSE(proto.acceleration().gpu().has_cached_kernel_path());
  EXPECT_EQ(proto.acceleration().gpu().serialized_model_dir(), kCachedModelDir);
  EXPECT_EQ(proto.acceleration().gpu().model_token(), kModelToken);
}

TEST(DelegateOptionsDeathTest, FailWrongDelegateOptionsType) {
  BaseOptions base_options;
  base_options.delegate = BaseOptions::Delegate::CPU;
  BaseOptions::GpuOptions gpu_options;
  gpu_options.cached_kernel_path = kCachedModelDir;
  gpu_options.model_token = kModelToken;
  base_options.delegate_options = gpu_options;
  ASSERT_DEATH(
      { proto::BaseOptions proto = ConvertBaseOptionsToProto(&base_options); },
      "Specified Delegate type does not match the provided "
      "delegate options.");
}

}  // namespace
}  // namespace core
}  // namespace tasks
}  // namespace mediapipe
