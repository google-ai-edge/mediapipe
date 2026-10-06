#include "mediapipe/tasks/cc/core/base_options.h"

#include <memory>
#include <optional>
#include <string>
#include <utility>
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

TEST(BaseOptionsTest, ConvertProtoToBaseOptionsWithFile) {
  proto::BaseOptions proto;
  proto.mutable_model_asset()->set_file_content("dummy_content");
  proto.mutable_model_asset()->mutable_file_descriptor_meta()->set_fd(123);
  BaseOptions base_options = ConvertProtoToBaseOptions(std::move(proto));
  ASSERT_NE(base_options.model_asset_buffer, nullptr);
  EXPECT_EQ(*base_options.model_asset_buffer, "dummy_content");
  EXPECT_EQ(base_options.model_asset_descriptor_meta.fd, 123);
}

TEST(BaseOptionsTest, ConvertProtoToBaseOptionsWithCpuDelegate) {
  proto::BaseOptions proto;
  proto.mutable_acceleration()->mutable_tflite();
  BaseOptions base_options = ConvertProtoToBaseOptions(std::move(proto));
  EXPECT_EQ(base_options.delegate, BaseOptions::Delegate::CPU);
}

TEST(BaseOptionsTest, ConvertProtoToBaseOptionsWithGpuDelegate) {
  proto::BaseOptions proto;
  auto* gpu = proto.mutable_acceleration()->mutable_gpu();
  gpu->set_serialized_model_dir(kCachedModelDir);
  gpu->set_model_token(kModelToken);
  BaseOptions base_options = ConvertProtoToBaseOptions(std::move(proto));
  EXPECT_EQ(base_options.delegate, BaseOptions::Delegate::GPU);
  ASSERT_TRUE(base_options.delegate_options.has_value());
  ASSERT_TRUE(std::holds_alternative<BaseOptions::GpuOptions>(
      *base_options.delegate_options));
  const auto& gpu_opts =
      std::get<BaseOptions::GpuOptions>(*base_options.delegate_options);
  EXPECT_EQ(gpu_opts.serialized_model_dir, kCachedModelDir);
  EXPECT_EQ(gpu_opts.model_token, kModelToken);
}

}  // namespace
}  // namespace core
}  // namespace tasks
}  // namespace mediapipe
