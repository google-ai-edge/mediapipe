#include "mediapipe/tasks/cc/core/base_options.h"

#include <cstdio>
#include <fstream>
#include <ios>
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

  base_options.delegate = BaseOptions::Delegate::NPU;
  proto = ConvertBaseOptionsToProto(&base_options);
  EXPECT_TRUE(proto.acceleration().has_litert());
  EXPECT_TRUE(proto.acceleration().litert().has_npu());
}

TEST(BaseOptionsTest, ConvertBaseOptionsToProtoWithUseLiteRt) {
  BaseOptions base_options;

  // CPU delegate with use_litert routes to LiteRT CPU
  base_options.delegate = BaseOptions::Delegate::CPU;
  proto::BaseOptions proto =
      ConvertBaseOptionsToProto(&base_options, /*use_litert=*/true);
  EXPECT_TRUE(proto.acceleration().has_litert());
  EXPECT_TRUE(proto.acceleration().litert().has_cpu());

  // GPU delegate with use_litert routes to LiteRT GPU
  base_options.delegate = BaseOptions::Delegate::GPU;
  base_options.delegate_options = std::nullopt;
  proto = ConvertBaseOptionsToProto(&base_options, /*use_litert=*/true);
  EXPECT_TRUE(proto.acceleration().has_litert());
  EXPECT_TRUE(proto.acceleration().litert().has_gpu());
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
      "Specified Delegate type does not match the provided delegate options.");
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

TEST(BaseOptionsTest, ConvertProtoToBaseOptionsWithLiteRtGpu) {
  proto::BaseOptions proto;
  auto* gpu = proto.mutable_acceleration()->mutable_litert()->mutable_gpu();
  gpu->mutable_cache_options()->set_serialization_dir(kCachedModelDir);
  gpu->mutable_cache_options()->set_model_cache_key(kModelToken);

  BaseOptions base_options = ConvertProtoToBaseOptions(std::move(proto));

  // A LiteRT GPU accelerator must map back to the GPU delegate, not silently
  // fall through to the default CPU delegate.
  EXPECT_EQ(base_options.delegate, BaseOptions::Delegate::GPU);
  ASSERT_TRUE(base_options.delegate_options.has_value());
  ASSERT_TRUE(std::holds_alternative<BaseOptions::GpuOptions>(
      *base_options.delegate_options));
  const auto& gpu_opts =
      std::get<BaseOptions::GpuOptions>(*base_options.delegate_options);
  EXPECT_EQ(gpu_opts.serialized_model_dir, kCachedModelDir);
  EXPECT_EQ(gpu_opts.model_token, kModelToken);
  // `cached_kernel_path` is not part of the LiteRT GPU cache options, so it
  // must stay empty rather than mirroring the serialization directory.
  EXPECT_TRUE(gpu_opts.cached_kernel_path.empty());
}

TEST(BaseOptionsTest, ConvertProtoToBaseOptionsWithLiteRtNpu) {
  proto::BaseOptions proto;
  proto.mutable_acceleration()
      ->mutable_litert()
      ->mutable_npu()
      ->set_dispatch_library_path("/tmp/dispatch");

  BaseOptions base_options = ConvertProtoToBaseOptions(std::move(proto));

  EXPECT_EQ(base_options.delegate, BaseOptions::Delegate::NPU);
  ASSERT_TRUE(base_options.delegate_options.has_value());
  ASSERT_TRUE(std::holds_alternative<BaseOptions::NpuOptions>(
      *base_options.delegate_options));
  EXPECT_EQ(std::get<BaseOptions::NpuOptions>(*base_options.delegate_options)
                .dispatch_library_directory,
            "/tmp/dispatch");
}

TEST(BaseOptionsTest, ConvertProtoToBaseOptionsWithLiteRtCpu) {
  proto::BaseOptions proto;
  proto.mutable_acceleration()->mutable_litert()->mutable_cpu();

  BaseOptions base_options = ConvertProtoToBaseOptions(std::move(proto));

  EXPECT_EQ(base_options.delegate, BaseOptions::Delegate::CPU);
}

// Round trips are what interactive_segmenter_jni.cc and
// interactive_segmenter_wasm.cc perform: they parse a BaseOptions proto,
// convert it to a BaseOptions, and let the task convert it back. The task
// re-supplies `use_litert`; what must survive the conversion is the
// accelerator choice, otherwise a GPU request silently downgrades to CPU.
TEST(BaseOptionsTest, LiteRtGpuSurvivesProtoRoundTrip) {
  proto::BaseOptions proto;
  auto* gpu = proto.mutable_acceleration()->mutable_litert()->mutable_gpu();
  gpu->set_precision(
      mediapipe::InferenceCalculatorOptions::Delegate::LiteRt::Gpu::FP32);
  gpu->set_backend(
      mediapipe::InferenceCalculatorOptions::Delegate::LiteRt::Gpu::OPENGL);
  gpu->mutable_cache_options()->set_serialization_dir(kCachedModelDir);
  gpu->mutable_cache_options()->set_model_cache_key(kModelToken);

  BaseOptions base_options = ConvertProtoToBaseOptions(std::move(proto));
  ASSERT_TRUE(base_options.delegate_options.has_value());
  ASSERT_TRUE(std::holds_alternative<BaseOptions::GpuOptions>(
      *base_options.delegate_options));
  const auto& gpu_opts =
      std::get<BaseOptions::GpuOptions>(*base_options.delegate_options);
  EXPECT_EQ(gpu_opts.precision, BaseOptions::GpuOptions::Precision::FP32);
  EXPECT_EQ(gpu_opts.backend, BaseOptions::GpuOptions::Backend::OPENGL);
  EXPECT_EQ(gpu_opts.serialized_model_dir, kCachedModelDir);
  EXPECT_EQ(gpu_opts.model_token, kModelToken);

  proto::BaseOptions round_tripped =
      ConvertBaseOptionsToProto(&base_options, /*use_litert=*/true);

  ASSERT_TRUE(round_tripped.acceleration().has_litert());
  EXPECT_TRUE(round_tripped.acceleration().litert().has_gpu());
  EXPECT_EQ(round_tripped.acceleration().litert().gpu().precision(),
            mediapipe::InferenceCalculatorOptions::Delegate::LiteRt::Gpu::FP32);
  EXPECT_EQ(
      round_tripped.acceleration().litert().gpu().backend(),
      mediapipe::InferenceCalculatorOptions::Delegate::LiteRt::Gpu::OPENGL);
  // Cache options must survive the round trip too, otherwise a caller that
  // opted into model serialization silently loses it and pays the compilation
  // cost on every run.
  const auto& round_tripped_cache =
      round_tripped.acceleration().litert().gpu().cache_options();
  EXPECT_EQ(round_tripped_cache.serialization_dir(), kCachedModelDir);
  EXPECT_EQ(round_tripped_cache.model_cache_key(), kModelToken);
  // Re-derived because both the directory and the cache key are present.
  EXPECT_TRUE(round_tripped_cache.serialize_program_cache());
  // Still GPU-only after the round trip.
  EXPECT_FALSE(round_tripped.acceleration().litert().has_cpu());
  EXPECT_FALSE(round_tripped.acceleration().has_gpu());
  EXPECT_FALSE(round_tripped.acceleration().has_tflite());
}

TEST(BaseOptionsTest, LiteRtNpuSurvivesProtoRoundTrip) {
  proto::BaseOptions proto;
  proto.mutable_acceleration()
      ->mutable_litert()
      ->mutable_npu()
      ->set_dispatch_library_path("/tmp/dispatch");

  BaseOptions base_options = ConvertProtoToBaseOptions(std::move(proto));
  // NPU always routes through LiteRT, so no `use_litert` argument is needed.
  proto::BaseOptions round_tripped = ConvertBaseOptionsToProto(&base_options);

  ASSERT_TRUE(round_tripped.acceleration().has_litert());
  ASSERT_TRUE(round_tripped.acceleration().litert().has_npu());
  EXPECT_EQ(round_tripped.acceleration().litert().npu().dispatch_library_path(),
            "/tmp/dispatch");
}

// A task that has not opted into LiteRT still gets the legacy backend, but it
// must keep the GPU accelerator the caller asked for.
TEST(BaseOptionsTest, LiteRtGpuProtoFallsBackToLegacyGpuWithoutUseLiteRt) {
  proto::BaseOptions proto;
  proto.mutable_acceleration()->mutable_litert()->mutable_gpu();

  BaseOptions base_options = ConvertProtoToBaseOptions(std::move(proto));
  proto::BaseOptions round_tripped = ConvertBaseOptionsToProto(&base_options);

  ASSERT_TRUE(round_tripped.acceleration().has_gpu());
  EXPECT_TRUE(round_tripped.acceleration().gpu().use_advanced_gpu_api());
  EXPECT_FALSE(round_tripped.acceleration().has_tflite());
}

// The legacy (non-LiteRT) delegates must keep round tripping unchanged.
TEST(BaseOptionsTest, LegacyGpuSurvivesProtoRoundTrip) {
  proto::BaseOptions proto;
  proto.mutable_acceleration()->mutable_gpu()->set_use_advanced_gpu_api(true);

  BaseOptions base_options = ConvertProtoToBaseOptions(std::move(proto));
  proto::BaseOptions round_tripped = ConvertBaseOptionsToProto(&base_options);

  ASSERT_TRUE(round_tripped.acceleration().has_gpu());
  EXPECT_TRUE(round_tripped.acceleration().gpu().use_advanced_gpu_api());
  EXPECT_FALSE(round_tripped.acceleration().has_litert());
}

TEST(BaseOptionsTest, IsLiteRtLmModelReturnsFalseForDefaultOptions) {
  BaseOptions base_options;
  EXPECT_FALSE(IsLiteRtLmModel(base_options));
}

TEST(BaseOptionsTest, IsLiteRtLmModelReturnsTrueForValidModelBuffer) {
  BaseOptions base_options;
  base_options.model_asset_buffer =
      std::make_unique<std::string>("LITERTLM_model_data");
  EXPECT_TRUE(IsLiteRtLmModel(base_options));
}

TEST(BaseOptionsTest, IsLiteRtLmModelReturnsFalseForInvalidModelBuffer) {
  BaseOptions base_options;
  base_options.model_asset_buffer =
      std::make_unique<std::string>("NOTLITERTLM_model_data");
  EXPECT_FALSE(IsLiteRtLmModel(base_options));
}

TEST(BaseOptionsTest, IsLiteRtLmModelReturnsTrueForValidModelPath) {
  std::string temp_path = testing::TempDir() + "test_litert_lm_model";
  std::ofstream out(temp_path, std::ios::binary);
  ASSERT_TRUE(out.is_open());
  out.write("LITERTLM_model_data", 19);
  out.close();

  BaseOptions base_options;
  base_options.model_asset_path = temp_path;
  EXPECT_TRUE(IsLiteRtLmModel(base_options));

  std::remove(temp_path.c_str());
}

TEST(BaseOptionsTest, IsLiteRtLmModelReturnsFalseForInvalidModelPath) {
  std::string temp_path = testing::TempDir() + "test_invalid_model";
  std::ofstream out(temp_path, std::ios::binary);
  ASSERT_TRUE(out.is_open());
  out.write("NOTLITERTLM_model_data", 22);
  out.close();

  BaseOptions base_options;
  base_options.model_asset_path = temp_path;
  EXPECT_FALSE(IsLiteRtLmModel(base_options));

  std::remove(temp_path.c_str());
}

}  // namespace
}  // namespace core
}  // namespace tasks
}  // namespace mediapipe
