#include "mediapipe/util/tflite/tflite_model_loader.h"

#include <memory>
#include <string>

#include "absl/flags/declare.h"
#include "absl/flags/flag.h"
#include "absl/strings/str_cat.h"
#include "mediapipe/framework/api2/packet.h"
#include "mediapipe/framework/calculator.pb.h"
#include "mediapipe/framework/calculator_context.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/calculator_state.h"
#include "mediapipe/framework/legacy_calculator_support.h"
#include "mediapipe/framework/port/gtest.h"
#include "mediapipe/framework/port/status_matchers.h"
#include "mediapipe/framework/tool/tag_map_helper.h"
#include "tensorflow/lite/test_util.h"

ABSL_DECLARE_FLAG(std::string, resource_root_dir);

namespace mediapipe {
namespace {

constexpr char kModelDir[] = "mediapipe/util/tflite/testdata";
constexpr char kModelFilename[] = "test_model.tflite";

class TfLiteModelLoaderTest : public tflite::testing::Test {
  void SetUp() override {
    // Create a stub calculator state.
    CalculatorGraphConfig::Node config;
    calculator_state_ = std::make_unique<CalculatorState>(
        "fake_node", 0, "fake_type", config, nullptr);

    // Create a stub calculator context.
    calculator_context_ = std::make_unique<CalculatorContext>(
        calculator_state_.get(), tool::CreateTagMap({}).value(),
        tool::CreateTagMap({}).value());
  }

 protected:
  std::unique_ptr<CalculatorState> calculator_state_;
  std::unique_ptr<CalculatorContext> calculator_context_;
  std::string model_path_ = absl::StrCat(kModelDir, "/", kModelFilename);
};

TEST_F(TfLiteModelLoaderTest, LoadFromPath) {
  // TODO: remove LegacyCalculatorSupport usage.
  LegacyCalculatorSupport::Scoped<CalculatorContext> scope(
      calculator_context_.get());
  MP_ASSERT_OK_AND_ASSIGN(api2::Packet<TfLiteModelPtr> model,
                          TfLiteModelLoader::LoadFromPath(model_path_));
  EXPECT_NE(model.Get(), nullptr);
}

TEST_F(TfLiteModelLoaderTest, LoadFromPathRelativeToRootDir) {
  absl::SetFlag(&FLAGS_resource_root_dir, kModelDir);

  // TODO: remove LegacyCalculatorSupport usage.
  LegacyCalculatorSupport::Scoped<CalculatorContext> scope(
      calculator_context_.get());
  MP_ASSERT_OK_AND_ASSIGN(api2::Packet<TfLiteModelPtr> model,
                          TfLiteModelLoader::LoadFromPath(kModelFilename));
  EXPECT_NE(model.Get(), nullptr);
}

TEST_F(TfLiteModelLoaderTest, LoadFromPathWithMmap) {
  // TODO: remove LegacyCalculatorSupport usage.
  LegacyCalculatorSupport::Scoped<CalculatorContext> scope(
      calculator_context_.get());
  MP_ASSERT_OK_AND_ASSIGN(
      api2::Packet<TfLiteModelPtr> model,
      TfLiteModelLoader::LoadFromPath(model_path_, /* try_mmap=*/true));
  EXPECT_NE(model.Get(), nullptr);
}

}  // namespace
}  // namespace mediapipe
