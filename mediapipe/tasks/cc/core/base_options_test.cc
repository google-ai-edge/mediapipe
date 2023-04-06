#include "mediapipe/tasks/cc/core/base_options.h"

#include <string>

#include "mediapipe/calculators/tensor/inference_calculator.pb.h"
#include "mediapipe/framework/port/gmock.h"
#include "mediapipe/framework/port/gtest.h"
#include "mediapipe/tasks/cc/core/proto/acceleration.pb.h"
#include "mediapipe/tasks/cc/core/proto/external_file.pb.h"
#include "mediapipe/tasks/cc/core/utils.h"

constexpr char kTestModelBundlePath[] =
    "mediapipe/tasks/testdata/core/dummy_gesture_recognizer.task";

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

}  // namespace
}  // namespace core
}  // namespace tasks
}  // namespace mediapipe
