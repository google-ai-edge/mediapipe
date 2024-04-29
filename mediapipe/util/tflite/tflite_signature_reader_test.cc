#include "mediapipe/util/tflite/tflite_signature_reader.h"

#include <memory>
#include <string>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "mediapipe/framework/deps/file_path.h"
#include "mediapipe/framework/packet.h"
#include "mediapipe/framework/port/gmock.h"
#include "mediapipe/framework/port/gtest.h"
#include "mediapipe/framework/port/status_macros.h"
#include "mediapipe/framework/port/status_matchers.h"
#include "mediapipe/util/tflite/tflite_model_loader.h"
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/interpreter_builder.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/test_util.h"

namespace mediapipe {
namespace {

using ::testing::HasSubstr;
using ::tflite::ops::builtin::BuiltinOpResolverWithoutDefaultDelegates;
using ::tflite::testing::Test;

constexpr char kModelPath[] = "mediapipe/util/tflite/testdata/";

class TfLiteSignatureReaderTest : public Test {
 protected:
  absl::StatusOr<std::unique_ptr<tflite::Interpreter>>
  ReadModelAndBuildInterpreter(const std::string& model_path) {
    MP_ASSIGN_OR_RETURN(Packet model_packet,
                        TfLiteModelLoader::LoadFromPath(
                            file::JoinPath(kModelPath, model_path)));
    const auto& model = *model_packet.Get<TfLiteModelPtr>();

    std::unique_ptr<tflite::Interpreter> interpreter;
    tflite::InterpreterBuilder(
        model, BuiltinOpResolverWithoutDefaultDelegates())(&interpreter);
    return interpreter;
  }
};

TEST_F(TfLiteSignatureReaderTest, ShouldReadInputOutputTensorNames) {
  MP_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<tflite::Interpreter> interpreter,
      ReadModelAndBuildInterpreter("test_single_signature_key_model.tflite"));

  MP_ASSERT_OK_AND_ASSIGN(
      auto input_output_tensor_names,
      TfLiteSignatureReader::GetInputOutputTensorNamesFromTfliteSignature(
          *interpreter));

  // The order of the input and output tensors is not guaranteed during TfLite
  // conversion. Executing tf.lite.TFLiteConverter.convert flipped the order of
  // the input tensors as shown below.
  const std::vector<std::string> expected_input_tensor_names = {
      "third_input", "first_input", "second_input"};
  EXPECT_EQ(input_output_tensor_names.input_tensor_names,
            expected_input_tensor_names);

  const std::vector<std::string> expected_output_tensor_names = {
      "output_1", "output_0", "output_2"};
  EXPECT_EQ(input_output_tensor_names.output_tensor_names,
            expected_output_tensor_names);
}

TEST_F(TfLiteSignatureReaderTest, ShouldFailIfMultipleSignaturesExist) {
  MP_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<tflite::Interpreter> interpreter,
      ReadModelAndBuildInterpreter("test_two_signature_keys_model.tflite"));

  EXPECT_THAT(
      TfLiteSignatureReader::GetInputOutputTensorNamesFromTfliteSignature(
          *interpreter),
      StatusIs(absl::StatusCode::kInvalidArgument,
               HasSubstr("Model contains multiple signatures")));
}

TEST_F(TfLiteSignatureReaderTest, ShouldReadTensorNamesForSignature) {
  MP_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<tflite::Interpreter> interpreter,
      ReadModelAndBuildInterpreter("test_two_signature_keys_model.tflite"));
  // The test model contains two signatures "model_one" and "model_two" with
  // identical input and output tensors.
  for (const std::string signature_key : {"model_one", "model_two"}) {
    MP_ASSERT_OK_AND_ASSIGN(
        auto input_output_tensor_names,
        TfLiteSignatureReader::GetInputOutputTensorNamesFromTfliteSignature(
            *interpreter, &signature_key));

    // The order of the input and output tensors is not guaranteed during TfLite
    // conversion. Executing tf.lite.TFLiteConverter.convert flipped the order
    // of the input tensors as shown below.
    const std::vector<std::string> expected_input_tensor_names = {
        "third_input", "first_input", "second_input"};
    EXPECT_EQ(input_output_tensor_names.input_tensor_names,
              expected_input_tensor_names);

    const std::vector<std::string> expected_output_tensor_names = {
        "output_1", "output_0", "output_2"};
    EXPECT_EQ(input_output_tensor_names.output_tensor_names,
              expected_output_tensor_names);
  }
}

TEST_F(TfLiteSignatureReaderTest, ShouldReadTensorNamesForAllSignature) {
  MP_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<tflite::Interpreter> interpreter,
      ReadModelAndBuildInterpreter("test_two_signature_keys_model.tflite"));
  // The test model contains two signatures "model_one" and "model_two" with
  // identical input and output tensors.
  MP_ASSERT_OK_AND_ASSIGN(
      auto input_output_tensor_names_map,
      TfLiteSignatureReader::GetInputOutputTensorNamesFromAllTfliteSignatures(
          *interpreter));

  for (const std::string signature_key : {"model_one", "model_two"}) {
    auto it = input_output_tensor_names_map.find(signature_key);
    EXPECT_NE(it, input_output_tensor_names_map.end());
    const auto input_output_tensor_names = it->second;

    // The order of the input and output tensors is not guaranteed during TfLite
    // conversion. Executing tf.lite.TFLiteConverter.convert flipped the order
    // of the input tensors as shown below.
    const std::vector<std::string> expected_input_tensor_names = {
        "third_input", "first_input", "second_input"};
    EXPECT_EQ(input_output_tensor_names.input_tensor_names,
              expected_input_tensor_names);

    const std::vector<std::string> expected_output_tensor_names = {
        "output_1", "output_0", "output_2"};
    EXPECT_EQ(input_output_tensor_names.output_tensor_names,
              expected_output_tensor_names);
  }
}

}  // namespace
}  // namespace mediapipe
