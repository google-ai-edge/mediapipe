#include "mediapipe/calculators/tensor/inference_interpreter_delegate_runner.h"

#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/functional/any_invocable.h"
#include "absl/status/status.h"
#include "mediapipe/calculators/tensor/tensor_span.h"
#include "mediapipe/calculators/tensor/tflite_delegate_ptr.h"
#include "mediapipe/framework/api2/builder.h"
#include "mediapipe/framework/api2/node.h"
#include "mediapipe/framework/api2/packet.h"
#include "mediapipe/framework/api2/port.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/calculator_state.h"
#include "mediapipe/framework/formats/tensor.h"
#include "mediapipe/framework/port/gmock.h"
#include "mediapipe/framework/port/gtest.h"
#include "mediapipe/framework/port/status_macros.h"
#include "mediapipe/framework/port/status_matchers.h"
#include "mediapipe/util/tflite/tflite_model_loader.h"
#include "tensorflow/lite/core/api/op_resolver.h"
#include "tensorflow/lite/delegates/xnnpack/xnnpack_delegate.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/util.h"

namespace mediapipe {
namespace api2 {
namespace {

using ::mediapipe::Tensor;
using ::testing::HasSubstr;

constexpr const char kInt32ModelFile[] =
    "mediapipe/calculators/tensor/testdata/"
    "1x3_square_int32.tflite";
constexpr const char kFloat32ModelFile[] =
    "mediapipe/calculators/tensor/testdata/"
    "1x3_square_float32.tflite";
// Signature of 3in3out_model_swaps_input_2_and_0.tflite model:
// ~~~~~~~~~~ INPUTS ~~~~~~~~~~
// 0 :  third_input :  [1 3] :  F32
// 1 :  first_input :  [1 1] :  F32
// 2 :  second_input :  [1 2] :  F32
// ~~~~~~~~~~ OUTPUTS ~~~~~~~~~
// 0 :  output_1 :  [1 2] :  F32
// 1 :  output_0 :  [1 1] :  F32
// 2 :  output_2 :  [1 3] :  F32
constexpr char k3In3OutSwaps2And0ModelPath[] =
    "mediapipe/calculators/tensor/testdata/"
    "3in3out_model_swaps_input_2_and_0.tflite";

class AnyInvocableCalculator : public Node {
 public:
  static constexpr Input<
      absl::AnyInvocable<absl::Status(CalculatorContext*) const>>
      kInput{"INPUT"};
  MEDIAPIPE_NODE_CONTRACT(kInput);
  absl::Status Process(CalculatorContext* cc) override {
    const auto& invokable = *kInput(cc);
    return invokable(cc);
  }
};
REGISTER_CALCULATOR(AnyInvocableCalculator);

class InferenceCalculatorDelegateRunnnerTest : public ::testing::Test {
 public:
  absl::Status ExecuteAnyInvocableInGraphCalculator(
      absl::AnyInvocable<absl::Status(CalculatorContext*) const> invokable) {
    mediapipe::api2::builder::Graph graph_builder;
    auto input = graph_builder.In("INPUT")
                     .SetName("input")
                     .Cast<absl::AnyInvocable<absl::Status() const>>();
    auto& inference_calculator =
        graph_builder.AddNode("AnyInvocableCalculator");
    input >> inference_calculator.In("INPUT")[0];
    auto config = graph_builder.GetConfig();
    CalculatorGraph graph;
    MP_RETURN_IF_ERROR(graph.Initialize(config));
    MP_RETURN_IF_ERROR(graph.StartRun({}));
    MP_RETURN_IF_ERROR(graph.AddPacketToInputStream(
        "input", mediapipe::MakePacket<absl::AnyInvocable<absl::Status(
                     CalculatorContext*) const>>(std::move(invokable))
                     .At(mediapipe::Timestamp(0))));
    MP_RETURN_IF_ERROR(graph.CloseAllInputStreams());
    return graph.WaitUntilDone();
  }

  template <typename VectorT, Tensor::ElementType TensorT>
  absl::Status CreateAndRunInferenceRunner(
      api2::Packet<TfLiteModelPtr> model,
      api2::Packet<tflite::OpResolver> op_resolver, TfLiteDelegatePtr delegate,
      bool enable_zero_copy_tensor_io,
      const std::vector<std::vector<VectorT>>& inputs,
      const std::vector<std::vector<VectorT>>& expected_outputs) {
    return ExecuteAnyInvocableInGraphCalculator(
        [&](CalculatorContext* cc) -> absl::Status {
          MP_ASSIGN_OR_RETURN(
              auto inference_runner,
              CreateInferenceInterpreterDelegateRunner(
                  std::move(model), std::move(op_resolver), std::move(delegate),
                  /*interpreter_num_threads=*/-1,
                  /*input_output_config=*/nullptr, enable_zero_copy_tensor_io));
          // Prepare input tensors.
          std::vector<Tensor> input_tensors;
          input_tensors.reserve(inputs.size());
          for (const auto& input_vec : inputs) {
            std::vector<int> dims({1});
            dims.push_back(input_vec.size());
            input_tensors.push_back(Tensor(TensorT, dims,
                                           /*memory_manager=*/nullptr,
                                           tflite::kDefaultTensorAlignment));
            {
              auto input_tensor_view = input_tensors.back().GetCpuWriteView();
              EXPECT_EQ(input_vec.size(),
                        input_tensors.back().shape().num_elements());
              VectorT* const input_buffer = input_tensor_view.buffer<VectorT>();
              for (int i = 0; i < input_vec.size(); ++i) {
                input_buffer[i] = input_vec[i];
              }
            }
          }
          // Execute inference.
          MP_ASSIGN_OR_RETURN(
              std::vector<Tensor> output_tensors,
              inference_runner->Run(cc, MakeTensorSpan(input_tensors)));
          // Check output tensors.
          EXPECT_EQ(output_tensors.size(), expected_outputs.size());
          for (int i = 0; i < output_tensors.size(); ++i) {
            const auto& output_tensor = output_tensors[i];
            const auto& expected_output = expected_outputs[i];
            EXPECT_EQ(output_tensor.element_type(), TensorT);
            EXPECT_EQ(output_tensor.shape().num_elements(),
                      expected_output.size());
            {
              auto output_tensor_view = output_tensor.GetCpuReadView();
              const VectorT* const output_buffer =
                  output_tensor_view.buffer<VectorT>();
              for (int j = 0; j < expected_output.size(); ++j) {
                EXPECT_EQ(output_buffer[j], expected_output[j]);
              }
            }
          }
          return absl::OkStatus();
        });
  }
};

TEST_F(InferenceCalculatorDelegateRunnnerTest,
       RunInt32ModelWithXNNPackDelegate) {
  MP_ASSERT_OK_AND_ASSIGN(auto model,
                          TfLiteModelLoader::LoadFromPath(kInt32ModelFile));
  // Create XNNPack delegate.
  auto op_resolver = PacketAdopting<tflite::OpResolver>(
      std::make_unique<
          tflite::ops::builtin::BuiltinOpResolverWithoutDefaultDelegates>());
  auto xnnpack_opts = TfLiteXNNPackDelegateOptionsDefault();
  auto delegate = TfLiteDelegatePtr(TfLiteXNNPackDelegateCreate(&xnnpack_opts),
                                    &TfLiteXNNPackDelegateDelete);
  MP_EXPECT_OK(
      (CreateAndRunInferenceRunner<int32_t, Tensor::ElementType::kInt32>(
          std::move(model), std::move(op_resolver), std::move(delegate),
          /*enable_zero_copy_tensor_io=*/false,
          /*inputs=*/{{0, 1, 2}},
          /*expected_outputs=*/{{0 * 0, 1 * 1, 2 * 2}})));
}

TEST_F(InferenceCalculatorDelegateRunnnerTest,
       RunFloatModelWithXNNPackDelegate) {
  MP_ASSERT_OK_AND_ASSIGN(auto model,
                          TfLiteModelLoader::LoadFromPath(kFloat32ModelFile));
  // Create XNNPack delegate.
  auto op_resolver = PacketAdopting<tflite::OpResolver>(
      std::make_unique<
          tflite::ops::builtin::BuiltinOpResolverWithoutDefaultDelegates>());
  auto xnnpack_opts = TfLiteXNNPackDelegateOptionsDefault();
  auto delegate = TfLiteDelegatePtr(TfLiteXNNPackDelegateCreate(&xnnpack_opts),
                                    &TfLiteXNNPackDelegateDelete);
  MP_EXPECT_OK(
      (CreateAndRunInferenceRunner<float, Tensor::ElementType::kFloat32>(
          std::move(model), std::move(op_resolver), std::move(delegate),
          /*enable_zero_copy_tensor_io=*/false,
          /*inputs=*/{{0.f, 1.f, 2.f}},
          /*expected_outputs=*/{{0.f * 0.f, 1.f * 1.f, 2.f * 2.f}})));
}

TEST_F(InferenceCalculatorDelegateRunnnerTest,
       RunInt32ModelWithXNNPackDelegateWithCustomAllocation) {
  MP_ASSERT_OK_AND_ASSIGN(auto model,
                          TfLiteModelLoader::LoadFromPath(kInt32ModelFile));
  // Create XNNPack delegate.
  auto op_resolver = PacketAdopting<tflite::OpResolver>(
      std::make_unique<
          tflite::ops::builtin::BuiltinOpResolverWithoutDefaultDelegates>());
  auto xnnpack_opts = TfLiteXNNPackDelegateOptionsDefault();
  auto delegate = TfLiteDelegatePtr(TfLiteXNNPackDelegateCreate(&xnnpack_opts),
                                    &TfLiteXNNPackDelegateDelete);
  MP_EXPECT_OK(
      (CreateAndRunInferenceRunner<int32_t, Tensor::ElementType::kInt32>(
          std::move(model), std::move(op_resolver), std::move(delegate),
          /*enable_zero_copy_tensor_io=*/true,
          /*inputs=*/{{0, 1, 2}},
          /*expected_outputs=*/{{0 * 0, 1 * 1, 2 * 2}})));
}

TEST_F(InferenceCalculatorDelegateRunnnerTest,
       RunFloat32ModelWithXNNPackDelegateWithCustomAllocation) {
  MP_ASSERT_OK_AND_ASSIGN(auto model,
                          TfLiteModelLoader::LoadFromPath(kFloat32ModelFile));
  // Create XNNPack delegate.
  auto op_resolver = PacketAdopting<tflite::OpResolver>(
      std::make_unique<
          tflite::ops::builtin::BuiltinOpResolverWithoutDefaultDelegates>());
  auto xnnpack_opts = TfLiteXNNPackDelegateOptionsDefault();
  auto delegate = TfLiteDelegatePtr(TfLiteXNNPackDelegateCreate(&xnnpack_opts),
                                    &TfLiteXNNPackDelegateDelete);
  MP_EXPECT_OK(
      (CreateAndRunInferenceRunner<float, Tensor::ElementType::kFloat32>(
          std::move(model), std::move(op_resolver), std::move(delegate),
          /*enable_zero_copy_tensor_io=*/true,
          /*inputs=*/{{0.f, 1.f, 2.f}},
          /*expected_outputs=*/{{0.f * 0.f, 1.f * 1.f, 2.f * 2.f}})));
}

TEST_F(InferenceCalculatorDelegateRunnnerTest,
       RunInt32ModelWithDefaultDelegate) {
  MP_ASSERT_OK_AND_ASSIGN(auto model,
                          TfLiteModelLoader::LoadFromPath(kInt32ModelFile));
  auto op_resolver = PacketAdopting<tflite::OpResolver>(
      std::make_unique<
          tflite::ops::builtin::BuiltinOpResolverWithoutDefaultDelegates>());
  MP_EXPECT_OK(
      (CreateAndRunInferenceRunner<int32_t, Tensor::ElementType::kInt32>(
          std::move(model), std::move(op_resolver), /*delegate=*/nullptr,
          /*enable_zero_copy_tensor_io=*/false,
          /*inputs=*/{{0, 1, 2}},
          /*expected_outputs=*/{{0 * 0, 1 * 1, 2 * 2}})));
}

TEST_F(InferenceCalculatorDelegateRunnnerTest,
       RunFloat32ModelWithDefaultDelegate) {
  MP_ASSERT_OK_AND_ASSIGN(auto model,
                          TfLiteModelLoader::LoadFromPath(kFloat32ModelFile));
  auto op_resolver = PacketAdopting<tflite::OpResolver>(
      std::make_unique<
          tflite::ops::builtin::BuiltinOpResolverWithoutDefaultDelegates>());
  MP_EXPECT_OK(
      (CreateAndRunInferenceRunner<float, Tensor::ElementType::kFloat32>(
          std::move(model), std::move(op_resolver), /*delegate=*/nullptr,
          /*enable_zero_copy_tensor_io=*/false,
          /*inputs=*/{{0.f, 1.f, 2.f}},
          /*expected_outputs=*/{{0.f * 0.f, 1.f * 1.f, 2.f * 2.f}})));
}

TEST_F(InferenceCalculatorDelegateRunnnerTest,
       RunInt32ModelWithDefaultDelegateWithCustomAllocation) {
  MP_ASSERT_OK_AND_ASSIGN(auto model,
                          TfLiteModelLoader::LoadFromPath(kInt32ModelFile));
  auto op_resolver = PacketAdopting<tflite::OpResolver>(
      std::make_unique<
          tflite::ops::builtin::BuiltinOpResolverWithoutDefaultDelegates>());
  MP_EXPECT_OK(
      (CreateAndRunInferenceRunner<int32_t, Tensor::ElementType::kInt32>(
          std::move(model), std::move(op_resolver), /*delegate=*/nullptr,
          /*enable_zero_copy_tensor_io=*/true,
          /*inputs=*/{{0, 1, 2}},
          /*expected_outputs=*/{{0 * 0, 1 * 1, 2 * 2}})));
}

TEST_F(InferenceCalculatorDelegateRunnnerTest,
       RunFloat32ModelWithDefaultDelegateWithCustomAllocation) {
  MP_ASSERT_OK_AND_ASSIGN(auto model,
                          TfLiteModelLoader::LoadFromPath(kFloat32ModelFile));
  auto op_resolver = PacketAdopting<tflite::OpResolver>(
      std::make_unique<
          tflite::ops::builtin::BuiltinOpResolverWithoutDefaultDelegates>());
  MP_EXPECT_OK(
      (CreateAndRunInferenceRunner<float, Tensor::ElementType::kFloat32>(
          std::move(model), std::move(op_resolver), /*delegate=*/nullptr,
          /*enable_zero_copy_tensor_io=*/true,
          /*inputs=*/{{0.f, 1.f, 2.f}},
          /*expected_outputs=*/{{0.f * 0.f, 1.f * 1.f, 2.f * 2.f}})));
}

TEST_F(InferenceCalculatorDelegateRunnnerTest,
       RunFloat32ModelWithXNNPackDelegatePassthrough) {
  MP_ASSERT_OK_AND_ASSIGN(
      auto model, TfLiteModelLoader::LoadFromPath(k3In3OutSwaps2And0ModelPath));
  // Create XNNPack delegate.
  auto op_resolver = PacketAdopting<tflite::OpResolver>(
      std::make_unique<
          tflite::ops::builtin::BuiltinOpResolverWithoutDefaultDelegates>());
  auto xnnpack_opts = TfLiteXNNPackDelegateOptionsDefault();
  auto delegate = TfLiteDelegatePtr(TfLiteXNNPackDelegateCreate(&xnnpack_opts),
                                    &TfLiteXNNPackDelegateDelete);
  // Signature of 3in3out_model_swaps_input_2_and_0.tflite model:
  // ~~~~~~~~~~ INPUTS ~~~~~~~~~~
  // 0 :  third_input :  [1 1] :  F32
  // 1 :  first_input :  [1 1] :  F32
  // 2 :  second_input :  [1 1] :  F32
  // ~~~~~~~~~~ OUTPUTS ~~~~~~~~~
  // 0 :  output_1 :  [1 1] :  F32
  // 1 :  output_0 :  [1 1] :  F32
  // 2 :  output_2 :  [1 1] :  F32
  MP_EXPECT_OK(
      (CreateAndRunInferenceRunner<float, Tensor::ElementType::kFloat32>(
          std::move(model), std::move(op_resolver), std::move(delegate),
          /*enable_zero_copy_tensor_io=*/false,
          /*inputs=*/{{1.f}, {2.f}, {3.f}},
          /*expected_outputs=*/{{3.f}, {2.f}, {1.f}})));
}

TEST_F(InferenceCalculatorDelegateRunnnerTest,
       RunPassthroughFloatModelWithXNNPackDelegateWithCustomAllocation) {
  MP_ASSERT_OK_AND_ASSIGN(
      auto model, TfLiteModelLoader::LoadFromPath(k3In3OutSwaps2And0ModelPath));
  // Create XNNPack delegate.
  auto op_resolver = PacketAdopting<tflite::OpResolver>(
      std::make_unique<
          tflite::ops::builtin::BuiltinOpResolverWithoutDefaultDelegates>());
  auto xnnpack_opts = TfLiteXNNPackDelegateOptionsDefault();
  auto delegate = TfLiteDelegatePtr(TfLiteXNNPackDelegateCreate(&xnnpack_opts),
                                    &TfLiteXNNPackDelegateDelete);
  EXPECT_THAT(
      (CreateAndRunInferenceRunner<float, Tensor::ElementType::kFloat32>(
          std::move(model), std::move(op_resolver), std::move(delegate),
          /*enable_zero_copy_tensor_io=*/true,
          /*inputs=*/{{1.f}, {2.f}, {3.f}},
          /*expected_outputs=*/{{3.f}, {2.f}, {1.f}})),
      StatusIs(absl::StatusCode::kInternal,
               HasSubstr("Custom allocation is not supported for models with "
                         "input->output passthrough tensors")));
}

}  // namespace
}  // namespace api2
}  // namespace mediapipe
