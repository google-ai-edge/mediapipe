// Copyright (c) 2023 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//

#ifndef MEDIAPIPE_CALCULATORS_OPENVINO_OPENVINO_INFERENCE_CALCULATOR_TEST_H_
#define MEDIAPIPE_CALCULATORS_OPENVINO_OPENVINO_INFERENCE_CALCULATOR_TEST_H_

#include <memory>
#include <string>
#include <vector>

#include "absl/strings/str_replace.h"
#include "absl/strings/string_view.h"
#include "mediapipe/calculators/openvino/openvino_inference_calculator.pb.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/calculator_runner.h"
#include "mediapipe/framework/deps/file_path.h"
#include "mediapipe/framework/port/gmock.h"
#include "mediapipe/framework/port/gtest.h"
#include "mediapipe/framework/port/integral_types.h"
#include "mediapipe/framework/port/parse_text_proto.h"
#include "mediapipe/framework/port/status_matchers.h"  // NOLINT
#include "mediapipe/framework/tool/validate_type.h"

#include <openvino/openvino.hpp>

#ifdef __APPLE__
#include <CoreFoundation/CoreFoundation.h>
#endif  // defined(__APPLE__)

namespace mediapipe {

template <typename T>
void DoSmokeTest(const std::string& graph_proto) {
  const int width = 8;
  const int height = 8;
  const int channels = 3;

  static_assert(std::is_same_v<T, uint8>,
                "Only uint8 currently supported.");

  // Prepare interpreter and input tensor.
  ov::Tensor input_tensor(ov::element::u8, {1, channels, width, height});

  T* input_tensor_buffer = input_tensor.data<T>();
  ASSERT_NE(input_tensor_buffer, nullptr);
  for (int i = 0; i < width * height * channels; i++) {
    input_tensor_buffer[i] = 2;
  }

  auto input_vec = absl::make_unique<std::vector<ov::Tensor>>();
  input_vec->emplace_back(input_tensor);

  auto input_vec2 = absl::make_unique<std::vector<ov::Tensor>>();
  input_vec2->emplace_back(input_tensor);

  // Prepare single calculator graph to and wait for packets.
  CalculatorGraphConfig graph_config =
      ParseTextProtoOrDie<CalculatorGraphConfig>(graph_proto);
  std::vector<Packet> output_packets;
  tool::AddVectorSink("tensor_out", &graph_config, &output_packets);
  CalculatorGraph graph(graph_config);
  MP_ASSERT_OK(graph.StartRun({}));

  // Push the tensor into the graph.
  MP_ASSERT_OK(graph.AddPacketToInputStream(
      "tensor_in", Adopt(input_vec.release()).At(Timestamp(0))));
  // Wait until the calculator done processing.
  MP_ASSERT_OK(graph.WaitUntilIdle());
  ASSERT_EQ(1, output_packets.size());

  // Get and process results - the model
  const std::vector<ov::Tensor>& result_vec =
      output_packets[0].Get<std::vector<ov::Tensor>>();
  ASSERT_EQ(1, result_vec.size());

  const ov::Tensor& out_tensor = result_vec[0];
  const ov::Shape& out_shape = out_tensor.get_shape();
  ASSERT_EQ(out_shape[0], 1);
  ASSERT_EQ(out_shape[1], channels * 2);
  ASSERT_EQ(out_shape[2], width);
  ASSERT_EQ(out_shape[3], height);

  const T* result_buffer = out_tensor.data<T>();
  ASSERT_NE(result_buffer, nullptr);
  for (int i = 0; i < 2 * width * height * channels; i++) {
    ASSERT_EQ(4, result_buffer[i]);
  }

  // Fully close graph at end, otherwise calculator+tensors are destroyed
  // after calling WaitUntilDone().
  MP_ASSERT_OK(graph.CloseInputStream("tensor_in"));
  MP_ASSERT_OK(graph.WaitUntilDone());
}

}  // namespace mediapipe

#endif  // MEDIAPIPE_CALCULATORS_OPENVINO_OPENVINO_INFERENCE_CALCULATOR_TEST_H_
