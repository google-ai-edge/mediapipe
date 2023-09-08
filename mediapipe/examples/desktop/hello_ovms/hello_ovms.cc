//
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

#include "ovms.h"
#include "c_api_test_utils.hpp"
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#include "mediapipe/framework/calculator_graph.h"
#include "mediapipe/framework/port/logging.h"
#include "mediapipe/framework/port/parse_text_proto.h"
#include "mediapipe/framework/port/status.h"
#pragma GCC diagnostic pop

#include <openvino/openvino.hpp>

#include <iostream>
using std::cout;
using std::endl;
namespace mediapipe {

#define CALL_AND_CHECK_RET(CALL)                                                                            \
    {                                                                                                       \ 
        auto absStatus = CALL;                                                                              \
        if (!absStatus.ok()) {                                                                              \
            cout << __FILE__ << ":" << __LINE__ << endl;                                                    \
          const std::string absMessage = absStatus.ToString();                                              \
          std::cout << "ERROR when calling: " << #CALL << " ERROR: " << std::move(absMessage)<< std::endl;  \
          exit(1);                                                                                          \
        }                                                                                                   \
    }                                                                                                       \

absl::Status RunMediapipeGraph() {
  // Configures a simple graph, which concatenates 2 PassThroughCalculators.
  CalculatorGraphConfig config =
      ParseTextProtoOrDie<CalculatorGraphConfig>(R"pb(
        input_stream: "in1"
        input_stream: "in2"
        output_stream: "out"
        node {
          calculator: "OpenVINOModelServerSessionCalculator"
          output_side_packet: "SESSION:session"
          node_options: {
            [type.googleapis.com / mediapipe.OpenVINOModelServerSessionCalculatorOptions]: {
              servable_name: "add"
              servable_version: "1"
            }
          }
        }
        node {
          calculator: "OpenVINOInferenceCalculator"
          input_side_packet: "SESSION:session"
          input_stream: "INPUT1:in1"
          input_stream: "INPUT2:in2"
          output_stream: "SUM:out"
          node_options: {
            [type.googleapis.com / mediapipe.OpenVINOInferenceCalculatorOptions]: {
              tag_to_input_tensor_names {
                key: "INPUT1"
                value: "input1"
              }
              tag_to_input_tensor_names {
                key: "INPUT2"
                value: "input2"
              }
              tag_to_output_tensor_names {
                key: "SUM"
                value: "sum"
              }
            }
          }
        }
      )pb");

  CalculatorGraph graph;
  CALL_AND_CHECK_RET(graph.Initialize(config));
  
  ASSIGN_OR_RETURN(OutputStreamPoller poller,
                   graph.AddOutputStreamPoller("out"));
  CALL_AND_CHECK_RET(graph.StartRun({}));

  ov::Tensor tensor1 = ov::Tensor(ov::element::f32, {1,10});
  float data[] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0};
  std::memcpy(tensor1.data(), data, tensor1.get_byte_size());
  CALL_AND_CHECK_RET(graph.AddPacketToInputStream(
      "in1", MakePacket<ov::Tensor>(std::move(tensor1)).At(Timestamp(0))));


  ov::Tensor tensor2 = ov::Tensor(ov::element::f32, {1,10});
  float data2[] = {10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0};
  std::memcpy(tensor2.data(), data2, tensor2.get_byte_size());
  CALL_AND_CHECK_RET(graph.AddPacketToInputStream(
      "in2", MakePacket<ov::Tensor>(std::move(tensor2)).At(Timestamp(0))));


  // Close the input stream "in".
  CALL_AND_CHECK_RET(graph.CloseInputStream("in1"));
  CALL_AND_CHECK_RET(graph.CloseInputStream("in2"));
  mediapipe::Packet packet;
  // Get the output packets string.
  while (poller.Next(&packet)) {
    ov::Tensor tensor3 = packet.Get<ov::Tensor>();
    float* ptr = reinterpret_cast<float*>(tensor3.data());
    for(int i = 0; i < 10; i++){
      cout << "Output tensor data: "<< i << " - " << ptr[i] << std::endl;
    }
  }
  return graph.WaitUntilDone();
}
}  // namespace mediapipe

void InitOvmsServer(OVMS_Server* srv, OVMS_ServerSettings* serverSettings, OVMS_ModelsSettings* modelsSettings)
{
    ASSERT_CAPI_STATUS_NULL(OVMS_ServerNew(&srv));
    ASSERT_CAPI_STATUS_NULL(OVMS_ServerSettingsNew(&serverSettings));
    ASSERT_CAPI_STATUS_NULL(OVMS_ServerSettingsSetLogLevel(serverSettings, OVMS_LOG_DEBUG));
    ASSERT_CAPI_STATUS_NULL(OVMS_ModelsSettingsNew(&modelsSettings));

    ASSERT_CAPI_STATUS_NULL(OVMS_ModelsSettingsSetConfigPath(modelsSettings, "/mediapipe/mediapipe/examples/desktop/hello_ovms/config.json"));
    ASSERT_CAPI_STATUS_NULL(OVMS_ServerStartFromConfigurationFile(srv, serverSettings, modelsSettings));
}

int main(int argc, char** argv) {
    google::InitGoogleLogging(argv[0]);

    OVMS_Server* srv = nullptr;
    OVMS_ServerSettings* serverSettings = nullptr;
    OVMS_ModelsSettings* modelsSettings = nullptr;

    InitOvmsServer(srv, serverSettings, modelsSettings);

    CALL_AND_CHECK_RET(mediapipe::RunMediapipeGraph());

    OVMS_ModelsSettingsDelete(modelsSettings);
    OVMS_ServerSettingsDelete(serverSettings);
    OVMS_ServerDelete(srv);
    return 0;
}
