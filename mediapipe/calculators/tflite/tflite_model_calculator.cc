// Copyright 2020 The MediaPipe Authors.
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

#include <functional>
#include <memory>
#include <string>

#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/packet.h"
#include "mediapipe/framework/port/ret_check.h"
#include "tensorflow/lite/model.h"

namespace mediapipe {

// Loads TfLite model from model blob specified as input side packet and outputs
// corresponding side packet.
//
// Input side packets:
//   MODEL_BLOB - TfLite model blob/file-contents (std::string). You can read
//                model blob from file (using whatever APIs you have) and pass
//                it to the graph as input side packet or you can use some of
//                calculators like LocalFileContentsCalculator to get model
//                blob and use it as input here.
//
// Output side packets:
//   MODEL - TfLite model. (std::unique_ptr<tflite::FlatBufferModel,
//           std::function<void(tflite::FlatBufferModel*)>>)
//
// Example use:
//
// node {
//   calculator: "TfLiteModelCalculator"
//   input_side_packet: "MODEL_BLOB:model_blob"
//   output_side_packet: "MODEL:model"
// }
//
class TfLiteModelCalculator : public CalculatorBase {
 public:
  using TfLiteModelPtr =
      std::unique_ptr<tflite::FlatBufferModel,
                      std::function<void(tflite::FlatBufferModel*)>>;

  static ::mediapipe::Status GetContract(CalculatorContract* cc) {
    cc->InputSidePackets().Tag("MODEL_BLOB").Set<std::string>();
    cc->OutputSidePackets().Tag("MODEL").Set<TfLiteModelPtr>();
    return ::mediapipe::OkStatus();
  }

  ::mediapipe::Status Open(CalculatorContext* cc) override {
    const Packet& model_packet = cc->InputSidePackets().Tag("MODEL_BLOB");
    const std::string& model_blob = model_packet.Get<std::string>();
    std::unique_ptr<tflite::FlatBufferModel> model =
        tflite::FlatBufferModel::BuildFromBuffer(model_blob.data(),
                                                 model_blob.size());
    RET_CHECK(model) << "Failed to load TfLite model from blob.";

    cc->OutputSidePackets().Tag("MODEL").Set(
        MakePacket<TfLiteModelPtr>(TfLiteModelPtr(
            model.release(), [model_packet](tflite::FlatBufferModel* model) {
              // Keeping model_packet in order to keep underlying model blob
              // which can be released only after TfLite model is not needed
              // anymore (deleted).
              delete model;
            })));

    return ::mediapipe::OkStatus();
  }

  ::mediapipe::Status Process(CalculatorContext* cc) override {
    return ::mediapipe::OkStatus();
  }
};
REGISTER_CALCULATOR(TfLiteModelCalculator);

}  // namespace mediapipe
