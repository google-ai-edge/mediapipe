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

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <utility>

#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/packet.h"
#include "mediapipe/framework/port/ret_check.h"
#include "tensorflow/lite/allocation.h"
#include "tensorflow/lite/model_builder.h"

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
//   MODEL_FD   - Tflite model file descriptor std::tuple<int, size_t, size_t>
//                containing (fd, offset, size).
//   MODEL_SPAN - TfLite model file contents in absl::Span<const uint8_t>, whose
//                underline buffer is owned outside of this calculator. User can
//                get the model span from a managed environment and pass it to
//                the graph as input side packet.
//
// Output side packets:
//   MODEL - TfLite model. (std::unique_ptr<tflite::FlatBufferModel,
//           std::function<void(tflite::FlatBufferModel*)>>)
//   SHARED_MODEL - TfLite model (std::shared_ptr<tflite::FlatBufferModel>) to
//           be shared by multiple downstream calculators.
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
  using SharedTfLiteModelPtr = std::shared_ptr<tflite::FlatBufferModel>;

  static constexpr absl::string_view kModelSpanTag = "MODEL_SPAN";
  static constexpr absl::string_view kModelBlobTag = "MODEL_BLOB";
  static constexpr absl::string_view kModelFDTag = "MODEL_FD";
  static constexpr absl::string_view kModelTag = "MODEL";
  static constexpr absl::string_view kSharedModelTag = "SHARED_MODEL";

  static absl::Status GetContract(CalculatorContract* cc) {
    if (cc->InputSidePackets().HasTag(kModelBlobTag)) {
      cc->InputSidePackets().Tag(kModelBlobTag).Set<std::string>();
    }

    if (cc->InputSidePackets().HasTag(kModelFDTag)) {
      cc->InputSidePackets()
          .Tag(kModelFDTag)
          .Set<std::tuple<int, size_t, size_t>>();
    }

    if (cc->InputSidePackets().HasTag(kModelSpanTag)) {
      cc->InputSidePackets()
          .Tag(kModelSpanTag)
          .Set<absl::Span<const uint8_t>>();
    }

    RET_CHECK(cc->OutputSidePackets().HasTag(kModelTag) ^
              cc->OutputSidePackets().HasTag(kSharedModelTag));

    if (cc->OutputSidePackets().HasTag(kModelTag)) {
      cc->OutputSidePackets().Tag(kModelTag).Set<TfLiteModelPtr>();
    } else if (cc->OutputSidePackets().HasTag(kSharedModelTag)) {
      cc->OutputSidePackets().Tag(kSharedModelTag).Set<SharedTfLiteModelPtr>();
    }

    return absl::OkStatus();
  }

  absl::Status Open(CalculatorContext* cc) override {
    Packet model_packet;
    std::unique_ptr<tflite::FlatBufferModel> model;

    if (cc->InputSidePackets().HasTag(kModelBlobTag)) {
      model_packet = cc->InputSidePackets().Tag(kModelBlobTag);
      const std::string& model_blob = model_packet.Get<std::string>();
      model = tflite::FlatBufferModel::BuildFromBuffer(model_blob.data(),
                                                       model_blob.size());
    }

    if (cc->InputSidePackets().HasTag(kModelSpanTag)) {
      model_packet = cc->InputSidePackets().Tag(kModelSpanTag);
      const absl::Span<const uint8_t>& model_view =
          model_packet.Get<absl::Span<const uint8_t>>();
      model = tflite::FlatBufferModel::BuildFromBuffer(
          reinterpret_cast<const char*>(model_view.data()), model_view.size());
    }

    if (cc->InputSidePackets().HasTag(kModelFDTag)) {
#if defined(ABSL_HAVE_MMAP) && !TFLITE_WITH_STABLE_ABI
      model_packet = cc->InputSidePackets().Tag(kModelFDTag);
      const auto& model_fd =
          model_packet.Get<std::tuple<int, size_t, size_t>>();
      auto model_allocation = std::make_unique<tflite::MMAPAllocation>(
          std::get<0>(model_fd), std::get<1>(model_fd), std::get<2>(model_fd),
          tflite::DefaultErrorReporter());
      model = tflite::FlatBufferModel::BuildFromAllocation(
          std::move(model_allocation), tflite::DefaultErrorReporter());
#else
      return absl::FailedPreconditionError(
          "Loading by file descriptor is not supported on this platform.");
#endif
    }

    RET_CHECK(model) << "Failed to load TfLite model.";

    TfLiteModelPtr output_model = TfLiteModelPtr(
        model.release(), [model_packet](tflite::FlatBufferModel* model) {
          // Keeping model_packet in order to keep underlying model blob
          // which can be released only after TfLite model is not needed
          // anymore (deleted).
          delete model;
        });
    if (cc->OutputSidePackets().HasTag(kModelTag)) {
      cc->OutputSidePackets().Tag(kModelTag).Set(
          MakePacket<TfLiteModelPtr>(std::move(output_model)));
    } else {
      cc->OutputSidePackets()
          .Tag(kSharedModelTag)
          .Set(MakePacket<SharedTfLiteModelPtr>(std::move(output_model)));
    }

    return absl::OkStatus();
  }

  absl::Status Process(CalculatorContext* cc) override {
    return absl::OkStatus();
  }
};
REGISTER_CALCULATOR(TfLiteModelCalculator);

}  // namespace mediapipe
