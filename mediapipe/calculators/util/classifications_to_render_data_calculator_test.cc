// Copyright 2019 The MediaPipe Authors.
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

#include "absl/memory/memory.h"
#include "mediapipe/calculators/util/classifications_to_render_data_calculator.pb.h"
#include "mediapipe/framework/calculator.pb.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/calculator_runner.h"
#include "mediapipe/framework/deps/message_matchers.h"
#include "mediapipe/framework/formats/classification.pb.h"
#include "mediapipe/framework/packet.h"
#include "mediapipe/framework/port/gmock.h"
#include "mediapipe/framework/port/gtest.h"
#include "mediapipe/framework/port/parse_text_proto.h"
#include "mediapipe/framework/port/status_matchers.h"
#include "mediapipe/util/color.pb.h"
#include "mediapipe/util/render_data.pb.h"

namespace mediapipe {

Classification CreateClassification(int32 index, float score,
                                    const std::string& label) {
  Classification classification;
  classification.set_score(score);
  classification.set_index(index);
  classification.set_label(label);
  return classification;
}

TEST(ClassificationsToRenderDataCalculatorTest, OnlyClassificationList) {
  CalculatorRunner runner(ParseTextProtoOrDie<CalculatorGraphConfig::Node>(R"(
    calculator: "ClassificationsToRenderDataCalculator"
    input_stream: "CLASSIFICATIONS:classifications"
    output_stream: "RENDER_DATA:render_data"
  )"));

  auto classifications(absl::make_unique<ClassificationList>());
  *(classifications->add_classification()) =
      CreateClassification(0, 0.9, "zeroth_label");
  *(classifications->add_classification()) = CreateClassification(1, 0.3, "");

  runner.MutableInputs()
      ->Tag("CLASSIFICATIONS")
      .packets.push_back(
          Adopt(classifications.release()).At(Timestamp::PostStream()));

  MP_ASSERT_OK(runner.Run()) << "Calculator execution failed.";
  const std::vector<Packet>& output =
      runner.Outputs().Tag("RENDER_DATA").packets;
  ASSERT_EQ(1, output.size());
  const auto& actual = output[0].Get<RenderData>();
  EXPECT_EQ(actual.render_annotations_size(), 2);
  // Labels
  EXPECT_EQ(actual.render_annotations(0).text().display_text(),
            "0.9 zeroth_label");
  EXPECT_EQ(actual.render_annotations(1).text().display_text(), "0.3 index=1");
}

}  // namespace mediapipe
