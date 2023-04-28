/* Copyright 2022 The MediaPipe Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include <cmath>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/strings/string_view.h"
#include "absl/strings/substitute.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/calculator_runner.h"
#include "mediapipe/framework/formats/classification.pb.h"
#include "mediapipe/framework/port/status_matchers.h"

namespace mediapipe {

namespace {

constexpr char kPredictionTag[] = "PREDICTION";

std::unique_ptr<CalculatorRunner> BuildNodeRunnerWithOptions(
    float drama_thresh, float llama_thresh, float bazinga_thresh,
    float joy_thresh, float peace_thresh) {
  constexpr absl::string_view kCalculatorProto = R"pb(
    calculator: "CombinedPredictionCalculator"
    input_stream: "custom_softmax_scores"
    input_stream: "canned_softmax_scores"
    output_stream: "PREDICTION:prediction"
    options {
      [mediapipe.CombinedPredictionCalculatorOptions.ext] {
        class { label: "CustomDrama" score_threshold: $0 }
        class { label: "CustomLlama" score_threshold: $1 }
        class { label: "CannedBazinga" score_threshold: $2 }
        class { label: "CannedJoy" score_threshold: $3 }
        class { label: "CannedPeace" score_threshold: $4 }
        background_label: "Negative"
      }
    }
  )pb";
  auto runner = std::make_unique<CalculatorRunner>(
      absl::Substitute(kCalculatorProto, drama_thresh, llama_thresh,
                       bazinga_thresh, joy_thresh, peace_thresh));
  return runner;
}

std::unique_ptr<ClassificationList> BuildCustomScoreInput(
    const float negative_score, const float drama_score,
    const float llama_score) {
  auto custom_scores = std::make_unique<ClassificationList>();
  auto custom_negative = custom_scores->add_classification();
  custom_negative->set_label("Negative");
  custom_negative->set_score(negative_score);
  auto drama = custom_scores->add_classification();
  drama->set_label("CustomDrama");
  drama->set_score(drama_score);
  auto llama = custom_scores->add_classification();
  llama->set_label("CustomLlama");
  llama->set_score(llama_score);
  return custom_scores;
}

std::unique_ptr<ClassificationList> BuildCannedScoreInput(
    const float negative_score, const float bazinga_score,
    const float joy_score, const float peace_score) {
  auto canned_scores = std::make_unique<ClassificationList>();
  auto canned_negative = canned_scores->add_classification();
  canned_negative->set_label("Negative");
  canned_negative->set_score(negative_score);
  auto bazinga = canned_scores->add_classification();
  bazinga->set_label("CannedBazinga");
  bazinga->set_score(bazinga_score);
  auto joy = canned_scores->add_classification();
  joy->set_label("CannedJoy");
  joy->set_score(joy_score);
  auto peace = canned_scores->add_classification();
  peace->set_label("CannedPeace");
  peace->set_score(peace_score);
  return canned_scores;
}

TEST(CombinedPredictionCalculatorPacketTest,
     CustomEmpty_CannedEmpty_ResultIsEmpty) {
  auto runner = BuildNodeRunnerWithOptions(
      /*drama_thresh=*/0.0, /*llama_thresh=*/0.0, /*bazinga_thresh=*/0.0,
      /*joy_thresh=*/0.0, /*peace_thresh=*/0.0);
  MP_ASSERT_OK(runner->Run()) << "Calculator execution failed.";
  EXPECT_THAT(runner->Outputs().Tag("PREDICTION").packets, testing::IsEmpty());
}

TEST(CombinedPredictionCalculatorPacketTest,
     CustomEmpty_CannedNotEmpty_ResultIsCanned) {
  auto runner = BuildNodeRunnerWithOptions(
      /*drama_thresh=*/0.0, /*llama_thresh=*/0.0, /*bazinga_thresh=*/0.9,
      /*joy_thresh=*/0.5, /*peace_thresh=*/0.8);
  auto canned_scores = BuildCannedScoreInput(
      /*negative_score=*/0.1,
      /*bazinga_score=*/0.1, /*joy_score=*/0.6, /*peace_score=*/0.2);
  runner->MutableInputs()->Index(1).packets.push_back(
      Adopt(canned_scores.release()).At(Timestamp(1)));
  MP_ASSERT_OK(runner->Run()) << "Calculator execution failed.";

  auto output_prediction_packets =
      runner->Outputs().Tag(kPredictionTag).packets;
  ASSERT_EQ(output_prediction_packets.size(), 1);
  Classification output_prediction =
      output_prediction_packets[0].Get<ClassificationList>().classification(0);

  EXPECT_EQ(output_prediction.label(), "CannedJoy");
  EXPECT_NEAR(output_prediction.score(), 0.6, 1e-4);
}

TEST(CombinedPredictionCalculatorPacketTest,
     CustomNotEmpty_CannedEmpty_ResultIsCustom) {
  auto runner = BuildNodeRunnerWithOptions(
      /*drama_thresh=*/0.3, /*llama_thresh=*/0.5, /*bazinga_thresh=*/0.0,
      /*joy_thresh=*/0.0, /*peace_thresh=*/0.0);
  auto custom_scores =
      BuildCustomScoreInput(/*negative_score=*/0.1,
                            /*drama_score=*/0.2, /*llama_score=*/0.7);
  runner->MutableInputs()->Index(0).packets.push_back(
      Adopt(custom_scores.release()).At(Timestamp(1)));
  MP_ASSERT_OK(runner->Run()) << "Calculator execution failed.";

  auto output_prediction_packets =
      runner->Outputs().Tag(kPredictionTag).packets;
  ASSERT_EQ(output_prediction_packets.size(), 1);
  Classification output_prediction =
      output_prediction_packets[0].Get<ClassificationList>().classification(0);

  EXPECT_EQ(output_prediction.label(), "CustomLlama");
  EXPECT_NEAR(output_prediction.score(), 0.7, 1e-4);
}

struct CombinedPredictionCalculatorTestCase {
  std::string test_name;
  float custom_negative_score;
  float drama_score;
  float llama_score;
  float drama_thresh;
  float llama_thresh;
  float canned_negative_score;
  float bazinga_score;
  float joy_score;
  float peace_score;
  float bazinga_thresh;
  float joy_thresh;
  float peace_thresh;
  std::string max_scoring_label;
  float max_score;
};

using CombinedPredictionCalculatorTest =
    testing::TestWithParam<CombinedPredictionCalculatorTestCase>;

TEST_P(CombinedPredictionCalculatorTest, OutputsCorrectResult) {
  const CombinedPredictionCalculatorTestCase& test_case = GetParam();

  auto runner = BuildNodeRunnerWithOptions(
      test_case.drama_thresh, test_case.llama_thresh, test_case.bazinga_thresh,
      test_case.joy_thresh, test_case.peace_thresh);

  auto custom_scores =
      BuildCustomScoreInput(test_case.custom_negative_score,
                            test_case.drama_score, test_case.llama_score);

  runner->MutableInputs()->Index(0).packets.push_back(
      Adopt(custom_scores.release()).At(Timestamp(1)));

  auto canned_scores = BuildCannedScoreInput(
      test_case.canned_negative_score, test_case.bazinga_score,
      test_case.joy_score, test_case.peace_score);
  runner->MutableInputs()->Index(1).packets.push_back(
      Adopt(canned_scores.release()).At(Timestamp(1)));

  MP_ASSERT_OK(runner->Run()) << "Calculator execution failed.";

  auto output_prediction_packets =
      runner->Outputs().Tag(kPredictionTag).packets;
  ASSERT_EQ(output_prediction_packets.size(), 1);
  Classification output_prediction =
      output_prediction_packets[0].Get<ClassificationList>().classification(0);

  EXPECT_EQ(output_prediction.label(), test_case.max_scoring_label);
  EXPECT_NEAR(output_prediction.score(), test_case.max_score, 1e-4);
}

INSTANTIATE_TEST_CASE_P(
    CombinedPredictionCalculatorTests, CombinedPredictionCalculatorTest,
    testing::ValuesIn<CombinedPredictionCalculatorTestCase>({
        {
            /* test_name= */ "TestCustomDramaWinnnerWith_HighCanned_Thresh",
            /* custom_negative_score= */ 0.1,
            /* drama_score= */ 0.5,
            /* llama_score= */ 0.3,
            /* drama_thresh= */ 0.25,
            /* llama_thresh= */ 0.7,
            /* canned_negative_score= */ 0.1,
            /* bazinga_score= */ 0.3,
            /* joy_score= */ 0.3,
            /* peace_score= */ 0.3,
            /* bazinga_thresh= */ 0.7,
            /* joy_thresh= */ 0.7,
            /* peace_thresh= */ 0.7,
            /* max_scoring_label= */ "CustomDrama",
            /* max_score= */ 0.5,
        },
        {
            /* test_name= */ "TestCannedWinnerWith_HighCustom_ZeroCanned_"
                             "Thresh",
            /* custom_negative_score= */ 0.1,
            /* drama_score= */ 0.3,
            /* llama_score= */ 0.6,
            /* drama_thresh= */ 0.4,
            /* llama_thresh= */ 0.8,
            /* canned_negative_score= */ 0.1,
            /* bazinga_score= */ 0.4,
            /* joy_score= */ 0.3,
            /* peace_score= */ 0.2,
            /* bazinga_thresh= */ 0.0,
            /* joy_thresh= */ 0.0,
            /* peace_thresh= */ 0.0,
            /* max_scoring_label= */ "CannedBazinga",
            /* max_score= */ 0.4,
        },
        {
            /* test_name= */ "TestNegativeWinnerWith_LowCustom_HighCanned_"
                             "Thresh",
            /* custom_negative_score= */ 0.5,
            /* drama_score= */ 0.1,
            /* llama_score= */ 0.4,
            /* drama_thresh= */ 0.1,
            /* llama_thresh= */ 0.05,
            /* canned_negative_score= */ 0.1,
            /* bazinga_score= */ 0.3,
            /* joy_score= */ 0.3,
            /* peace_score= */ 0.3,
            /* bazinga_thresh= */ 0.7,
            /* joy_thresh= */ 0.7,
            /* peace_thresh= */ 0.7,
            /* max_scoring_label= */ "Negative",
            /* max_score= */ 0.5,
        },
        {
            /* test_name= */ "TestNegativeWinnerWith_HighCustom_HighCanned_"
                             "Thresh",
            /* custom_negative_score= */ 0.8,
            /* drama_score= */ 0.1,
            /* llama_score= */ 0.1,
            /* drama_thresh= */ 0.25,
            /* llama_thresh= */ 0.7,
            /* canned_negative_score= */ 0.1,
            /* bazinga_score= */ 0.3,
            /* joy_score= */ 0.3,
            /* peace_score= */ 0.3,
            /* bazinga_thresh= */ 0.7,
            /* joy_thresh= */ 0.7,
            /* peace_thresh= */ 0.7,
            /* max_scoring_label= */ "Negative",
            /* max_score= */ 0.8,
        },
        {
            /* test_name= */ "TestNegativeWinnerWith_HighCustom_"
                             "HighCannedThresh2",
            /* custom_negative_score= */ 0.1,
            /* drama_score= */ 0.2,
            /* llama_score= */ 0.7,
            /* drama_thresh= */ 1.1,
            /* llama_thresh= */ 1.1,
            /* canned_negative_score= */ 0.1,
            /* bazinga_score= */ 0.3,
            /* joy_score= */ 0.3,
            /* peace_score= */ 0.3,
            /* bazinga_thresh= */ 0.7,
            /* joy_thresh= */ 0.7,
            /* peace_thresh= */ 0.7,
            /* max_scoring_label= */ "Negative",
            /* max_score= */ 0.1,
        },
        {
            /* test_name= */ "TestNegativeWinnerWith_HighCustom_HighCanned_"
                             "Thresh3",
            /* custom_negative_score= */ 0.1,
            /* drama_score= */ 0.3,
            /* llama_score= */ 0.6,
            /* drama_thresh= */ 0.4,
            /* llama_thresh= */ 0.8,
            /* canned_negative_score= */ 0.3,
            /* bazinga_score= */ 0.2,
            /* joy_score= */ 0.3,
            /* peace_score= */ 0.2,
            /* bazinga_thresh= */ 0.5,
            /* joy_thresh= */ 0.5,
            /* peace_thresh= */ 0.5,
            /* max_scoring_label= */ "Negative",
            /* max_score= */ 0.1,
        },
    }),
    [](const testing::TestParamInfo<
        CombinedPredictionCalculatorTest::ParamType>& info) {
      return info.param.test_name;
    });

}  // namespace

}  // namespace mediapipe
