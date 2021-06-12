// Copyright 2018 The MediaPipe Authors.
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

#include "mediapipe/framework/profiler/reporter/reporter.h"

#include <fcntl.h>
#include <unistd.h>

#include <memory>
#include <sstream>
#include <string>

#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "mediapipe/framework/calculator.pb.h"
#include "mediapipe/framework/calculator_profile.pb.h"
#include "mediapipe/framework/port/advanced_proto_inc.h"
#include "mediapipe/framework/port/gmock.h"
#include "mediapipe/framework/port/gtest.h"
#include "mediapipe/framework/port/proto_ns.h"
#include "mediapipe/framework/port/status_matchers.h"
#include "mediapipe/framework/profiler/reporter/statistic.h"
#include "mediapipe/framework/tool/test_util.h"

namespace mediapipe {

using mediapipe::reporter::Reporter;
using ::testing::ElementsAre;
using ::testing::HasSubstr;
using ::testing::IsSupersetOf;

void LoadGraphProfile(const std::string& path, GraphProfile* proto) {
  int fd = open(path.c_str(), O_RDONLY);
  if (fd == -1) {
    LOG(ERROR) << "could not open test graph: " << path
               << ", error: " << strerror(errno);
    return;
  }
  proto_ns::io::FileInputStream input(fd);
  bool success = proto->ParseFromZeroCopyStream(&input);
  close(fd);
  if (!success) {
    LOG(ERROR) << "could not parse test graph: " << path;
  }
}

std::unique_ptr<Reporter> loadReporter(const std::vector<std::string>& paths) {
  auto reporter = std::make_unique<Reporter>();
  for (const auto path : paths) {
    GraphProfile profile;
    LoadGraphProfile(
        absl::StrCat(GetTestDataDir("mediapipe/framework/profiler"), path),
        &profile);
    reporter->Accumulate(profile);
  }
  return reporter;
}

TEST(Reporter, Trivial) {
  auto reporter = loadReporter({"profile_opencv_0.binarypb"});
}

TEST(Reporter, ReportAllColumns) {
  auto reporter = loadReporter({"profile_opencv_0.binarypb"});
  auto report = reporter->Report();

  EXPECT_THAT(report->headers(),
              IsSupersetOf({"calculator", "time_mean", "time_stddev",
                            "time_total", "input_latency_mean",
                            "input_latency_stddev", "input_latency_total"}));
  EXPECT_EQ(report->headers()[0], "calculator");
}

// Reports only the "calculator" column and one additional column using no
// wildcards.
TEST(Reporter, ReportOneColumn) {
  auto reporter = loadReporter({"profile_opencv_0.binarypb"});
  MEDIAPIPE_CHECK_OK(reporter->set_columns({"time_mean"}));
  auto report = reporter->Report();
  EXPECT_THAT(report->headers(), ElementsAre("calculator", "time_mean"));
}

// Reports the "calculator" column and additional columns using wildcards.
// Notice the columns are alphabetical except for the first column, 'calendar'.
TEST(Reporter, ReportColumnsWithWildcards) {
  auto reporter = loadReporter({"profile_opencv_0.binarypb"});
  MEDIAPIPE_CHECK_OK(reporter->set_columns({"*_m??n", "*l?t*cy*"}));
  EXPECT_THAT(reporter->Report()->headers(),
              ElementsAre("calculator", "input_latency_mean", "time_mean",
                          "input_latency_stddev", "input_latency_total"));
}

TEST(Reporter, AggregatesAreRecorded) {
  auto reporter = loadReporter({"profile_opencv_0.binarypb"});
  MEDIAPIPE_CHECK_OK(reporter->set_columns({"time_*", "*latency*"}));
  const auto& report = reporter->Report();
  const auto& lines = report->lines();
  EXPECT_EQ(lines.size(), 3);
  EXPECT_THAT(lines[2],
              ElementsAre("OpenCvWriteTextCalculator", "13823.77", "100.00",
                          "5541.47", "1976799", "245.13", "464.27", "35054"));
}

TEST(Reporter, JoinsFiles) {
  auto reporter = loadReporter({
      "profile_opencv_0.binarypb",
      "profile_opencv_1.binarypb",
  });
  MEDIAPIPE_CHECK_OK(reporter->set_columns({"time_*", "*latency*"}));
  const auto& report = reporter->Report();
  const auto& lines = report->lines();
  EXPECT_EQ(lines.size(), 3);
  EXPECT_THAT(lines[2],
              ElementsAre("OpenCvWriteTextCalculator", "14707.77", "100.00",
                          "5630.52", "3000385", "237.50", "389.35", "48449"));
}

TEST(Reporter, PrintAllColumns) {
  auto reporter = loadReporter({"profile_opencv_0.binarypb"});
  auto report = reporter->Report();

  std::stringstream output;
  report->Print(output);
  output.seekp(0);

  std::string header;
  std::getline(output, header);
  EXPECT_THAT(header,
              AllOf(HasSubstr("calculator"), HasSubstr("input_latency_mean"),
                    HasSubstr("input_latency_stddev"),
                    HasSubstr("input_latency_total"), HasSubstr("time_mean"),
                    HasSubstr("time_stddev"), HasSubstr("time_total")));
}

TEST(Reporter, CanReportBadColumns) {
  auto reporter = loadReporter({"profile_opencv_0.binarypb"});
  auto result = reporter->set_columns({"il[leg]al"});
  EXPECT_EQ(result.code(), StatusCode::kInvalidArgument);
  EXPECT_EQ(result.message(), "Column 'il[leg]al' is invalid.\n");
}

TEST(Reporter, CanReportNonMatchingColumns) {
  auto reporter = loadReporter({"profile_opencv_0.binarypb"});
  auto result = reporter->set_columns({"time_*", "il[leg]al"});
  EXPECT_EQ(result.code(), StatusCode::kInvalidArgument);
  EXPECT_EQ(result.message(), "Column 'il[leg]al' is invalid.\n");
  // Should not affect active columns, which is currently still "*".
  auto report = reporter->Report();
  EXPECT_THAT(report->headers(),
              IsSupersetOf({"calculator", "time_mean", "time_stddev"}));
}

TEST(Reporter, BadPatternsIgnored) {
  auto reporter = loadReporter({"profile_opencv_0.binarypb"});
  auto result = reporter->set_columns({"time_mean", "il[leg]al", "^bad"});
  EXPECT_EQ(result.code(), StatusCode::kInvalidArgument);
  // Can report multiple errors at once, separated by newlines.
  EXPECT_EQ(result.message(),
            "Column 'il[leg]al' is invalid.\n"
            "Column '^bad' is invalid.\n");
  // Should not affect active columns, which is currently still "*".
  auto report = reporter->Report();
  EXPECT_THAT(report->headers(), ElementsAre("calculator", "time_mean"));
}

TEST(Reporter, NonMatchingColumnsIgnored) {
  auto reporter = loadReporter({"profile_opencv_0.binarypb"});
  auto result = reporter->set_columns({"koopa*"});
  EXPECT_EQ(result.code(), StatusCode::kInvalidArgument);
  EXPECT_EQ(result.message(), "Column 'koopa*' did not match any columns.\n");
}

// Tests a much simpler, fabricated log where results can easily be hand
// calculated.
TEST(Reporter, ProcessCalculatedCorrectly) {
  auto reporter = loadReporter({"profile_process_test.binarypb"});
  auto report = reporter->Report();
  EXPECT_THAT(report->calculator_data().at("ACalculator").time_percent,
              testing::DoubleEq(75));
  EXPECT_THAT(report->calculator_data().at("ACalculator").time_stat.mean(),
              testing::DoubleEq(450));
  EXPECT_THAT(report->calculator_data().at("ACalculator").time_stat.stddev(),
              testing::DoubleNear(70.71, 0.01));
  EXPECT_THAT(report->calculator_data().at("ACalculator").time_stat.total(),
              testing::DoubleEq(900));
  EXPECT_THAT(report->calculator_data().at("BCalculator").time_percent,
              testing::DoubleEq(25));
  EXPECT_THAT(report->calculator_data().at("BCalculator").time_stat.mean(),
              testing::DoubleEq(300));
  // BCalculator has only one data point, so stddev is zero.
  EXPECT_THAT(report->calculator_data().at("BCalculator").time_stat.stddev(),
              testing::DoubleEq(0));
  EXPECT_THAT(report->calculator_data().at("BCalculator").time_stat.total(),
              testing::DoubleEq(300));
}

TEST(Reporter, LatencyCalculatedCorrectly) {
  auto reporter = loadReporter({"profile_latency_test.binarypb"});
  auto report = reporter->Report();
  EXPECT_THAT(
      report->calculator_data().at("ACalculator").input_latency_stat.mean(),
      testing::DoubleEq(150));
  EXPECT_THAT(
      report->calculator_data().at("ACalculator").input_latency_stat.stddev(),
      testing::DoubleNear(70.71, 0.01));
  EXPECT_THAT(
      report->calculator_data().at("ACalculator").input_latency_stat.total(),
      testing::DoubleEq(300));
  EXPECT_THAT(
      report->calculator_data().at("BCalculator").input_latency_stat.mean(),
      testing::DoubleEq(750));
  EXPECT_THAT(
      report->calculator_data().at("BCalculator").input_latency_stat.stddev(),
      testing::DoubleNear(212.13, 0.01));
  EXPECT_THAT(
      report->calculator_data().at("BCalculator").input_latency_stat.total(),
      testing::DoubleEq(1500));
}

}  // namespace mediapipe
