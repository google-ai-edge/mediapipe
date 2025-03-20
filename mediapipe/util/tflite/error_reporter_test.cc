#include "mediapipe/util/tflite/error_reporter.h"

#include <string>

#include "mediapipe/framework/port/gtest.h"

namespace mediapipe::util::tflite {
namespace {

TEST(ErrorReporterTest, ReportNoErrors) {
  ErrorReporter error_reporter;
  EXPECT_FALSE(error_reporter.HasError());
  EXPECT_TRUE(error_reporter.message().empty());
  EXPECT_TRUE(error_reporter.previous_message().empty());
}

TEST(ErrorReporterTest, ReportOneError) {
  ErrorReporter error_reporter;
  error_reporter.Report("error %i", 1);
  EXPECT_TRUE(error_reporter.HasError());
  EXPECT_EQ(error_reporter.message(), "error 1");
  EXPECT_TRUE(error_reporter.previous_message().empty());
}

TEST(ErrorReporterTest, ReportTwoErrors) {
  ErrorReporter error_reporter;
  error_reporter.Report("error %i", 1);
  error_reporter.Report("error %i", 2);
  EXPECT_TRUE(error_reporter.HasError());
  EXPECT_EQ(error_reporter.message(), "error 2");
  EXPECT_EQ(error_reporter.previous_message(), "error 1");
}

TEST(ErrorReporterTest, ReportThreeErrors) {
  ErrorReporter error_reporter;
  error_reporter.Report("error %i", 1);
  error_reporter.Report("error %i", 2);
  error_reporter.Report("error %i", 3);
  EXPECT_TRUE(error_reporter.HasError());
  EXPECT_EQ(error_reporter.message(), "error 3");
  EXPECT_EQ(error_reporter.previous_message(), "error 2");
}

TEST(ErrorReporterTest, VeryLongErrorIsTruncated) {
  ErrorReporter error_reporter;
  std::string long_error;
  long_error.resize(ErrorReporter::kBufferSize * 2, 'x');
  error_reporter.Report(long_error.c_str());
  EXPECT_TRUE(error_reporter.HasError());
  EXPECT_EQ(error_reporter.message(),
            long_error.substr(0, ErrorReporter::kBufferSize - 1));
}

}  // namespace
}  // namespace mediapipe::util::tflite
