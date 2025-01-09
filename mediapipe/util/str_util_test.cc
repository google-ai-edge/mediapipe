#include "mediapipe/util/str_util.h"

#include <string>
#include <vector>

#include "absl/strings/string_view.h"
#include "mediapipe/framework/port/gmock.h"
#include "mediapipe/framework/port/gtest.h"

namespace {

using ::testing::ElementsAreArray;
using ::testing::TestWithParam;

struct ForEachLineTestCase {
  std::string text;
  std::vector<std::string> expected_lines;
};

using ForEachLineTest = TestWithParam<ForEachLineTestCase>;

TEST_P(ForEachLineTest, ForEachLineWorks) {
  const absl::string_view text = GetParam().text;
  const std::vector<std::string>& expected_lines = GetParam().expected_lines;

  std::vector<absl::string_view> lines;
  mediapipe::ForEachLine(
      text, [&lines](absl::string_view line) { lines.push_back(line); });

  EXPECT_THAT(lines, ElementsAreArray(expected_lines));
}

INSTANTIATE_TEST_SUITE_P(ForEachLineTestSuiteInstantiation, ForEachLineTest,
                         testing::ValuesIn<ForEachLineTestCase>(
                             {[]() -> ForEachLineTestCase {
                                ForEachLineTestCase test_case;
                                test_case.text = "";
                                test_case.expected_lines = {};
                                return test_case;
                              }(),
                              []() -> ForEachLineTestCase {
                                ForEachLineTestCase test_case;
                                test_case.text =
                                    "line1\n"
                                    "line2\r"
                                    "line3\r\n"
                                    "line4\n"
                                    "\n";
                                test_case.expected_lines = {
                                    "line1", "line2", "line3", "line4", "",
                                };
                                return test_case;
                              }(),
                              []() -> ForEachLineTestCase {
                                ForEachLineTestCase test_case;
                                test_case.text =
                                    "\n"
                                    "\r"
                                    "\r\n"
                                    "\n"
                                    "\n";
                                test_case.expected_lines = {
                                    "", "", "", "", "",
                                };
                                return test_case;
                              }(),
                              []() -> ForEachLineTestCase {
                                ForEachLineTestCase test_case;
                                test_case.text =
                                    "\n"
                                    "\n"
                                    "\n"
                                    "\n"
                                    "\n";
                                test_case.expected_lines = {
                                    "", "", "", "", "",
                                };
                                return test_case;
                              }(),
                              []() -> ForEachLineTestCase {
                                ForEachLineTestCase test_case;
                                test_case.text =
                                    "\r"
                                    "\r"
                                    "\r"
                                    "\r"
                                    "\r";
                                test_case.expected_lines = {
                                    "", "", "", "", "",
                                };
                                return test_case;
                              }(),
                              []() -> ForEachLineTestCase {
                                ForEachLineTestCase test_case;
                                test_case.text =
                                    "\r\n"
                                    "\r\n"
                                    "\r\n"
                                    "\r\n"
                                    "\r\n";
                                test_case.expected_lines = {
                                    "", "", "", "", "",
                                };
                                return test_case;
                              }()}));

}  // namespace
