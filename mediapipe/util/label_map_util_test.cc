#include "mediapipe/util/label_map_util.h"

#include <string>
#include <vector>

#include "absl/status/status.h"
#include "absl/strings/str_join.h"
#include "absl/strings/string_view.h"
#include "mediapipe/framework/port/gmock.h"
#include "mediapipe/framework/port/gtest.h"
#include "mediapipe/framework/port/status_matchers.h"

namespace mediapipe {
namespace {

TEST(BuildLabelMapFromFiles, WorksForLables) {
  std::vector<std::string> labels = {"label1", "label2", "label3"};
  std::string labels_file_contents = absl::StrJoin(labels, "\n");

  MP_ASSERT_OK_AND_ASSIGN(
      auto map, BuildLabelMapFromFiles(labels_file_contents,
                                       /*display_names_file_contents*/ {}));
  ASSERT_EQ(map.size(), labels.size());
  for (int i = 0; i < labels.size(); ++i) {
    auto label = map.find(i);
    ASSERT_NE(label, map.end());
    EXPECT_EQ(label->second.name(), labels[i]);
  }
}

TEST(BuildLabelMapFromFiles, WorksForLablesWithContentsEmptyLineAtTheEnd) {
  std::vector<std::string> labels = {"label1", "label2", "label3", ""};
  std::string labels_file_contents = absl::StrJoin(labels, "\n");

  MP_ASSERT_OK_AND_ASSIGN(
      auto map, BuildLabelMapFromFiles(labels_file_contents,
                                       /*display_names_file_contents*/ {}));
  ASSERT_EQ(map.size(), labels.size() - 1);
  for (int i = 0; i < labels.size() - 1; ++i) {
    auto label = map.find(i);
    ASSERT_NE(label, map.end());
    EXPECT_EQ(label->second.name(), labels[i]);
  }
}

TEST(BuildLabelMapFromFiles, WorksForLablesAndDisplayNames) {
  std::vector<std::string> labels = {"label1", "label2", "label3"};
  std::string labels_file_contents = absl::StrJoin(labels, "\n");
  std::vector<std::string> display_names = {"display_name1", "display_name2",
                                            "display_name3"};
  std::string display_names_file_contents = absl::StrJoin(display_names, "\n");

  MP_ASSERT_OK_AND_ASSIGN(auto map,
                          BuildLabelMapFromFiles(labels_file_contents,
                                                 display_names_file_contents));
  ASSERT_EQ(map.size(), labels.size());
  for (int i = 0; i < labels.size(); ++i) {
    auto label = map.find(i);
    ASSERT_NE(label, map.end());
    EXPECT_EQ(label->second.name(), labels[i]);
    EXPECT_EQ(label->second.display_name(), display_names[i]);
  }
}

TEST(BuildLabelMapFromFiles,
     WorksForLablesAndDisplayNamesWithContentsEmptyLineAtTheEnd) {
  std::vector<std::string> labels = {"label1", "label2", "label3"};
  std::string labels_file_contents = absl::StrJoin(labels, "\n");
  std::vector<std::string> display_names = {"display_name1", "display_name2",
                                            "display_name3", ""};
  std::string display_names_file_contents = absl::StrJoin(display_names, "\n");

  MP_ASSERT_OK_AND_ASSIGN(auto map,
                          BuildLabelMapFromFiles(labels_file_contents,
                                                 display_names_file_contents));
  ASSERT_EQ(map.size(), labels.size());
  for (int i = 0; i < labels.size(); ++i) {
    auto label = map.find(i);
    ASSERT_NE(label, map.end());
    EXPECT_EQ(label->second.name(), labels[i]);
    EXPECT_EQ(label->second.display_name(), display_names[i]);
  }
}

TEST(BuildLabelMapFromFiles, HandlesInvalidArguments) {
  EXPECT_THAT(BuildLabelMapFromFiles(
                  absl::StrJoin({"label1"}, "\n"),
                  absl::StrJoin({"display_name1", "display_name2"}, "\n")),
              mediapipe::StatusIs(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(BuildLabelMapFromFiles(absl::StrJoin({"label1", "label2"}, "\n"),
                                     absl::StrJoin({"display_name1"}, "\n")),
              mediapipe::StatusIs(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(BuildLabelMapFromFiles(absl::StrJoin({}, "\n"),
                                     absl::StrJoin({"display_name1"}, "\n")),
              mediapipe::StatusIs(absl::StatusCode::kInvalidArgument));
}

}  // namespace
}  // namespace mediapipe
