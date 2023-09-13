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

#include "mediapipe/framework/tool/tag_map.h"

#include "absl/log/absl_log.h"
#include "absl/strings/str_join.h"
#include "mediapipe/framework/port/gmock.h"
#include "mediapipe/framework/port/gtest.h"
#include "mediapipe/framework/port/map_util.h"
#include "mediapipe/framework/port/status_matchers.h"
#include "mediapipe/framework/tool/tag_map_helper.h"

namespace mediapipe {
namespace {

TEST(TagMapTest, Create) {
  // Create using tags.
  MP_EXPECT_OK(tool::CreateTagMapFromTags({}));
  MP_EXPECT_OK(tool::CreateTagMapFromTags({"BLAH"}));
  MP_EXPECT_OK(tool::CreateTagMapFromTags({"BLAH1", "BLAH2"}));
  // Tags must be uppercase.
  EXPECT_FALSE(tool::CreateTagMapFromTags({"blah1", "BLAH2"}).ok());

  // Create with TAG:<index>:names.
  MP_EXPECT_OK(tool::CreateTagMap({}));
  MP_EXPECT_OK(tool::CreateTagMap({"blah"}));
  MP_EXPECT_OK(tool::CreateTagMap({"blah1", "blah2"}));
  MP_EXPECT_OK(tool::CreateTagMap({"BLAH:blah"}));
  MP_EXPECT_OK(tool::CreateTagMap({"BLAH1:blah1", "BLAH2:blah2"}));
  MP_EXPECT_OK(tool::CreateTagMap({"BLAH:0:blah1", "BLAH:1:blah2"}));
  MP_EXPECT_OK(tool::CreateTagMap({"BLAH:blah1", "BLAH:1:blah2"}));
  MP_EXPECT_OK(tool::CreateTagMap(
      {"A:2:a2", "B:1:b1", "C:c0", "A:0:a0", "B:b0", "A:1:a1"}));
  MP_EXPECT_OK(tool::CreateTagMap({"w", "A:2:a2", "x", "B:1:b1", "C:c0", "y",
                                   "A:0:a0", "B:b0", "z", "A:1:a1"}));
  MP_EXPECT_OK(tool::CreateTagMap({"A:2:a2", "w", "x", "B:1:b1", "C:c0", "y",
                                   "A:0:a0", "B:b0", "z", "A:1:a1"}));

  // Reuse name.
  MP_EXPECT_OK(tool::CreateTagMap({"a", "A:a"}));
  // Reuse name.
  MP_EXPECT_OK(tool::CreateTagMap({"a", "a"}));
  // Reuse name.
  MP_EXPECT_OK(tool::CreateTagMap({"C:c", "a", "a"}));
  // Reuse name.
  MP_EXPECT_OK(tool::CreateTagMap({"A:a", "B:a"}));

  // Reuse same tag.
  EXPECT_FALSE(tool::CreateTagMap({"BLAH:blah1", "BLAH:blah2"}).ok());
  // Tag starts with number.
  EXPECT_FALSE(tool::CreateTagMap({"0:blah1", "1:blah2"}).ok());
  // Skipped index 0.
  EXPECT_FALSE(tool::CreateTagMap({"BLAH:1:blah1", "BLAH:2:blah2"}).ok());
  // Reuse index 0.
  EXPECT_FALSE(tool::CreateTagMap({"BLAH:blah1", "BLAH:0:blah2"}).ok());
  // Mixing tags and no tags.
  EXPECT_FALSE(
      tool::CreateTagMap({"blah0", "BLAH:1:blah1", "BLAH:2:blah2"}).ok());

  // Create using an index.
  MP_EXPECT_OK(tool::CreateTagMap(0));
  MP_EXPECT_OK(tool::CreateTagMap(3));
  // Negative number of entries.
  EXPECT_FALSE(tool::CreateTagMap(-1).ok());

  // Create using a TagAndNameInfo.
  tool::TagAndNameInfo info;
  info.names = {"blah1", "blah2"};
  MP_EXPECT_OK(tool::TagMap::Create(info));
  info.tags = {"BLAH1", "BLAH2", "BLAH3"};
  // Number of tags and names do not match.
  EXPECT_FALSE(tool::TagMap::Create(info).ok());
  info.names.push_back("blah3");
  MP_EXPECT_OK(tool::TagMap::Create(info));
}

void TestSuccessTagMap(const std::vector<std::string>& tag_index_names,
                       bool create_from_tags, int num_entries,
                       const std::vector<std::string>& tags,
                       const std::vector<std::string>& names) {
  std::shared_ptr<tool::TagMap> tag_map;
  if (create_from_tags) {
    tag_map = tool::CreateTagMapFromTags(tag_index_names).value();
  } else {
    tag_map = tool::CreateTagMap(tag_index_names).value();
  }

  EXPECT_EQ(num_entries, tag_map->NumEntries())
      << "Parameters: in " << tag_map->DebugString();
  EXPECT_EQ(tags.size(), tag_map->Mapping().size())
      << "Parameters: in " << tag_map->DebugString();
  for (int i = 0; i < tags.size(); ++i) {
    EXPECT_TRUE(tag_map->Mapping().contains(tags[i]))
        << "Parameters: Trying to find \"" << tags[i] << "\" in\n"
        << tag_map->DebugString();
  }
  EXPECT_THAT(tag_map->Names(), testing::ContainerEq(names));
}

TEST(TagMapTest, AccessFunctions) {
  TestSuccessTagMap({}, /*create_from_tags=*/true, 0, {}, {});
  TestSuccessTagMap({"BLAH"}, /*create_from_tags=*/true, 1, {"BLAH"},
                    {"name0"});
  TestSuccessTagMap({"BLAH1", "BLAH2"}, /*create_from_tags=*/true, 2,
                    {"BLAH1", "BLAH2"}, {"name0", "name1"});

  // Just names.
  TestSuccessTagMap({}, /*create_from_tags=*/false, 0, {}, {});
  TestSuccessTagMap({"blah"}, /*create_from_tags=*/false, 1, {""}, {"blah"});
  TestSuccessTagMap({"blah1", "blah2"}, /*create_from_tags=*/false, 2, {""},
                    {"blah1", "blah2"});

  // Just Tags.
  // 1 tag.
  TestSuccessTagMap({"BLAH:blah"}, /*create_from_tags=*/false, 1, {"BLAH"},
                    {"blah"});
  // 2 tags.
  TestSuccessTagMap({"BLAH1:blah1", "BLAH2:blah2"}, /*create_from_tags=*/false,
                    2, {"BLAH1", "BLAH2"}, {"blah1", "blah2"});
  TestSuccessTagMap({"BLAH2:blah2", "BLAH1:blah1"}, /*create_from_tags=*/false,
                    2, {"BLAH1", "BLAH2"}, {"blah1", "blah2"});

  // 1 Tag, 2 indexes.
  TestSuccessTagMap({"BLAH:0:blah1", "BLAH:1:blah2"},
                    /*create_from_tags=*/false, 2, {"BLAH"},
                    {"blah1", "blah2"});
  TestSuccessTagMap({"BLAH:blah1", "BLAH:1:blah2"},
                    /*create_from_tags=*/false, 2, {"BLAH"},
                    {"blah1", "blah2"});
  TestSuccessTagMap({"BLAH:1:blah2", "BLAH:blah1"},
                    /*create_from_tags=*/false, 2, {"BLAH"},
                    {"blah1", "blah2"});
  TestSuccessTagMap({"BLAH:1:blah2", "BLAH:0:blah1"},
                    /*create_from_tags=*/false, 2, {"BLAH"},
                    {"blah1", "blah2"});

  // Mixing tags with 1 index and more indexes.
  TestSuccessTagMap({"A:2:a2", "B:1:b1", "C:c0", "A:0:a0", "B:b0", "A:1:a1"},
                    /*create_from_tags=*/false, 6, {"A", "B", "C"},
                    {"a0", "a1", "a2", "b0", "b1", "c0"});
  // Mixing tags with 1 index and more indexes and positional names (tag "").
  TestSuccessTagMap({"w", "A:2:a2", "x", "B:1:b1", "C:c0", "y", "A:0:a0",
                     "B:b0", "z", "A:1:a1"},
                    /*create_from_tags=*/false, 10, {"", "A", "B", "C"},
                    {"w", "x", "y", "z", "a0", "a1", "a2", "b0", "b1", "c0"});
  // Same as previous, but change the order (so we don't start with "w").
  TestSuccessTagMap({"A:2:a2", "w", "x", "B:1:b1", "C:c0", "y", "A:0:a0",
                     "B:b0", "z", "A:1:a1"},
                    /*create_from_tags=*/false, 10, {"", "A", "B", "C"},
                    {"w", "x", "y", "z", "a0", "a1", "a2", "b0", "b1", "c0"});
}

TEST(TagMapTest, SameAs) {
  // A bunch of initialization vectors and their equivalence classes.
  // First argument is the equivalence class id.  Everything is SameAs()
  // everything else with the same id and different from (!SameAs())
  // everything in a different equivalence class.  Second argument is
  // whether the vector is of just tags (and not a full tag/index/name).
  int count = 0;
  std::vector<std::tuple<int, bool, std::vector<std::string>>>
      initialization_parameters = {
          std::make_tuple(++count, true, std::vector<std::string>({})),
          std::make_tuple(count, false, std::vector<std::string>({})),

          // One tag.
          std::make_tuple(++count, true, std::vector<std::string>({"BLAH"})),
          std::make_tuple(count, false,
                          std::vector<std::string>({"BLAH:blah"})),
          std::make_tuple(count, false,
                          std::vector<std::string>({"BLAH:different"})),

          // Two tags.
          std::make_tuple(++count, true,
                          std::vector<std::string>({"BLAH1", "BLAH2"})),
          std::make_tuple(
              count, false,
              std::vector<std::string>({"BLAH1:blah1", "BLAH2:blah2"})),
          std::make_tuple(count, false,
                          std::vector<std::string>({"BLAH1:a", "BLAH2:b"})),
          std::make_tuple(count, false,
                          std::vector<std::string>({"BLAH2:a", "BLAH1:b"})),

          // Two (different) tags.
          std::make_tuple(++count, true,
                          std::vector<std::string>({"BLAH2", "BLAH3"})),
          std::make_tuple(
              count, false,
              std::vector<std::string>({"BLAH2:blah1", "BLAH3:blah2"})),
          std::make_tuple(count, false,
                          std::vector<std::string>({"BLAH3:a", "BLAH2:b"})),

          // Three tags.
          std::make_tuple(++count, true,
                          std::vector<std::string>({"A", "B", "C"})),
          std::make_tuple(
              count, false,
              std::vector<std::string>({"B:blah1", "A:blah3", "C:blah2"})),
          std::make_tuple(count, false,
                          std::vector<std::string>({"C:a", "A:b", "B:c"})),

          // 2 indexes.
          std::make_tuple(++count, false, std::vector<std::string>({"a", "b"})),
          std::make_tuple(count, false, std::vector<std::string>({"c", "d"})),
          std::make_tuple(count, false, std::vector<std::string>({"a", "d"})),
          std::make_tuple(count, false, std::vector<std::string>({"d", "a"})),

          // 3 indexes (switch with different sort orders).
          std::make_tuple(++count, false,
                          std::vector<std::string>({"a", "b", "c"})),
          std::make_tuple(count, false,
                          std::vector<std::string>({"c", "b", "a"})),
          std::make_tuple(count, false,
                          std::vector<std::string>({"d", "e", "f"})),
          std::make_tuple(count, false,
                          std::vector<std::string>({"a", "b", "f"})),
          std::make_tuple(count, false,
                          std::vector<std::string>({"f", "e", "d"})),
          std::make_tuple(count, false,
                          std::vector<std::string>({"f", "c", "d"})),

          // 1 Tag, 2 indexes.
          std::make_tuple(
              ++count, false,
              std::vector<std::string>({"BLAH:0:blah1", "BLAH:1:blah2"})),
          std::make_tuple(
              count, false,
              std::vector<std::string>({"BLAH:blah1", "BLAH:1:blah2"})),
          std::make_tuple(
              count, false,
              std::vector<std::string>({"BLAH:1:blah1", "BLAH:0:blah2"})),
          std::make_tuple(
              count, false,
              std::vector<std::string>({"BLAH:1:blah1", "BLAH:blah2"})),
          std::make_tuple(count, false,
                          std::vector<std::string>({"BLAH:1:a", "BLAH:b"})),

          // Mixing tags with 1 index and more indexes.
          std::make_tuple(
              ++count, false,
              std::vector<std::string>(
                  {"A:2:a2", "B:1:b1", "C:c0", "A:0:a0", "B:b0", "A:1:a1"})),
          // Reordered.
          std::make_tuple(
              count, false,
              std::vector<std::string>(
                  {"A:0:a0", "A:2:a2", "A:1:a1", "B:1:b1", "C:c0", "B:b0"})),
          // Renamed names.
          std::make_tuple(count, false,
                          std::vector<std::string>({"A:0:a", "A:2:b", "A:1:c",
                                                    "B:1:d", "C:e", "B:f"})),
          // Change which strings have index 0 specified.
          std::make_tuple(
              count, false,
              std::vector<std::string>(
                  {"A:a", "A:2:b", "A:1:c", "B:1:d", "C:0:e", "B:0:f"})),

          // Mixing tags with 1 index and more indexes and positional
          // names (tag "").
          std::make_tuple(
              ++count, false,
              std::vector<std::string>({"w", "A:2:a2", "x", "B:1:b1", "C:c0",
                                        "y", "A:0:a0", "B:b0", "z", "A:1:a1"})),
          // Reordered.
          std::make_tuple(
              count, false,
              std::vector<std::string>({"C:c0", "A:1:a1", "y", "A:0:a0", "w",
                                        "A:2:a2", "x", "B:1:b1", "B:b0", "z"})),
          // Rename names.
          std::make_tuple(
              count, false,
              std::vector<std::string>({"C:a", "A:1:b", "c", "A:2:d", "e",
                                        "B:1:f", "g", "A:0:h", "B:i", "j"})),
          // Change which strings have index 0 specified.
          std::make_tuple(
              count, false,
              std::vector<std::string>({"C:0:a", "A:1:b", "c", "A:2:d", "e",
                                        "B:1:f", "g", "A:h", "B:0:i", "j"})),
      };

  // Create a TagMap for each entry in initialization_parameters.
  std::vector<std::shared_ptr<tool::TagMap>> tag_maps;
  for (const auto& parameters : initialization_parameters) {
    if (std::get<1>(parameters)) {
      auto statusor_tag_map =
          tool::CreateTagMapFromTags(std::get<2>(parameters));
      MP_ASSERT_OK(statusor_tag_map);
      tag_maps.push_back(std::move(statusor_tag_map.value()));
    } else {
      auto statusor_tag_map = tool::CreateTagMap(std::get<2>(parameters));
      MP_ASSERT_OK(statusor_tag_map);
      tag_maps.push_back(std::move(statusor_tag_map.value()));
    }
  }

  // Check every TagMap against every other (in both orders).
  for (int i = 0; i < initialization_parameters.size(); ++i) {
    int equivalence = std::get<0>(initialization_parameters[i]);
    for (int k = 0; k < initialization_parameters.size(); ++k) {
      EXPECT_EQ(std::get<0>(initialization_parameters[k]) == equivalence,
                tag_maps[i]->SameAs(*tag_maps[k]))
          << "ShortDebugStrings i, k\n"
          << tag_maps[i]->ShortDebugString() << "\n"
          << tag_maps[k]->ShortDebugString() << "\nDebugString tag_maps[i]\n"
          << tag_maps[i]->DebugString() << "\nDebugString tag_maps[k]\n"
          << tag_maps[k]->DebugString();
    }
  }
}

// A helper function to test that a TagMap's debug string and short
// debug string each satisfy a matcher.
template <typename Matcher>
void TestDebugString(
    const absl::StatusOr<std::shared_ptr<tool::TagMap>>& statusor_tag_map,
    const std::vector<std::string>& canonical_entries,
    Matcher short_string_matcher) {
  MP_ASSERT_OK(statusor_tag_map);
  tool::TagMap& tag_map = *statusor_tag_map.value();
  std::string debug_string = tag_map.DebugString();
  std::string short_string = tag_map.ShortDebugString();
  ABSL_LOG(INFO) << "ShortDebugString:\n" << short_string << "\n";
  ABSL_LOG(INFO) << "DebugString:\n" << debug_string << "\n\n";

  std::vector<std::string> actual_entries;
  for (const auto& field : tag_map.CanonicalEntries()) {
    actual_entries.push_back(field);
  }
  EXPECT_THAT(actual_entries, testing::ContainerEq(canonical_entries));
  if (canonical_entries.empty()) {
    EXPECT_THAT(debug_string, testing::Eq("empty"));
  } else {
    EXPECT_THAT(debug_string,
                testing::Eq(absl::StrJoin(canonical_entries, "\n")));
  }
  EXPECT_THAT(short_string, short_string_matcher);
}

TEST(TagMapTest, DebugStrings) {
  // The ContainsRegex test checks a tag and a number (of indexes)
  // appear together.
  // For example: testing::ContainsRegex("\"BLAH\"[^\\d]+\\b2\\b") tests
  // that "BLAH" is followed by the number 2 (with no numbers in between)
  // and that the number 2 is surrounded by word breaks (\b).

  // In addition to testing the tag name and the number of indexes in it,
  // the presence of each stream name is tested (that it exists in the
  // DebugString() and doesn't in the ShortDebugString()).
  TestDebugString(
      // The TagMap to test.
      tool::CreateTagMap({"BLAH:blah1", "BLAH:1:blah2"}),
      // Canonical Entries (used to test DebugString() too.
      {"BLAH:0:blah1", "BLAH:1:blah2"},
      // Must be satisfied by ShortDebugString().
      testing::AllOf(testing::ContainsRegex("\"BLAH\"[^\\d]+\\b2\\b"),
                     testing::Not(testing::HasSubstr("\"blah1\"")),
                     testing::Not(testing::HasSubstr("\"blah2\""))));

  TestDebugString(tool::CreateTagMap({"A:a", "B:b"}), {"A:a", "B:b"},
                  testing::AllOf(testing::ContainsRegex("\"A\"[^\\d]+\\b1\\b"),
                                 testing::ContainsRegex("\"B\"[^\\d]+\\b1\\b"),
                                 testing::Not(testing::HasSubstr("\"a\"")),
                                 testing::Not(testing::HasSubstr("\"b\""))));
  TestDebugString(tool::CreateTagMap({"B:b", "A:a"}), {"A:a", "B:b"},
                  testing::AllOf(testing::ContainsRegex("\"A\"[^\\d]+\\b1\\b"),
                                 testing::ContainsRegex("\"B\"[^\\d]+\\b1\\b"),
                                 testing::Not(testing::HasSubstr("\"a\"")),
                                 testing::Not(testing::HasSubstr("\"b\""))));
  TestDebugString(tool::CreateTagMap({"a", "b"}), {"a", "b"},
                  testing::AllOf(testing::ContainsRegex("\"\"[^\\d]+\\b2\\b"),
                                 testing::Not(testing::HasSubstr("\"a\"")),
                                 testing::Not(testing::HasSubstr("\"b\""))));
  TestDebugString(tool::CreateTagMap({"b", "a"}), {"b", "a"},
                  testing::AllOf(testing::ContainsRegex("\"\"[^\\d]+\\b2\\b"),
                                 testing::Not(testing::HasSubstr("\"a\"")),
                                 testing::Not(testing::HasSubstr("\"b\""))));
  TestDebugString(tool::CreateTagMap(3), {"name0", "name1", "name2"},
                  testing::ContainsRegex("\"\"[^\\d]+\\b3\\b"));
  TestDebugString(tool::CreateTagMap(
                      {"A:2:a2", "B:1:b1", "C:c0", "A:0:a0", "B:b0", "A:1:a1"}),
                  {"A:0:a0", "A:1:a1", "A:2:a2", "B:0:b0", "B:1:b1", "C:c0"},
                  testing::AllOf(testing::ContainsRegex("\"A\"[^\\d]+\\b3\\b"),
                                 testing::ContainsRegex("\"B\"[^\\d]+\\b2\\b"),
                                 testing::ContainsRegex("\"C\"[^\\d]+\\b1\\b"),
                                 testing::Not(testing::HasSubstr("\"a0\"")),
                                 testing::Not(testing::HasSubstr("\"a1\"")),
                                 testing::Not(testing::HasSubstr("\"a2\"")),
                                 testing::Not(testing::HasSubstr("\"b0\"")),
                                 testing::Not(testing::HasSubstr("\"b1\"")),
                                 testing::Not(testing::HasSubstr("\"c0\""))));
  TestDebugString(tool::CreateTagMap({"A:2:a2", "x", "B:1:b1", "C:c0", "y",
                                      "A:0:a0", "B:b0", "z", "A:1:a1", "w"}),
                  {"x", "y", "z", "w", "A:0:a0", "A:1:a1", "A:2:a2", "B:0:b0",
                   "B:1:b1", "C:c0"},
                  testing::AllOf(testing::ContainsRegex("\"\"[^\\d]+\\b4\\b"),
                                 testing::ContainsRegex("\"A\"[^\\d]+\\b3\\b"),
                                 testing::ContainsRegex("\"B\"[^\\d]+\\b2\\b"),
                                 testing::ContainsRegex("\"C\"[^\\d]+\\b1\\b"),
                                 testing::Not(testing::HasSubstr("\"w\"")),
                                 testing::Not(testing::HasSubstr("\"x\"")),
                                 testing::Not(testing::HasSubstr("\"y\"")),
                                 testing::Not(testing::HasSubstr("\"z\"")),
                                 testing::Not(testing::HasSubstr("\"a0\"")),
                                 testing::Not(testing::HasSubstr("\"a1\"")),
                                 testing::Not(testing::HasSubstr("\"a2\"")),
                                 testing::Not(testing::HasSubstr("\"b0\"")),
                                 testing::Not(testing::HasSubstr("\"b1\"")),
                                 testing::Not(testing::HasSubstr("\"c0\""))));

  // Test that empty TagMap states "empty" as its DebugString and
  // ShortDebugString().
  TestDebugString(tool::CreateTagMap(0), {},
                  testing::ContainsRegex("\\bempty\\b"));
  TestDebugString(tool::CreateTagMap({}), {},
                  testing::ContainsRegex("\\bempty\\b"));
  TestDebugString(tool::CreateTagMapFromTags({}), {},
                  testing::ContainsRegex("\\bempty\\b"));

  // Test that TagAndNameInfo can be used as well.
  tool::TagAndNameInfo info;
  info.names = {"blah1", "blah2", "blah3"};
  info.tags = {"BLAH1", "BLAH2", "BLAH3"};
  TestDebugString(
      tool::TagMap::Create(info), {"BLAH1:blah1", "BLAH2:blah2", "BLAH3:blah3"},
      testing::AllOf(testing::ContainsRegex("\"BLAH1\"[^\\d]+\\b1\\b"),
                     testing::ContainsRegex("\"BLAH2\"[^\\d]+\\b1\\b"),
                     testing::ContainsRegex("\"BLAH3\"[^\\d]+\\b1\\b"),
                     testing::Not(testing::HasSubstr("\"blah1\"")),
                     testing::Not(testing::HasSubstr("\"blah2\"")),
                     testing::Not(testing::HasSubstr("\"blah3\""))));
}

}  // namespace
}  // namespace mediapipe
