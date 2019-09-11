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

#include "mediapipe/framework/tool/validate_name.h"

#include "absl/strings/str_cat.h"
#include "absl/strings/substitute.h"
#include "mediapipe/framework/calculator.pb.h"
#include "mediapipe/framework/deps/message_matchers.h"
#include "mediapipe/framework/port/gmock.h"
#include "mediapipe/framework/port/gtest.h"
#include "mediapipe/framework/port/status_matchers.h"

namespace mediapipe {

namespace {

TEST(ValidateNameTest, ValidateName) {
  MP_EXPECT_OK(tool::ValidateName("humphrey"));
  MP_EXPECT_OK(tool::ValidateName("humphrey_bogart"));
  MP_EXPECT_OK(tool::ValidateName("humphrey_bogart_1899"));
  MP_EXPECT_OK(tool::ValidateName("aa"));
  MP_EXPECT_OK(tool::ValidateName("b1"));
  MP_EXPECT_OK(tool::ValidateName("_1"));
  EXPECT_FALSE(tool::ValidateName("").ok());
  EXPECT_FALSE(tool::ValidateName("humphrey bogart").ok());
  EXPECT_FALSE(tool::ValidateName("humphreyBogart").ok());
  EXPECT_FALSE(tool::ValidateName("humphrey-bogart").ok());
  EXPECT_FALSE(tool::ValidateName("humphrey/bogart").ok());
  EXPECT_FALSE(tool::ValidateName("humphrey.bogart").ok());
  EXPECT_FALSE(tool::ValidateName("humphrey:bogart").ok());
  EXPECT_FALSE(tool::ValidateName("1ST").ok());
  EXPECT_FALSE(tool::ValidateName("7_ELEVEN").ok());
  EXPECT_FALSE(tool::ValidateName("401K").ok());
  EXPECT_FALSE(tool::ValidateName("0").ok());
  EXPECT_FALSE(tool::ValidateName("1").ok());
  EXPECT_FALSE(tool::ValidateName("11").ok());
  EXPECT_FALSE(tool::ValidateName("92091").ok());
  EXPECT_FALSE(tool::ValidateName("1st").ok());
  EXPECT_FALSE(tool::ValidateName("7_eleven").ok());
  EXPECT_FALSE(tool::ValidateName("401k").ok());
  EXPECT_FALSE(tool::ValidateName("\0ContainsEscapes\t").ok());
}

TEST(ValidateNameTest, ValidateTag) {
  MP_EXPECT_OK(tool::ValidateTag("MALE"));
  MP_EXPECT_OK(tool::ValidateTag("MALE_ACTOR"));
  MP_EXPECT_OK(tool::ValidateTag("ACTOR_1899"));
  MP_EXPECT_OK(tool::ValidateTag("AA"));
  MP_EXPECT_OK(tool::ValidateTag("B1"));
  MP_EXPECT_OK(tool::ValidateTag("_1"));
  EXPECT_FALSE(tool::ValidateTag("").ok());
  EXPECT_FALSE(tool::ValidateTag("MALE ACTOR").ok());
  EXPECT_FALSE(tool::ValidateTag("MALEaCTOR").ok());
  EXPECT_FALSE(tool::ValidateTag("MALE-ACTOR").ok());
  EXPECT_FALSE(tool::ValidateTag("MALE/ACTOR").ok());
  EXPECT_FALSE(tool::ValidateTag("MALE.ACTOR").ok());
  EXPECT_FALSE(tool::ValidateTag("MALE:ACTOR").ok());
  EXPECT_FALSE(tool::ValidateTag("0").ok());
  EXPECT_FALSE(tool::ValidateTag("1").ok());
  EXPECT_FALSE(tool::ValidateTag("11").ok());
  EXPECT_FALSE(tool::ValidateTag("92091").ok());
  EXPECT_FALSE(tool::ValidateTag("1ST").ok());
  EXPECT_FALSE(tool::ValidateTag("7_ELEVEN").ok());
  EXPECT_FALSE(tool::ValidateTag("401K").ok());
  EXPECT_FALSE(tool::ValidateTag("\0ContainsEscapes\t").ok());
}

TEST(ValidateNameTest, ParseTagAndName) {
  std::string tag;
  std::string name;
  tag = "blah";
  name = "blah";
  MP_EXPECT_OK(tool::ParseTagAndName("MALE:humphrey", &tag, &name));
  EXPECT_EQ("MALE", tag);
  EXPECT_EQ("humphrey", name);
  tag = "blah";
  name = "blah";
  MP_EXPECT_OK(tool::ParseTagAndName("ACTOR:humphrey_bogart", &tag, &name));
  EXPECT_EQ("ACTOR", tag);
  EXPECT_EQ("humphrey_bogart", name);
  tag = "blah";
  name = "blah";
  MP_EXPECT_OK(tool::ParseTagAndName("ACTOR_1899:humphrey_1899", &tag, &name));
  EXPECT_EQ("ACTOR_1899", tag);
  EXPECT_EQ("humphrey_1899", name);
  tag = "blah";
  name = "blah";
  MP_EXPECT_OK(tool::ParseTagAndName("humphrey_bogart", &tag, &name));
  EXPECT_EQ("", tag);
  EXPECT_EQ("humphrey_bogart", name);

  tag = "blah";
  name = "blah";
  EXPECT_FALSE(tool::ParseTagAndName(":humphrey", &tag, &name).ok());
  EXPECT_EQ("", tag);
  EXPECT_EQ("", name);

  tag = "blah";
  name = "blah";
  EXPECT_FALSE(tool::ParseTagAndName("humphrey bogart", &tag, &name).ok());
  EXPECT_EQ("", tag);
  EXPECT_EQ("", name);

  tag = "blah";
  name = "blah";
  EXPECT_FALSE(tool::ParseTagAndName("actor:humphrey", &tag, &name).ok());
  EXPECT_EQ("", tag);
  EXPECT_EQ("", name);
  tag = "blah";
  name = "blah";
  MP_EXPECT_OK(tool::ParseTagAndName("ACTOR:humphrey", &tag, &name));
  EXPECT_EQ("ACTOR", tag);
  EXPECT_EQ("humphrey", name);

  tag = "blah";
  name = "blah";
  EXPECT_FALSE(tool::ParseTagAndName("ACTOR:HUMPHREY", &tag, &name).ok());
  EXPECT_EQ("", tag);
  EXPECT_EQ("", name);

  tag = "blah";
  name = "blah";
  EXPECT_FALSE(tool::ParseTagAndName("MALE:ACTOR:humphrey", &tag, &name).ok());
  EXPECT_EQ("", tag);
  EXPECT_EQ("", name);

  // Test various bad characters.
  for (std::string character : {" ", "-", "/", ".", ":"}) {
    tag = "blah";
    name = "blah";
    EXPECT_FALSE(
        tool::ParseTagAndName(absl::StrCat("MALE", character, "ACTOR:humphrey"),
                              &tag, &name)
            .ok());
    EXPECT_EQ("", tag);
    EXPECT_EQ("", name);
    tag = "blah";
    name = "blah";
    EXPECT_FALSE(
        tool::ParseTagAndName(
            absl::StrCat("ACTOR:humphrey", character, "bogart"), &tag, &name)
            .ok());
    EXPECT_EQ("", tag);
    EXPECT_EQ("", name);
  }
}

void TestPassParseTagIndexName(const std::string& tag_index_name,
                               const std::string& expected_tag,
                               const int expected_index,
                               const std::string& expected_name) {
  std::string actual_tag = "UNTOUCHED";
  int actual_index = -100;
  std::string actual_name = "untouched";
  MP_ASSERT_OK(tool::ParseTagIndexName(tag_index_name, &actual_tag,
                                       &actual_index, &actual_name))
      << "With tag_index_name " << tag_index_name;
  EXPECT_EQ(expected_tag, actual_tag)
      << "With tag_index_name " << tag_index_name;
  EXPECT_EQ(expected_index, actual_index)
      << "With tag_index_name " << tag_index_name;
  EXPECT_EQ(expected_name, actual_name)
      << "With tag_index_name " << tag_index_name;
}

void TestFailParseTagIndexName(const std::string& tag_index_name) {
  std::string actual_tag = "UNTOUCHED";
  int actual_index = -100;
  std::string actual_name = "untouched";
  ASSERT_FALSE(tool::ParseTagIndexName(tag_index_name, &actual_tag,
                                       &actual_index, &actual_name)
                   .ok())
      << "With tag_index_name " << tag_index_name;
  EXPECT_EQ("UNTOUCHED", actual_tag)
      << "With tag_index_name " << tag_index_name;
  EXPECT_EQ(-100, actual_index) << "With tag_index_name " << tag_index_name;
  EXPECT_EQ("untouched", actual_name)
      << "With tag_index_name " << tag_index_name;
}

TEST(ValidateNameTest, ParseTagIndexName) {
  // Success cases.
  // Test with tag.
  TestPassParseTagIndexName("MALE:humphrey", "MALE", 0, "humphrey");
  TestPassParseTagIndexName("ACTOR:humphrey_bogart", "ACTOR", 0,
                            "humphrey_bogart");
  TestPassParseTagIndexName("ACTOR_1899:humphrey_1899", "ACTOR_1899", 0,
                            "humphrey_1899");
  // Test without tag.
  TestPassParseTagIndexName("humphrey_bogart", "", -1, "humphrey_bogart");
  // Test with index.
  TestPassParseTagIndexName("ACTRESS:3:mieko_harada", "ACTRESS", 3,
                            "mieko_harada");
  TestPassParseTagIndexName("ACTRESS:0:mieko_harada", "ACTRESS", 0,
                            "mieko_harada");
  TestPassParseTagIndexName("A1:100:mieko1", "A1", 100, "mieko1");
  TestPassParseTagIndexName(
      absl::StrCat("A1:", ::mediapipe::internal::kMaxCollectionItemId,
                   ":mieko1"),
      "A1", ::mediapipe::internal::kMaxCollectionItemId, "mieko1");

  // Failure cases.
  TestFailParseTagIndexName("");    // Empty name.
  TestFailParseTagIndexName("A");   // Upper case name.
  TestFailParseTagIndexName("Aa");  // Upper case name.
  TestFailParseTagIndexName("aA");  // Upper case name.
  TestFailParseTagIndexName("1a");  // Name starts with number.
  TestFailParseTagIndexName("1");   // Name is number.
  // With tag.
  TestFailParseTagIndexName(":name");    // Missing tag.
  TestFailParseTagIndexName("A:");       // Missing name.
  TestFailParseTagIndexName("a:name");   // Lower case tag.
  TestFailParseTagIndexName("Aa:name");  // Lower case tag.
  TestFailParseTagIndexName("aA:name");  // Lower case tag.
  TestFailParseTagIndexName("1A:name");  // Tag starts with number.
  TestFailParseTagIndexName("1:name");   // Tag is number.
  // With index.
  TestFailParseTagIndexName("1:name");     // Missing tag.
  TestFailParseTagIndexName(":1:name");    // Missing tag.
  TestFailParseTagIndexName("A:1:");       // Missing name.
  TestFailParseTagIndexName("A::name");    // Missing index.
  TestFailParseTagIndexName("a:1:name");   // Lower case tag.
  TestFailParseTagIndexName("Aa:1:name");  // Lower case tag.
  TestFailParseTagIndexName("aA:1:name");  // Lower case tag.
  TestFailParseTagIndexName("1A:1:name");  // Tag starts with number.
  TestFailParseTagIndexName("1:1:name");   // Tag is number.
  TestFailParseTagIndexName("A:1:N");      // Upper case name.
  TestFailParseTagIndexName("A:1:nN");     // Upper case name.
  TestFailParseTagIndexName("A:1:Nn");     // Upper case name.
  TestFailParseTagIndexName("A:1:1name");  // Name starts with number.
  TestFailParseTagIndexName("A:1:1");      // Name is number.
  TestFailParseTagIndexName("A:-0:name");  // Negative index.
  TestFailParseTagIndexName("A:-1:name");  // Negative index.
  TestFailParseTagIndexName("A:01:name");  // Leading zero.
  TestFailParseTagIndexName("A:00:name");  // Leading zero.
  TestFailParseTagIndexName(
      absl::StrCat("A:", ::mediapipe::internal::kMaxCollectionItemId + 1,
                   ":a"));  // Too large an index.
  // Extra field
  TestFailParseTagIndexName("A:1:a:");   // extra field.
  TestFailParseTagIndexName(":A:1:a");   // extra field.
  TestFailParseTagIndexName("A:1:a:a");  // extra field.
  TestFailParseTagIndexName("A:1:a:A");  // extra field.
  TestFailParseTagIndexName("A:1:a:1");  // extra field.

  // Test various bad characters.
  for (char character : {'!', '@', '#', '$',  '%', '^', '&', '*', '(',  ')',
                         '{', '}', '[', ']',  '/', '=', '?', '+', '\\', '|',
                         '-', ';', ':', '\'', '"', ',', '<', '.', '>'}) {
    TestFailParseTagIndexName(absl::Substitute("$0", character));
    TestFailParseTagIndexName(absl::Substitute("$0a", character));
    TestFailParseTagIndexName(absl::Substitute("a$0", character));
    TestFailParseTagIndexName(absl::Substitute("$0:a", character));
    TestFailParseTagIndexName(absl::Substitute("A$0:a", character));
    TestFailParseTagIndexName(absl::Substitute("$0A:a", character));
    TestFailParseTagIndexName(absl::Substitute("A:$0:a", character));
    TestFailParseTagIndexName(absl::Substitute("A:$01:a", character));
    TestFailParseTagIndexName(absl::Substitute("A:1$0:a", character));
    TestFailParseTagIndexName(absl::Substitute("A:1:a$0", character));
    TestFailParseTagIndexName(absl::Substitute("$0A:1:a", character));
  }
}

void TestPassParseTagIndex(const std::string& tag_index,
                           const std::string& expected_tag,
                           const int expected_index) {
  std::string actual_tag = "UNTOUCHED";
  int actual_index = -100;
  MP_ASSERT_OK(tool::ParseTagIndex(tag_index, &actual_tag, &actual_index))
      << "With tag_index" << tag_index;
  EXPECT_EQ(expected_tag, actual_tag) << "With tag_index " << tag_index;
  EXPECT_EQ(expected_index, actual_index) << "With tag_index " << tag_index;
}

void TestFailParseTagIndex(const std::string& tag_index) {
  std::string actual_tag = "UNTOUCHED";
  int actual_index = -100;
  ASSERT_FALSE(tool::ParseTagIndex(tag_index, &actual_tag, &actual_index).ok())
      << "With tag_index " << tag_index;
  EXPECT_EQ("UNTOUCHED", actual_tag) << "With tag_index " << tag_index;
  EXPECT_EQ(-100, actual_index) << "With tag_index " << tag_index;
}

TEST(ValidateNameTest, ParseTagIndex) {
  // Success cases.
  TestPassParseTagIndex("", "", 0);
  TestPassParseTagIndex("VIDEO:0", "VIDEO", 0);
  TestPassParseTagIndex("VIDEO:1", "VIDEO", 1);
  TestPassParseTagIndex("AUDIO:2", "AUDIO", 2);
  TestPassParseTagIndex(":0", "", 0);
  TestPassParseTagIndex(":1", "", 1);
  TestPassParseTagIndex(":100", "", 100);

  // Failure cases.
  TestFailParseTagIndex("a");   // Lower case tag.
  TestFailParseTagIndex("Aa");  // Lower case tag.
  TestFailParseTagIndex("aA");  // Lower case tag.
  TestFailParseTagIndex("1A");  // tag starts with number.
  TestFailParseTagIndex("1");   // tag is number.
  // Two fields.
  TestFailParseTagIndex(":");     // Missing number.
  TestFailParseTagIndex(":a");    // lower case number.
  TestFailParseTagIndex(":A");    // upper case number.
  TestFailParseTagIndex(":-0");   // Negative index.
  TestFailParseTagIndex(":-1");   // Negative index.
  TestFailParseTagIndex(":01");   // Leading zero.
  TestFailParseTagIndex(":00");   // Leading zero.
  TestFailParseTagIndex("A:");    // Missing number.
  TestFailParseTagIndex("A:a");   // lower case number.
  TestFailParseTagIndex("A:A");   // upper case number.
  TestFailParseTagIndex("A:-0");  // Negative index.
  TestFailParseTagIndex("A:-1");  // Negative index.
  TestFailParseTagIndex("A:01");  // Leading zero.
  TestFailParseTagIndex("A:00");  // Leading zero.
  // Extra field
  TestFailParseTagIndex("A:1:");   // extra field.
  TestFailParseTagIndex(":A:1");   // extra field.
  TestFailParseTagIndex("A:1:2");  // extra field.
  TestFailParseTagIndex("A:A:1");  // extra field.

  // Test various bad characters.
  for (char character : {'!', '@', '#', '$',  '%', '^', '&', '*', '(',  ')',
                         '{', '}', '[', ']',  '/', '=', '?', '+', '\\', '|',
                         '-', ';', ':', '\'', '"', ',', '<', '.', '>'}) {
    TestFailParseTagIndex(absl::Substitute("$0", character));
    TestFailParseTagIndex(absl::Substitute("$0A", character));
    TestFailParseTagIndex(absl::Substitute("A$0", character));
    TestFailParseTagIndex(absl::Substitute("$0:1", character));
    TestFailParseTagIndex(absl::Substitute("A$0:1", character));
    TestFailParseTagIndex(absl::Substitute("$0A:1", character));
    TestFailParseTagIndex(absl::Substitute("A:1$0", character));
    TestFailParseTagIndex(absl::Substitute("A:$01", character));
  }
}

TEST(ValidateNameTest, GetTagAndNameInfo) {
  CalculatorGraphConfig::Node node_config1;
  CalculatorGraphConfig::Node node_config2;
  proto_ns::RepeatedPtrField<std::string>& fields =
      *node_config1.mutable_input_stream();
  proto_ns::RepeatedPtrField<std::string>& fields_copy =
      *node_config2.mutable_input_stream();

  // Single input using indexes.
  fields.Clear();
  fields.Add()->assign("transcoded_input_file");
  tool::TagAndNameInfo info;
  MP_ASSERT_OK(tool::GetTagAndNameInfo(fields, &info));
  ASSERT_EQ(0, info.tags.size());
  ASSERT_EQ(1, info.names.size());
  EXPECT_EQ(fields.Get(0), info.names[0]);
  MP_ASSERT_OK(tool::SetFromTagAndNameInfo(info, &fields_copy));
  EXPECT_THAT(node_config2, EqualsProto(node_config1));

  // Single input using tags.
  fields.Clear();
  fields.Add()->assign("FILE:transcoded_input_file");
  MP_ASSERT_OK(tool::GetTagAndNameInfo(fields, &info));
  ASSERT_EQ(1, info.tags.size());
  ASSERT_EQ(1, info.names.size());
  EXPECT_EQ("FILE", info.tags[0]);
  EXPECT_EQ("transcoded_input_file", info.names[0]);
  MP_ASSERT_OK(tool::SetFromTagAndNameInfo(info, &fields_copy));
  EXPECT_THAT(node_config2, EqualsProto(node_config1));

  // Mixing indexes and tags.
  fields.Clear();
  fields.Add()->assign("transcoded_input_file");
  fields.Add()->assign("FILE:transcoded_input_file");
  ASSERT_FALSE(tool::GetTagAndNameInfo(fields, &info).ok());

  // Valid configuration with more than one input using tags.
  fields.Clear();
  fields.Add()->assign("TAG1:input1");
  fields.Add()->assign("TAG2:input2");
  fields.Add()->assign("TAG3:input3");
  fields.Add()->assign("TAG4:input4");
  MP_ASSERT_OK(tool::GetTagAndNameInfo(fields, &info));
  ASSERT_EQ(4, info.tags.size());
  ASSERT_EQ(4, info.names.size());
  EXPECT_EQ("TAG1", info.tags[0]);
  EXPECT_EQ("TAG2", info.tags[1]);
  EXPECT_EQ("TAG3", info.tags[2]);
  EXPECT_EQ("TAG4", info.tags[3]);
  EXPECT_EQ("input1", info.names[0]);
  EXPECT_EQ("input2", info.names[1]);
  EXPECT_EQ("input3", info.names[2]);
  EXPECT_EQ("input4", info.names[3]);
  MP_ASSERT_OK(tool::SetFromTagAndNameInfo(info, &fields_copy));
  EXPECT_THAT(node_config2, EqualsProto(node_config1));

  // Valid configuration with more than one input using indexes.
  fields.Clear();
  fields.Add()->assign("input1");
  fields.Add()->assign("input2");
  fields.Add()->assign("input3");
  fields.Add()->assign("input4");
  MP_ASSERT_OK(tool::GetTagAndNameInfo(fields, &info));
  ASSERT_EQ(0, info.tags.size());
  ASSERT_EQ(4, info.names.size());
  EXPECT_EQ("input1", info.names[0]);
  EXPECT_EQ("input2", info.names[1]);
  EXPECT_EQ("input3", info.names[2]);
  EXPECT_EQ("input4", info.names[3]);
  MP_ASSERT_OK(tool::SetFromTagAndNameInfo(info, &fields_copy));
  EXPECT_THAT(node_config2, EqualsProto(node_config1));

  // Add an invalid character into the name.
  fields.Clear();
  fields.Add()->assign("TAG1:input1");
  fields.Add()->assign("TAG2:inv*alid");
  fields.Add()->assign("TAG3:input3");
  fields.Add()->assign("TAG4:input4");
  ASSERT_FALSE(tool::GetTagAndNameInfo(fields, &info).ok());

  // Add an invalid character into the tag.
  fields.Clear();
  fields.Add()->assign("TAG1:input1");
  fields.Add()->assign("INVA*LID:input2");
  fields.Add()->assign("TAG3:input3");
  fields.Add()->assign("TAG4:input4");
  ASSERT_FALSE(tool::GetTagAndNameInfo(fields, &info).ok());

  // Add an invalid character into the name and use indexes.
  fields.Clear();
  fields.Add()->assign("input1");
  fields.Add()->assign("inv*alid");
  fields.Add()->assign("input3");
  fields.Add()->assign("input4");
  ASSERT_FALSE(tool::GetTagAndNameInfo(fields, &info).ok());

  info.tags.clear();
  info.names.clear();
  info.names.push_back("a");
  info.tags.push_back("A");
  info.tags.push_back("B");
  ASSERT_FALSE(tool::SetFromTagAndNameInfo(info, &fields_copy).ok());

  info.names.push_back("b");
  info.names.push_back("c");
  ASSERT_FALSE(tool::SetFromTagAndNameInfo(info, &fields_copy).ok());

  info.tags.clear();
  info.names.clear();
  info.names.push_back("input1");
  info.names.push_back("inv*alid");
  info.names.push_back("input3");
  info.names.push_back("input4");
  ASSERT_FALSE(tool::SetFromTagAndNameInfo(info, &fields_copy).ok());

  info.tags.clear();
  info.names.clear();
  info.names.push_back("input1");
  info.names.push_back("input2");
  info.names.push_back("input3");
  info.names.push_back("input4");
  info.tags.push_back("INPUT1");
  info.tags.push_back("IN*VALID");
  info.tags.push_back("INPUT3");
  info.tags.push_back("INPUT4");
  ASSERT_FALSE(tool::SetFromTagAndNameInfo(info, &fields_copy).ok());
}

}  // namespace
}  // namespace mediapipe
